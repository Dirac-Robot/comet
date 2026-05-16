"""LangChain chat wrapper for Claude Code OAuth/subscription auth.

Claude Code owns the browser login and token refresh lifecycle. This wrapper
reuses that login by invoking ``claude -p`` in print mode, matching the
OpenClaw-style CLI backend instead of trying to copy private OAuth tokens into
the Anthropic SDK.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


DEFAULT_TIMEOUT_S = 600.0
CLAUDE_CODE_CLEAR_ENV = (
    'ANTHROPIC_API_KEY',
    'ANTHROPIC_API_KEY_OLD',
    'ANTHROPIC_API_TOKEN',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
    'ANTHROPIC_CUSTOM_HEADERS',
    'ANTHROPIC_OAUTH_TOKEN',
    'ANTHROPIC_UNIX_SOCKET',
    'CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR',
    'CLAUDE_CODE_OAUTH_REFRESH_TOKEN',
    'CLAUDE_CODE_OAUTH_SCOPES',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR',
    'CLAUDE_CODE_USE_BEDROCK',
    'CLAUDE_CODE_USE_FOUNDRY',
    'CLAUDE_CODE_USE_VERTEX',
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get('name') or tool.get('function', {}).get('name') or '')
    return str(getattr(tool, 'name', '') or '')


def _tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get('description') or tool.get('function', {}).get('description') or '')
    return str(getattr(tool, 'description', '') or '')


def _tool_parameters(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        params = tool.get('parameters') or tool.get('function', {}).get('parameters')
        return params if isinstance(params, dict) else {'type': 'object', 'properties': {}}

    args_schema = getattr(tool, 'args_schema', None)
    if args_schema is not None:
        try:
            if hasattr(args_schema, 'model_json_schema'):
                schema = args_schema.model_json_schema()
            elif hasattr(args_schema, 'schema'):
                schema = args_schema.schema()
            else:
                schema = {}
            if isinstance(schema, dict):
                return schema
        except Exception:
            pass

    try:
        args = getattr(tool, 'args', None)
    except Exception:
        args = None
    if isinstance(args, dict):
        return {'type': 'object', 'properties': args}
    return {'type': 'object', 'properties': {}}


def _tool_manifest(tools: tuple[Any, ...]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for tool in tools:
        name = _tool_name(tool)
        if not name:
            continue
        manifest.append({
            'name': name,
            'description': _tool_description(tool),
            'parameters': _tool_parameters(tool),
        })
    return manifest


def _anthropic_tool_manifest(tools: tuple[Any, ...]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for item in _tool_manifest(tools):
        manifest.append({
            'name': item['name'],
            'description': item.get('description', ''),
            'input_schema': item.get('parameters') or {'type': 'object', 'properties': {}},
        })
    return manifest


def _tool_use_prompt(tools: tuple[Any, ...]) -> str:
    manifest = _tool_manifest(tools)
    return (
        'CoBrA tool request envelope (strict):\n'
        'The tools listed below are executed only by the CoBrA host after your '
        'plain assistant text is parsed. They are not Claude Code built-in '
        'tools. Never try to call Task, Bash, Read, Edit, Write, or any native '
        'Claude Code/MCP tool, and never emit Anthropic content blocks such as '
        '{"type":"tool_use"}.\n'
        'If a listed CoBrA tool is needed, it is available through the envelope '
        'below even if Claude Code would reject its own tool channel. Do not '
        'say "No such tool available", "tool layer refused", or similar; emit '
        'the envelope instead.\n'
        'When you need a CoBrA tool, respond with ONLY this JSON object shape:\n'
        '{"content":"brief note","cobra_tool_calls":[{"name":"tool_name",'
        '"args":{},"id":"cobra_call_short_unique_id"}]}\n'
        'Use args for tool parameters. For multiple tool calls, add multiple '
        'objects to cobra_tool_calls. Use valid JSON, no Markdown fences, and '
        'call only listed tool names. The JSON object must be the entire '
        'assistant response for a tool request: no explanation before or after, '
        'and no role/content-block wrapper. If no tool is needed, answer '
        'normally in plain text. Do not mention this contract.\n'
        f'Available tools:\n{_json_dumps(manifest)}'
    )


def _load_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in re.finditer(
        r'```(?:json)?\s*(\{.*?\})\s*```',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ))
    start = text.find('{')
    end = text.rfind('}')
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def _load_json_value(raw: str) -> Any | None:
    text = raw.strip()
    if not text:
        return None
    candidates = [text]
    start = text.find('[')
    end = text.rfind(']')
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    start = text.find('{')
    end = text.rfind('}')
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _coerce_tool_calls(
    raw_calls: Any,
    allowed_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    tool_calls: list[dict[str, Any]] = []
    for idx, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            continue
        name = raw_call.get('name')
        args = raw_call.get('args', {})
        if not isinstance(name, str) or not name:
            continue
        if allowed_names is not None and name not in allowed_names:
            continue
        if not isinstance(args, dict):
            args = {}
        call_id = raw_call.get('id')
        if not isinstance(call_id, str) or not call_id:
            call_id = f'cobra_call_{idx}_{uuid.uuid4().hex[:12]}'
        tool_calls.append({'name': name, 'args': args, 'id': call_id})
    return tool_calls


def _coerce_tool_use_blocks(
    blocks: Any,
    allowed_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(blocks, list):
        return []
    tool_calls: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        if block.get('type') != 'tool_use':
            continue
        name = block.get('name')
        args = block.get('input', block.get('args', {}))
        if not isinstance(name, str) or not name:
            continue
        if allowed_names is not None and name not in allowed_names:
            continue
        if not isinstance(args, dict):
            args = {}
        call_id = block.get('id')
        if not isinstance(call_id, str) or not call_id:
            call_id = f'toolu_{idx}_{uuid.uuid4().hex[:12]}'
        tool_calls.append({'name': name, 'args': args, 'id': call_id})
    return tool_calls


def _content_blocks_from_payload(data: Any) -> list[Any] | None:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return None
    content = data.get('content')
    if isinstance(content, list):
        return content
    message = data.get('message')
    if isinstance(message, dict) and isinstance(message.get('content'), list):
        return message['content']
    return None


def _looks_like_content_blocks(blocks: list[Any]) -> bool:
    for block in blocks:
        if isinstance(block, dict) and block.get('type') in {'text', 'tool_use', 'tool_result'}:
            return True
    return False


def _text_from_content_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get('type')
        if block_type == 'text' and isinstance(block.get('text'), str):
            parts.append(block['text'])
        elif block_type not in {'tool_use', 'tool_result'}:
            text = block.get('text') or block.get('content')
            if isinstance(text, str):
                parts.append(text)
    return '\n'.join(part for part in parts if part).strip()


def _strip_cobra_tool_call_block(text: str) -> str:
    marker = 'Requested CoBrA tool calls:'
    idx = text.find(marker)
    if idx < 0:
        return text.strip()
    return text[:idx].strip()


def _clean_cobra_content(content: str, *, has_tool_calls: bool) -> str:
    cleaned = _strip_cobra_tool_call_block(content)
    if has_tool_calls and re.search(
        r'cobra\s*(?:tool\s*)?(?:protocol|request\s+envelope)|cobra\s*툴\s*프로토콜|anthropic-style\s+tool\s+use\s+contract',
        cleaned,
        re.IGNORECASE,
    ):
        return ''
    return cleaned


def _extract_tool_calls(
    text: str,
    allowed_names: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    data = _load_json_object(text)
    blocks = _content_blocks_from_payload(data)
    if blocks is not None:
        tool_calls = _coerce_tool_use_blocks(blocks, allowed_names)
        content = _text_from_content_blocks(blocks)
        return _clean_cobra_content(content, has_tool_calls=bool(tool_calls)), tool_calls

    raw_blocks = _load_json_value(text)
    if isinstance(raw_blocks, list) and _looks_like_content_blocks(raw_blocks):
        tool_calls = _coerce_tool_use_blocks(raw_blocks, allowed_names)
        content = _text_from_content_blocks(raw_blocks)
        return _clean_cobra_content(content, has_tool_calls=bool(tool_calls)), tool_calls

    # Legacy compatibility for transcripts produced before the wrapper mirrored
    # Anthropic content blocks.
    if data and isinstance(data.get('cobra_tool_calls'), list):
        tool_calls = _coerce_tool_calls(data.get('cobra_tool_calls'), allowed_names)
        raw_content = data.get('content', '')
        content = raw_content if isinstance(raw_content, str) else ''
        return _clean_cobra_content(content, has_tool_calls=bool(tool_calls)), tool_calls

    marker = 'Requested CoBrA tool calls:'
    idx = text.find(marker)
    if idx >= 0:
        raw_calls = _load_json_value(text[idx + len(marker):])
        tool_calls = _coerce_tool_calls(raw_calls, allowed_names)
        content = _clean_cobra_content(text[:idx], has_tool_calls=bool(tool_calls))
        return content, tool_calls

    return _clean_cobra_content(text, has_tool_calls=False), []


def resolve_claude_binary(env_prefix: str = 'COMET') -> str | None:
    """Find the Claude Code CLI binary."""
    configured = (
        os.environ.get(f'{env_prefix}_CLAUDE_CODE_BIN')
        or os.environ.get('CLAUDE_CODE_BIN')
    )
    if configured:
        expanded = os.path.expanduser(configured)
        if os.path.sep not in expanded:
            return shutil.which(expanded)
        return expanded if os.access(expanded, os.X_OK) else None

    found = shutil.which('claude')
    if found:
        return found

    home = Path.home()
    candidates = [
        Path('/opt/homebrew/bin/claude'),
        Path('/usr/local/bin/claude'),
        home / '.local' / 'bin' / 'claude',
        home / '.npm-global' / 'bin' / 'claude',
        home / '.yarn' / 'bin' / 'claude',
        home / '.bun' / 'bin' / 'claude',
    ]
    nvm_dir = home / '.nvm' / 'versions' / 'node'
    try:
        candidates.extend(sorted(nvm_dir.glob('*/bin/claude'), reverse=True))
    except Exception:
        pass
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except Exception:
            continue
    return None


def _extension_for_media_type(media_type: str) -> str:
    mt = media_type.lower().split(';', 1)[0].strip()
    return {
        'image/jpeg': 'jpg',
        'image/jpg': 'jpg',
        'image/png': 'png',
        'image/gif': 'gif',
        'image/webp': 'webp',
        'image/bmp': 'bmp',
        'image/svg+xml': 'svg',
    }.get(mt, 'png')


def _image_url_to_claude_ref(url: str, image_dir: str | None = None) -> str:
    url = str(url or '').strip()
    if not url:
        return '[image unavailable: empty URL]'
    if url.startswith('data:'):
        if not image_dir:
            return '[image omitted: image data URL received outside Claude CLI invocation]'
        try:
            header, b64 = url.split(',', 1)
            media_type = header[5:].split(';', 1)[0] or 'image/png'
            ext = _extension_for_media_type(media_type)
            path = Path(image_dir) / f'claude_image_{uuid.uuid4().hex[:12]}.{ext}'
            path.write_bytes(base64.b64decode(b64))
            return f'@{path}'
        except Exception as e:
            return f'[image unavailable: failed to decode data URL: {e}]'

    path = Path(url).expanduser()
    try:
        if path.is_file():
            return f'@{path.resolve()}'
    except Exception:
        pass
    if url.startswith('http://') or url.startswith('https://'):
        return f'[image URL: {url}]'
    return f'[image unavailable: {url}]'


def _image_block_to_claude_ref(block: dict[str, Any], image_dir: str | None = None) -> str | None:
    if block.get('type') == 'image_url' and isinstance(block.get('image_url'), dict):
        return _image_url_to_claude_ref(str(block['image_url'].get('url') or ''), image_dir)
    if block.get('type') == 'image':
        source = block.get('source')
        if isinstance(source, dict):
            if source.get('type') == 'base64' and isinstance(source.get('data'), str):
                media_type = str(source.get('media_type') or 'image/png')
                return _image_url_to_claude_ref(f'data:{media_type};base64,{source["data"]}', image_dir)
            if source.get('type') in {'url', 'image_url'} and isinstance(source.get('url'), str):
                return _image_url_to_claude_ref(source['url'], image_dir)
        if isinstance(block.get('data'), str):
            media_type = str(block.get('media_type') or 'image/png')
            return _image_url_to_claude_ref(f'data:{media_type};base64,{block["data"]}', image_dir)
    return None


def _content_to_text(content: Any, image_dir: str | None = None) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                image_ref = _image_block_to_claude_ref(block, image_dir)
                if image_ref:
                    parts.append(f'Image attachment: {image_ref}')
                    continue
                text = block.get('text') or block.get('content')
                if isinstance(text, str):
                    parts.append(text)
        return '\n'.join(parts)
    return str(content) if content else ''


def _role_name(message: BaseMessage) -> str:
    msg_type = getattr(message, 'type', '')
    if msg_type == 'human':
        return 'user'
    if msg_type == 'ai':
        return 'assistant'
    if msg_type == 'tool':
        return 'tool'
    return msg_type or message.__class__.__name__


def _tool_call_field(call: Any, key: str, default: Any = None) -> Any:
    if isinstance(call, dict):
        return call.get(key, default)
    return getattr(call, key, default)


def _assistant_content_blocks(text: str, tool_calls: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({'type': 'text', 'text': text})
    for idx, call in enumerate(tool_calls):
        name = _tool_call_field(call, 'name', '')
        if not isinstance(name, str) or not name:
            continue
        args = _tool_call_field(call, 'args', {})
        if not isinstance(args, dict):
            args = {}
        call_id = _tool_call_field(call, 'id', '')
        if not isinstance(call_id, str) or not call_id:
            call_id = f'toolu_history_{idx}_{uuid.uuid4().hex[:12]}'
        blocks.append({
            'type': 'tool_use',
            'id': call_id,
            'name': name,
            'input': args,
        })
    return blocks


def _tool_result_payload(tool_call_id: str, content: str) -> dict[str, Any]:
    return {
        'role': 'user',
        'content': [{
            'type': 'tool_result',
            'tool_use_id': tool_call_id,
            'content': content,
        }],
    }


def messages_to_claude_prompt(
    messages: list[BaseMessage],
    image_dir: str | None = None,
) -> tuple[str, str]:
    system_parts: list[str] = []
    body_parts: list[str] = []
    tool_call_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        text = _content_to_text(message.content, image_dir=image_dir).strip()
        tool_calls = getattr(message, 'tool_calls', None) or []
        if tool_calls:
            for call in tool_calls:
                call_id = _tool_call_field(call, 'id', '')
                if not isinstance(call_id, str) or not call_id:
                    continue
                tool_call_by_id[call_id] = {
                    'name': _tool_call_field(call, 'name', ''),
                    'args': _tool_call_field(call, 'args', {}),
                    'id': call_id,
                }
        if isinstance(message, SystemMessage) or getattr(message, 'type', '') == 'system':
            if not text:
                continue
            system_parts.append(text)
        elif tool_calls:
            payload = {
                'role': 'assistant',
                'content': _assistant_content_blocks(text, tool_calls),
            }
            body_parts.append(f'ASSISTANT:\n{_json_dumps(payload)}')
        elif getattr(message, 'type', '') == 'tool':
            call_id = getattr(message, 'tool_call_id', '')
            if not isinstance(call_id, str):
                call_id = ''
            # Keep the lookup for forward compatibility and validation in tests:
            # the tool_result block itself mirrors Anthropic's native transcript,
            # where the ID is enough to pair it to the prior tool_use block.
            tool_call_by_id.get(call_id, {})
            body_parts.append(f'USER:\n{_json_dumps(_tool_result_payload(call_id, text))}')
        else:
            if not text:
                continue
            role = _role_name(message).upper()
            body_parts.append(f'{role}:\n{text}')
    return '\n\n'.join(system_parts), '\n\n'.join(body_parts)


def extract_claude_result(stdout: str) -> tuple[str, dict[str, Any]]:
    """Parse ``claude -p --output-format json`` while tolerating text output."""
    raw = stdout.strip()
    if not raw:
        return '', {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {}
    if isinstance(data, dict):
        result = data.get('result')
        if isinstance(result, str):
            return result, data
        message = data.get('message')
        if isinstance(message, dict):
            content = message.get('content')
            if isinstance(content, str):
                return content, data
            if isinstance(content, list):
                parts = [
                    block.get('text')
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get('text'), str)
                ]
                return '\n'.join(parts), data
    return raw, {}


def _apply_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text
    cut = len(text)
    for marker in stop:
        idx = text.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut]


def _claude_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in CLAUDE_CODE_CLEAR_ENV:
        env.pop(name, None)
    env.setdefault('HOME', str(Path.home()))
    return env


class ClaudeCodeOAuthChatModel(BaseChatModel):
    """Minimal LangChain adapter around Claude Code print mode."""

    model: str
    timeout: float = DEFAULT_TIMEOUT_S
    claude_bin: str | None = None
    cwd: str | None = None
    effort: str | None = None
    bound_tools: tuple[Any, ...] = ()

    @property
    def _llm_type(self) -> str:
        return 'claude_code_oauth'

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {'model': self.model, 'claude_bin': self.claude_bin}

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> 'ClaudeCodeOAuthChatModel':
        """Bind CoBrA/LangChain tools through Anthropic-style content blocks."""
        del tool_choice, kwargs
        if tools is None:
            bound_tools: tuple[Any, ...] = ()
        elif isinstance(tools, dict):
            bound_tools = (tools,)
        else:
            try:
                bound_tools = tuple(tools)
            except TypeError:
                bound_tools = (tools,)
        return self.model_copy(update={'bound_tools': bound_tools})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        image_tmp = tempfile.TemporaryDirectory(prefix='cobra-claude-images-')
        try:
            system_prompt, prompt = messages_to_claude_prompt(messages, image_dir=image_tmp.name)
            return self._generate_with_prompt(
                system_prompt=system_prompt,
                prompt=prompt,
                stop=stop,
                kwargs=kwargs,
            )
        finally:
            image_tmp.cleanup()

    def _generate_with_prompt(
        self,
        *,
        system_prompt: str,
        prompt: str,
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> ChatResult:
        if self.bound_tools:
            tool_prompt = _tool_use_prompt(self.bound_tools)
            system_prompt = f'{system_prompt}\n\n{tool_prompt}'.strip()
        if not prompt:
            prompt = 'Respond to the system prompt.'

        claude_bin = self.claude_bin or resolve_claude_binary('COMET')
        if not claude_bin:
            raise RuntimeError(
                'Claude Code CLI not found. Install Claude Code, add `claude` to PATH, '
                'or set COMET_CLAUDE_CODE_BIN/CLAUDE_CODE_BIN.'
            )

        args = [
            claude_bin,
            '-p',
            '--output-format',
            'json',
            '--no-session-persistence',
            '--setting-sources',
            'user',
            '--model',
            self.model,
        ]
        effort = kwargs.get('effort') or self.effort
        if effort:
            args.extend(['--effort', str(effort)])
        args.extend(['--tools', ''])
        if system_prompt:
            args.extend(['--system-prompt', system_prompt])

        env = _claude_subprocess_env()
        cwd = self.cwd or os.environ.get('COMET_CLAUDE_CODE_CWD') or tempfile.gettempdir()
        try:
            completed = subprocess.run(
                args,
                input=prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(kwargs.get('timeout') or self.timeout),
                cwd=cwd,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f'Claude Code CLI timed out after {self.timeout:g}s') from e
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f'Claude Code CLI exited with code {completed.returncode}: {stderr}')
        text, llm_output = extract_claude_result(completed.stdout)
        text = _apply_stop(text, stop)
        if self.bound_tools:
            allowed_names = {item['name'] for item in _tool_manifest(self.bound_tools)}
            content, tool_calls = _extract_tool_calls(text, allowed_names)
        else:
            content, tool_calls = text, []
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=tool_calls))],
            llm_output=llm_output,
        )
