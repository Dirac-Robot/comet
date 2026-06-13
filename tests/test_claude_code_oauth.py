import base64
import json
import re
from pathlib import Path

from ato.adict import ADict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from comet import claude_code_oauth
from comet.claude_code_oauth import (
    ClaudeCodeOAuthChatModel,
    extract_claude_result,
    messages_to_claude_prompt,
)
from comet.llm_factory import create_chat_model


def test_create_chat_model_with_claude_code_oauth_provider_config():
    model = create_chat_model(
        'claude-opus-4-7',
        ADict(llm=ADict(provider='claude_code_oauth')),
    )

    assert isinstance(model, ClaudeCodeOAuthChatModel)
    assert model.model == 'claude-opus-4-7'


def test_create_chat_model_with_claude_oauth_route_prefix():
    model = create_chat_model(
        'oauth:claude-sonnet-4-6',
        ADict(llm=ADict(provider='openai')),
    )

    assert isinstance(model, ClaudeCodeOAuthChatModel)
    assert model.model == 'claude-sonnet-4-6'


def test_extract_claude_result_json_result():
    text, metadata = extract_claude_result(
        '{"type":"result","subtype":"success","result":"hello"}'
    )

    assert text == 'hello'
    assert metadata['type'] == 'result'


def test_messages_to_claude_prompt_forwards_cobra_system_prompt_and_tool_results():
    system_prompt, prompt = messages_to_claude_prompt([
        SystemMessage(content='CoBrA core harness\n\nsession_harness'),
        HumanMessage(content='hi'),
        AIMessage(content='', tool_calls=[{
            'name': 'think',
            'args': {'thought': 'inspect'},
            'id': 'call-think-1',
        }]),
        ToolMessage(content='done', tool_call_id='call-think-1'),
    ])

    assert system_prompt == 'CoBrA core harness\n\nsession_harness'
    assert 'USER:\nhi' in prompt
    assert '"type": "tool_use"' in prompt
    assert '"name": "think"' in prompt
    assert '"input": {"thought": "inspect"}' in prompt
    assert '"type": "tool_result"' in prompt
    assert '"tool_use_id": "call-think-1"' in prompt
    assert '"thought": "inspect"' in prompt
    assert 'COBRA TOOL RESULT' not in prompt
    assert 'Requested CoBrA tool calls' not in prompt


def test_messages_to_claude_prompt_writes_data_url_images_for_claude_cli(tmp_path):
    raw = b'fake png bytes'
    data_url = 'data:image/png;base64,' + base64.b64encode(raw).decode()

    system_prompt, prompt = messages_to_claude_prompt([
        HumanMessage(content=[
            {'type': 'text', 'text': 'look at this'},
            {'type': 'image_url', 'image_url': {'url': data_url}},
        ]),
    ], image_dir=str(tmp_path))

    assert system_prompt == ''
    assert 'USER:\nlook at this\nImage attachment: @' in prompt
    match = re.search(r'Image attachment: @([^\n]+)', prompt)
    assert match
    image_path = Path(match.group(1))
    assert image_path.parent == tmp_path
    assert image_path.suffix == '.png'
    assert image_path.read_bytes() == raw


def test_messages_to_claude_prompt_references_local_image_paths(tmp_path):
    image_path = tmp_path / 'attached.jpg'
    image_path.write_bytes(b'jpeg bytes')

    _, prompt = messages_to_claude_prompt([
        HumanMessage(content=[
            {'type': 'text', 'text': 'inspect file'},
            {'type': 'image_url', 'image_url': {'url': str(image_path)}},
        ]),
    ])

    assert f'Image attachment: @{image_path.resolve()}' in prompt


def test_messages_to_claude_prompt_keeps_tool_result_images_visible(tmp_path):
    raw = b'tool image bytes'
    data_url = 'data:image/webp;base64,' + base64.b64encode(raw).decode()

    _, prompt = messages_to_claude_prompt([
        AIMessage(content='', tool_calls=[{
            'name': 'read_file_tool',
            'args': {'file_path': '/tmp/a.webp'},
            'id': 'call-read',
        }]),
        ToolMessage(content=[
            {'type': 'text', 'text': 'image preview'},
            {'type': 'image_url', 'image_url': {'url': data_url}},
        ], tool_call_id='call-read'),
    ], image_dir=str(tmp_path))

    assert '"type": "tool_result"' in prompt
    assert '"content": "image preview\\nImage attachment: @' in prompt
    match = re.search(r'Image attachment: @([^"\\]+)', prompt)
    assert match
    image_path = Path(match.group(1))
    assert image_path.suffix == '.webp'
    assert image_path.read_bytes() == raw


def test_claude_code_oauth_bind_tools_returns_bound_copy():
    model = ClaudeCodeOAuthChatModel(model='claude-opus-4-7')

    bound = model.bind_tools([{
        'name': 'think',
        'description': 'reason privately',
        'parameters': {'type': 'object', 'properties': {'thought': {'type': 'string'}}},
    }])

    assert bound is not model
    assert bound.bound_tools
    assert model.bound_tools == ()


def test_claude_code_oauth_disables_claude_tools_by_default(monkeypatch):
    calls = {}

    def fake_run(args, **kwargs):
        calls['args'] = args
        calls['kwargs'] = kwargs
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"type":"result","subtype":"success","result":"ok"}',
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    model = ClaudeCodeOAuthChatModel(model='claude-opus-4-7', claude_bin='/usr/bin/claude')
    result = model.invoke([HumanMessage(content='hi')])

    assert result.content == 'ok'
    assert calls['args'][calls['args'].index('--tools') + 1] == ''


def test_claude_code_oauth_prepends_host_cli_neutralizer(monkeypatch):
    """The --system-prompt passed to ``claude -p`` must lead with the host-CLI
    neutralization preamble. OAuth mode runs the full Claude Code harness (it
    must, to read the keychain login), which injects scaffolding it cannot
    suppress — claudeMd/memory blocks and keyword-triggered "use the Workflow
    tool" <system-reminder>s. The preamble tells the model to ignore them."""
    calls = {}

    def fake_run(args, **kwargs):
        calls['args'] = args
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"type":"result","subtype":"success","result":"ok"}',
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    model = ClaudeCodeOAuthChatModel(model='claude-opus-4-7', claude_bin='/usr/bin/claude')
    model.invoke([HumanMessage(content='hi')])

    system_prompt = calls['args'][calls['args'].index('--system-prompt') + 1]
    assert system_prompt.startswith('[CoBrA runtime')
    # The preamble is read fresh from the template each call (runtime-tunable),
    # so the system prompt must carry it.
    neutralize = claude_code_oauth._host_cli_neutralize()
    assert neutralize in system_prompt
    # Forbid not just following the leak but the DECLINE/NOTE form too — "I won't
    # use that / it doesn't exist here" is the observed narration leak.
    assert "don't follow it" in neutralize
    assert "don't decline it" in neutralize
    assert 'no response at all' in neutralize
    # Call suppression is general + positive (function list, not named tools).
    assert 'call only tools that appear in your function definitions' in neutralize
    # Naming specific leaked tools backfired — the preamble must name none.
    for leaked in ('Workflow tool', 'TodoWrite', 'LSP', 'Read', 'Edit', 'Bash', 'Glob', 'Grep'):
        assert leaked not in neutralize, f'neutralizer must not name {leaked!r}'


def test_neutralizer_is_runtime_tunable_not_cached():
    """The preamble is read fresh from disk each call (NOT lru_cached), so a
    wording change lands without a daemon restart."""
    import comet.claude_code_oauth as cco
    original = cco._NEUTRALIZE_PATH.read_text(encoding='utf-8')
    try:
        cco._NEUTRALIZE_PATH.write_text(original + '\nSENTINEL_RUNTIME_TUNE\n', encoding='utf-8')
        assert 'SENTINEL_RUNTIME_TUNE' in cco._host_cli_neutralize()
    finally:
        cco._NEUTRALIZE_PATH.write_text(original, encoding='utf-8')
    # and a missing file falls back, never raises
    assert cco._host_cli_neutralize()


def test_subprocess_env_disables_host_harness_oauth_safe(monkeypatch):
    """The subprocess env cuts host-harness scaffolding at the source (Workflows
    feature, CLAUDE.md, auto-memory, git instructions) WITHOUT setting
    CLAUDE_CODE_SIMPLE / clearing keychain — OAuth must survive."""
    from comet import claude_code_oauth as cco
    # a stray OAuth-conflicting var must still be cleared; HOME preserved
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-should-be-removed')
    env = cco._claude_subprocess_env()
    assert 'ANTHROPIC_API_KEY' not in env            # OAuth-conflict cleared
    assert env['CLAUDE_CODE_DISABLE_WORKFLOWS'] == '1'   # the reminder's source
    assert env['CLAUDE_CODE_DISABLE_CLAUDE_MDS'] == '1'
    assert env['CLAUDE_CODE_DISABLE_AUTO_MEMORY'] == '1'
    # MUST NOT disable OAuth: CLAUDE_CODE_SIMPLE skips keychain reads.
    assert 'CLAUDE_CODE_SIMPLE' not in env


def test_subprocess_env_respects_explicit_override(monkeypatch):
    """setdefault — an explicit outer value for a harness toggle still wins."""
    from comet import claude_code_oauth as cco
    monkeypatch.setenv('CLAUDE_CODE_DISABLE_WORKFLOWS', '0')
    env = cco._claude_subprocess_env()
    assert env['CLAUDE_CODE_DISABLE_WORKFLOWS'] == '0'


def test_claude_code_oauth_passes_image_refs_to_claude_prompt(monkeypatch):
    raw = b'invoke image bytes'
    data_url = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()
    calls = {}

    def fake_run(args, **kwargs):
        calls['args'] = args
        calls['kwargs'] = kwargs
        match = re.search(r'Image attachment: @([^\n]+)', kwargs['input'])
        assert match
        image_path = Path(match.group(1))
        calls['image_path'] = image_path
        assert image_path.exists()
        assert image_path.suffix == '.jpg'
        assert image_path.read_bytes() == raw
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"type":"result","subtype":"success","result":"seen"}',
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    model = ClaudeCodeOAuthChatModel(model='claude-opus-4-7', claude_bin='/usr/bin/claude')
    result = model.invoke([HumanMessage(content=[
        {'type': 'text', 'text': 'describe this'},
        {'type': 'image_url', 'image_url': {'url': data_url}},
    ])])

    assert result.content == 'seen'
    args = calls['args']
    image_path = calls['image_path']
    assert args[args.index('--tools') + 1] == 'Read'
    assert '--add-dir' in args
    assert str(image_path.parent.resolve()) in args
    system_prompt = args[args.index('--system-prompt') + 1]
    assert 'use the Read tool' in system_prompt
    assert str(image_path) in system_prompt


def test_claude_code_oauth_emits_anthropic_style_tool_calls(monkeypatch):
    calls = {}

    def fake_run(args, **kwargs):
        calls['args'] = args
        calls['kwargs'] = kwargs
        result = json.dumps({
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'I will inspect memory.'},
                {
                    'type': 'tool_use',
                    'name': 'think',
                    'input': {'thought': 'inspect first'},
                    'id': 'call-think-1',
                },
            ],
        })
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({'type': 'result', 'subtype': 'success', 'result': result}),
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    # Envelope-path mechanics under test — opt out of the default MCP bridge.
    model = ClaudeCodeOAuthChatModel(
        model='claude-opus-4-7', claude_bin='/usr/bin/claude', use_mcp_bridge=False,
    )
    bound = model.bind_tools([{
        'name': 'think',
        'description': 'reason privately',
        'parameters': {'type': 'object', 'properties': {'thought': {'type': 'string'}}},
    }])
    result = bound.invoke([HumanMessage(content='hi')])

    assert result.content == 'I will inspect memory.'
    assert result.tool_calls[0]['name'] == 'think'
    assert result.tool_calls[0]['args'] == {'thought': 'inspect first'}
    assert result.tool_calls[0]['id'] == 'call-think-1'
    args = calls['args']
    assert args[args.index('--tools') + 1] == ''
    system_prompt = args[args.index('--system-prompt') + 1]
    assert 'CoBrA tool request envelope' in system_prompt
    assert 'cobra_tool_calls' in system_prompt
    assert 'parameters' in system_prompt
    assert 'never emit Anthropic content blocks' in system_prompt
    assert 'No such tool available' in system_prompt
    assert 'think' in system_prompt


def test_claude_code_oauth_emits_neutral_cobra_tool_calls(monkeypatch):
    def fake_run(args, **kwargs):
        result = json.dumps({
            'content': 'I will inspect memory.',
            'cobra_tool_calls': [{
                'name': 'think',
                'args': {'thought': 'inspect first'},
                'id': 'cobra_call_think_1',
            }],
        })
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({'type': 'result', 'subtype': 'success', 'result': result}),
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    # Envelope-path mechanics under test — opt out of the default MCP bridge.
    model = ClaudeCodeOAuthChatModel(
        model='claude-opus-4-7', claude_bin='/usr/bin/claude', use_mcp_bridge=False,
    )
    bound = model.bind_tools([{
        'name': 'think',
        'description': 'reason privately',
        'parameters': {'type': 'object', 'properties': {'thought': {'type': 'string'}}},
    }])
    result = bound.invoke([HumanMessage(content='hi')])

    assert result.content == 'I will inspect memory.'
    assert result.tool_calls[0]['name'] == 'think'
    assert result.tool_calls[0]['args'] == {'thought': 'inspect first'}
    assert result.tool_calls[0]['id'] == 'cobra_call_think_1'


def test_claude_code_oauth_recovers_partial_cobra_tool_envelope(monkeypatch):
    result = (
        'Worker status note that should not block parsing.\n\n'
        '{"content":"Verify dev server is up.",'
        '"cobra_tool_calls":[{"name":"run_command_tool",'
        '"args":{"command":"curl -sS http://127.0.0.1:5173/","blocking":true},'
        '"id":"cobra_call_run_1"}]'
    )

    def fake_run(args, **kwargs):
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({'type': 'result', 'subtype': 'success', 'result': result}),
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    # Envelope-path mechanics under test — opt out of the default MCP bridge.
    model = ClaudeCodeOAuthChatModel(
        model='claude-opus-4-7', claude_bin='/usr/bin/claude', use_mcp_bridge=False,
    )
    response = model.bind_tools([{'name': 'run_command_tool'}]).invoke([HumanMessage(content='check')])

    assert response.content == 'Verify dev server is up.'
    assert response.tool_calls[0]['name'] == 'run_command_tool'
    assert response.tool_calls[0]['args'] == {
        'command': 'curl -sS http://127.0.0.1:5173/',
        'blocking': True,
    }
    assert response.tool_calls[0]['id'] == 'cobra_call_run_1'
    assert response.invalid_tool_calls == []


def test_claude_code_oauth_marks_unparseable_cobra_envelope_invalid(monkeypatch):
    def fake_run(args, **kwargs):
        result = '{"content":"Broken tool request","cobra_tool_calls":]'
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({'type': 'result', 'subtype': 'success', 'result': result}),
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    # Envelope-path mechanics under test — opt out of the default MCP bridge.
    model = ClaudeCodeOAuthChatModel(
        model='claude-opus-4-7', claude_bin='/usr/bin/claude', use_mcp_bridge=False,
    )
    response = model.bind_tools([{'name': 'run_command_tool'}]).invoke([HumanMessage(content='check')])

    assert response.content == ''
    assert response.tool_calls == []
    assert response.invalid_tool_calls
    assert response.response_metadata['finish_reason'] == 'MALFORMED_FUNCTION_CALL'


def test_claude_code_oauth_salvages_malformed_anthropic_tool_blocks(monkeypatch):
    result = (
        '{"role":"assistant","content":[{"type":"text","text":"Trying workaround."},'
        '{"type":"tool_use","id":"wr1","name":"write_file_tool","input":'
        '{"file_path":"/tmp/run.py","content":"print(1)\\n"}}]},'
        '{"type":"tool_use","id":"run1","name":"run_command_tool","input":'
        '{"command":"python /tmp/run.py","blocking":true}}]}'
    )

    def fake_run(args, **kwargs):
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({'type': 'result', 'subtype': 'success', 'result': result}),
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    # Envelope-path mechanics under test — opt out of the default MCP bridge.
    model = ClaudeCodeOAuthChatModel(
        model='claude-opus-4-7', claude_bin='/usr/bin/claude', use_mcp_bridge=False,
    )
    response = model.bind_tools([
        {'name': 'write_file_tool'},
        {'name': 'run_command_tool'},
    ]).invoke([HumanMessage(content='run workaround')])

    assert response.content == 'Trying workaround.'
    assert [call['name'] for call in response.tool_calls] == [
        'write_file_tool',
        'run_command_tool',
    ]
    assert response.tool_calls[0]['args'] == {
        'file_path': '/tmp/run.py',
        'content': 'print(1)\n',
    }
    assert response.tool_calls[1]['args'] == {
        'command': 'python /tmp/run.py',
        'blocking': True,
    }
    assert response.invalid_tool_calls == []


def test_claude_code_oauth_filters_unbound_tool_use_blocks(monkeypatch):
    def fake_run(args, **kwargs):
        result = json.dumps({
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'I should not run that.'},
                {
                    'type': 'tool_use',
                    'name': 'run_command_tool',
                    'input': {'command': 'whoami'},
                },
            ],
        })
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({'type': 'result', 'subtype': 'success', 'result': result}),
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    # Envelope-path mechanics under test — opt out of the default MCP bridge.
    model = ClaudeCodeOAuthChatModel(
        model='claude-opus-4-7', claude_bin='/usr/bin/claude', use_mcp_bridge=False,
    )
    result = model.bind_tools([{'name': 'think'}]).invoke([HumanMessage(content='hi')])

    assert result.content == 'I should not run that.'
    assert result.tool_calls == []


def test_claude_code_oauth_recovers_leaked_requested_tool_call_block(monkeypatch):
    result = (
        'I will call the Anthropic-style tool use contract again.\n'
        'Requested CoBrA tool calls:\n'
        '[{"name":"read_file_tool","args":{"file_path":"/tmp/a.png"},"id":"call-read"}]'
    )

    def fake_run(args, **kwargs):
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({'type': 'result', 'subtype': 'success', 'result': result}),
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    # Envelope-path mechanics under test — opt out of the default MCP bridge.
    model = ClaudeCodeOAuthChatModel(
        model='claude-opus-4-7', claude_bin='/usr/bin/claude', use_mcp_bridge=False,
    )
    response = model.bind_tools([{'name': 'read_file_tool'}]).invoke([HumanMessage(content='read it')])

    assert response.content == ''
    assert response.tool_calls[0]['name'] == 'read_file_tool'
    assert response.tool_calls[0]['args'] == {'file_path': '/tmp/a.png'}
    assert response.tool_calls[0]['id'] == 'call-read'


def test_claude_code_oauth_unwraps_content_only_envelope(monkeypatch):
    """A calls-less envelope — the model answered a no-tool turn with the
    taught {"content": "..."} shape instead of plain text — must be unwrapped
    so the user sees the text, never the protocol JSON (2026-06-10 incident:
    the raw envelope leaked to the chat verbatim)."""
    result = json.dumps({
        'content': 'Option B is in flight — the loop is dispatched.\n\n'
                   '(The injected "use the Workflow tool" reminder isn\'t a '
                   'CoBrA primitive.)',
    })

    def fake_run(args, **kwargs):
        return claude_code_oauth.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({'type': 'result', 'subtype': 'success', 'result': result}),
            stderr='',
        )

    monkeypatch.setattr(claude_code_oauth.subprocess, 'run', fake_run)

    # Envelope-path mechanics under test — opt out of the default MCP bridge.
    model = ClaudeCodeOAuthChatModel(
        model='claude-opus-4-7', claude_bin='/usr/bin/claude', use_mcp_bridge=False,
    )
    response = model.bind_tools([{'name': 'think'}]).invoke([HumanMessage(content='status?')])

    assert response.content.startswith('Option B is in flight')
    assert '"content"' not in response.content
    assert response.tool_calls == []


def test_content_only_envelope_unwrap_variants():
    """_extract_tool_calls unwraps STRICT content-only envelopes (bare or
    singly-fenced whole-response) and nothing else."""
    extract = claude_code_oauth._extract_tool_calls

    # bare whole-response envelope → unwrapped
    content, calls, invalid = extract('{"content": "plain answer"}')
    assert (content, calls, invalid) == ('plain answer', [], [])

    # fenced whole-response envelope → unwrapped
    content, calls, invalid = extract('```json\n{"content": "fenced answer"}\n```')
    assert (content, calls, invalid) == ('fenced answer', [], [])

    # explicit empty calls list → already unwrapped by the legacy branch
    content, calls, invalid = extract('{"content": "empty calls", "cobra_tool_calls": []}')
    assert (content, calls, invalid) == ('empty calls', [], [])

    # inner content that itself looks like JSON → single unwrap, no recursion
    inner = '{"content": "nested"}'
    content, calls, invalid = extract(json.dumps({'content': inner}))
    assert (content, calls, invalid) == (inner, [], [])


def test_content_only_envelope_unwrap_declines_non_envelopes():
    """Prose around a fenced object, or any non-envelope key, is a legitimate
    answer — it must pass through untouched."""
    extract = claude_code_oauth._extract_tool_calls

    # fenced envelope-shaped example INSIDE prose → raw text preserved
    prose = ('The envelope looks like this:\n'
             '```json\n{"content": "an example"}\n```\n'
             'and that is all.')
    content, calls, invalid = extract(prose)
    assert content == prose and calls == [] and invalid == []

    # a legitimate structured JSON answer with other keys → untouched
    structured = json.dumps({'content': 'summary', 'score': 0.93})
    content, calls, invalid = extract(structured)
    assert content == structured and calls == [] and invalid == []

    # plain prose stays plain
    content, calls, invalid = extract('just a normal sentence.')
    assert content == 'just a normal sentence.' and calls == [] and invalid == []
