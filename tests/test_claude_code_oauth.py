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


def test_claude_code_oauth_passes_image_refs_to_claude_prompt(monkeypatch):
    raw = b'invoke image bytes'
    data_url = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()

    def fake_run(args, **kwargs):
        match = re.search(r'Image attachment: @([^\n]+)', kwargs['input'])
        assert match
        image_path = Path(match.group(1))
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

    model = ClaudeCodeOAuthChatModel(model='claude-opus-4-7', claude_bin='/usr/bin/claude')
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

    model = ClaudeCodeOAuthChatModel(model='claude-opus-4-7', claude_bin='/usr/bin/claude')
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

    model = ClaudeCodeOAuthChatModel(model='claude-opus-4-7', claude_bin='/usr/bin/claude')
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

    model = ClaudeCodeOAuthChatModel(model='claude-opus-4-7', claude_bin='/usr/bin/claude')
    response = model.bind_tools([{'name': 'read_file_tool'}]).invoke([HumanMessage(content='read it')])

    assert response.content == ''
    assert response.tool_calls[0]['name'] == 'read_file_tool'
    assert response.tool_calls[0]['args'] == {'file_path': '/tmp/a.png'}
    assert response.tool_calls[0]['id'] == 'call-read'
