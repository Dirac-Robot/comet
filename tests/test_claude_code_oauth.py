import json

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
    assert 'Anthropic-style tool use contract' in system_prompt
    assert 'input_schema' in system_prompt
    assert 'think' in system_prompt


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
