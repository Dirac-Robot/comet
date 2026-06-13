"""Per-model llm_config routing + provider-aware structured output."""
from ato.adict import ADict

from comet.llm_factory import create_chat_model, structured_output_kwargs


def test_llm_config_overrides_shared_provider_block():
    """A model passed with its own llm_config must not inherit the shared
    config.llm provider (the cross-wiring that 404'd a GPT compacter on
    the Google API when the sensor was a Gemini model)."""
    config = ADict(llm=ADict(provider='google', api_key='gk'))
    model = create_chat_model(
        'gpt-5.4-mini', config, llm_config=ADict(provider='openai', api_key='ok'),
    )
    assert type(model).__name__ == 'ChatOpenAI'

    fallback = create_chat_model('gemini-3.1-flash-lite', config)
    assert type(fallback).__name__ == 'ChatGoogleGenerativeAI'


def test_openai_oauth_defaults_streaming_on():
    """Codex Responses rejects non-streamed calls — the factory must
    default streaming=True for openai_oauth (overridable)."""
    config = ADict(llm=ADict(
        provider='openai_oauth', api_key='tok', base_url='http://x',
    ))
    model = create_chat_model('gpt-5.4-mini', config)
    assert model.streaming is True

    explicit = create_chat_model(
        'gpt-5.4-mini', config,
        llm_config=ADict(provider='openai_oauth', api_key='tok',
                         base_url='http://x', streaming=False),
    )
    assert explicit.streaming is False


def test_structured_output_kwargs_per_provider():
    """openai_oauth must use function_calling (streamed json_schema never
    populates 'parsed' on the Codex backend); others keep the default."""
    assert structured_output_kwargs(ADict(provider='openai_oauth')) == {
        'method': 'function_calling',
    }
    assert structured_output_kwargs(ADict(provider='google')) == {}
    assert structured_output_kwargs(None) == {}
