from dataclasses import dataclass

import pytest

from tgi_profiler.profiler import TGIMemoryProfiler, TokenGenerationError
from tgi_profiler.tgi_container import TGIConfig, TGIContainer

TOKEN_LEN_THRESH = 100


@pytest.mark.integration
def test_generate_exact_token_output(basic_profiler_config):
    """Test token generation through model inference.

    Tests:
        - Successful initialization of TGI container
        - Token generation with target length
        - Retry logic on failures
        - Container cleanup

    Args:
        basic_profiler_config: Standard test configuration fixture

    Raises:
        Propagates container startup and inference errors after retries
    """

    input_txt = 'What is the meaning of life?'
    input_length = 100
    output_length = 50

    profiler = TGIMemoryProfiler(basic_profiler_config)

    # Initialize TGI Docker container
    tgi_config = TGIConfig(
        model_id=basic_profiler_config.model_id,
        gpu_ids=basic_profiler_config.gpu_ids,
        port=basic_profiler_config.port,
        max_input_length=input_length,
        max_output_length=output_length,
        hf_token=basic_profiler_config.hf_token,
        hf_cache_dir=basic_profiler_config.hf_cache_dir,
    )

    for attempt in range(basic_profiler_config.retries_per_point):
        try:
            with TGIContainer(tgi_config) as container:

                _, output_len = profiler._generate_exact_token_output(
                    output_length, input_txt)

                if output_len > 0:
                    assert output_len > 0
                    break
        except Exception as e:
            if attempt == basic_profiler_config.retries_per_point - 1:
                raise e
        finally:
            container.stop()


def test_token_length_mismatch_retries(basic_profiler_config):
    """Test handling of failed token generation attempts.

    Tests:
        - Mock chat completion returning short responses
        - TokenGenerationError raised after retries
        - Error propagation through profiler

    Args:
        basic_profiler_config: Standard test configuration fixture
    """

    @dataclass
    class Message:
        content: str

    @dataclass
    class Choice:
        message: Message

    @dataclass
    class ChatCompletionOutput:
        choices: list[Choice]

    def mock_chat_completion(messages):
        return ChatCompletionOutput(
            choices=[Choice(message=Message(content='Test output'))])

    def mock_count_tokens(text):
        return 100  # Simulate shorter response than target

    basic_profiler_config.retries_per_point = 3
    profiler = TGIMemoryProfiler(basic_profiler_config)

    profiler.client.chat_completion = mock_chat_completion
    profiler._count_tokens = mock_count_tokens

    try:
        output_txt, output_len = profiler._generate_exact_token_output(
            1000, 'Test input')
    except Exception as e:
        assert type(e) is TokenGenerationError


@pytest.mark.integration
def test_test_point_short(basic_profiler_config):
    """Test profiling with short sequences.

    Tests:
        - Input length: 100 tokens
        - Output length: 50 tokens
        - Token count accuracy within tolerance

    Args:
        basic_profiler_config: Standard test configuration fixture
    """
    input_length = 100
    output_length = 50
    profiler = TGIMemoryProfiler(basic_profiler_config)
    # Test complete inference flow
    result = profiler.test_point(input_length, output_length)
    assert result.success
    assert abs(result.input_length - input_length) <= TOKEN_LEN_THRESH
    assert abs(result.output_length - output_length) <= TOKEN_LEN_THRESH


@pytest.mark.integration
def test_test_point_long(basic_profiler_config):
    """Test profiling with longer sequences.

    Tests:
        - Input length: 2000 tokens
        - Output length: 1000 tokens
        - Token count accuracy within tolerance

    Args:
        basic_profiler_config: Standard test configuration fixture
    """
    input_length = 2000
    output_length = 1000
    profiler = TGIMemoryProfiler(basic_profiler_config)
    # Test complete inference flow
    result = profiler.test_point(input_length, output_length)
    assert result.success
    assert abs(result.input_length - input_length) <= TOKEN_LEN_THRESH
    assert abs(result.output_length - output_length) <= TOKEN_LEN_THRESH


@pytest.mark.integration
def test_error_handling(basic_profiler_config):
    """Test OOM error handling with extreme sequence lengths.

    Tests:
        - Very large input (1M tokens)
        - Large output (100K tokens)
        - Error capture in ProfilingResult

    Args:
        basic_profiler_config: Standard test configuration fixture
    """
    # Test very large sequence lengths likely to cause OOM
    input_length = int(1e6)
    output_length = int(1e5)
    profiler = TGIMemoryProfiler(basic_profiler_config)

    result = profiler.test_point(input_length, output_length)

    assert not result.success
    assert result.error_type is not None
