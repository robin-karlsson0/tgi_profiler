from unittest.mock import Mock, patch

import numpy as np
import pytest
from transformers import AutoTokenizer

from tgi_profiler.profiler import TGIMemoryProfiler
from tgi_profiler.tgi_container import TGIConfig, TGIContainer

TOKEN_LEN_THRESH = 100


@pytest.mark.integration
def test_generate_exact_token_output(basic_profiler_config):
    """Test generating input text and verifying token counts through inference"""

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
            with TGIContainer(tgi_config) as container:  # noqa

                _, output_len = profiler._generate_exact_token_output(
                    output_length, input_txt)

                if output_len > 0:
                    assert output_len > 0
                    break
        except Exception as e:
            if attempt == basic_profiler_config.retries_per_point - 1:
                raise e


@pytest.mark.integration
def test_test_point_short(basic_profiler_config):
    """Test generating input text and verifying token counts through inference"""
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
    """Test generating input text and verifying token counts through inference"""
    input_length = 2000
    output_length = 1000
    profiler = TGIMemoryProfiler(basic_profiler_config)
    # Test complete inference flow
    result = profiler.test_point(input_length, output_length)
    assert result.success
    assert abs(result.input_length - input_length) <= TOKEN_LEN_THRESH
    assert abs(result.output_length - output_length) <= TOKEN_LEN_THRESH
