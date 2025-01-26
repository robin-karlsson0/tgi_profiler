from unittest.mock import Mock, patch

import numpy as np
import pytest
from transformers import AutoTokenizer

from tgi_profiler.profiler import TGIMemoryProfiler
from tgi_profiler.tgi_container import TGIConfig, TGIContainer


@pytest.mark.integration
def test_generate_and_verify_token_lengths_short(basic_profiler_config):
    """Test generating input text and verifying token counts through inference"""

    # Test points
    input_length = 100
    output_length = 50

    basic_profiler_config.min_input_length = input_length

    profiler = TGIMemoryProfiler(basic_profiler_config)

    # Input length
    input_txt = profiler._generate_exact_token_input(input_length)

    model_id = profiler.config.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert len(tokenizer.tokenize(input_txt)) == input_length

    # Output length

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

                output_txt = profiler._generate_exact_token_output(
                    input_txt, output_length)

                print('Generated output length: '
                      f'{len(tokenizer.tokenize(output_txt))} '
                      f'expected: {output_length}')

                if len(tokenizer.tokenize(output_txt)) == output_length:
                    assert len(tokenizer.tokenize(output_txt)) == output_length
                    break
        except Exception as e:
            if attempt == basic_profiler_config.retries_per_point - 1:
                raise e


@pytest.mark.integration
def test_generate_and_verify_token_lengths_long(basic_profiler_config):
    """Test generating input text and verifying token counts through inference"""

    # Test points
    input_length = 1000
    output_length = 500

    basic_profiler_config.min_input_length = input_length

    profiler = TGIMemoryProfiler(basic_profiler_config)

    # Input length
    input_txt = profiler._generate_exact_token_input(input_length)

    model_id = profiler.config.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert len(tokenizer.tokenize(input_txt)) == input_length

    # Output length

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

                output_txt = profiler._generate_exact_token_output(
                    input_txt, output_length)

                print('Generated output length: '
                      f'{len(tokenizer.tokenize(output_txt))} '
                      f'expected: {output_length}')

                if len(tokenizer.tokenize(output_txt)) == output_length:
                    assert len(tokenizer.tokenize(output_txt)) == output_length
                    break

        except Exception as e:
            if attempt == basic_profiler_config.retries_per_point - 1:
                raise e
