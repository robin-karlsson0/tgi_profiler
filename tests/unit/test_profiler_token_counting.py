from unittest.mock import Mock, patch

import pytest
from transformers import AutoTokenizer

from tgi_profiler.profiler import TGIMemoryProfiler

LOREM_IPSUM = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'  # noqa


def test_count_tokens(basic_profiler_config):
    profiler = TGIMemoryProfiler(basic_profiler_config)

    # Test basic token counting
    num_tokens = profiler._count_tokens(LOREM_IPSUM)

    model_id = profiler.config.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    num_tokens_gt = len(tokenizer.tokenize(LOREM_IPSUM))

    assert num_tokens == num_tokens_gt


def test_generate_exact_token_input(basic_profiler_config):

    target_length = 666

    profiler = TGIMemoryProfiler(basic_profiler_config)
    input_txt = profiler._generate_exact_token_input(target_length)

    model_id = profiler.config.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    num_tokens = len(tokenizer.tokenize(input_txt))

    assert num_tokens == target_length
