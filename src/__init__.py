"""Cross-linguistic LLM performance and tokenization research pipeline.

Top-level package. Submodules:
    src.utils       -- config, logging, IO, run management
    src.data        -- corpus loaders + normalization
    src.tokenizers  -- tokenizer registry (tiktoken, sentencepiece, HF, byte, char)
    src.backends    -- model backends (HF local, OpenAI, Together, dummy)
    src.experiments -- the five experiments
    src.plotting    -- matplotlib publication helpers
    src.stats       -- bootstrap CIs, paired tests, regressions
"""

__version__ = "0.1.0"
