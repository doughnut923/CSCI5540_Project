"""Optimus-specific overrides of the unchanged team model configuration."""


def model_arguments(seq_length=1024):
    from pp2_equivalence import mcore_argv

    if seq_length < 1:
        raise ValueError("sequence length must be positive")
    argv = list(mcore_argv(seq_length))
    # The upstream text runner keeps Qwen2Tokenizer. Our multimodal embedding
    # calls require the image token IDs exposed by Qwen2VLTokenizer.
    argv[argv.index("--patch-tokenizer-type") + 1] = "Qwen2VLTokenizer"
    return argv
