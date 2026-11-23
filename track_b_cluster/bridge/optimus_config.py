"""Optimus-specific overrides of the unchanged team model configuration."""

from copy import deepcopy

PP8_LAYERS = (2, 4, 4, 4, 4, 4, 4, 2)


def llm_layer_counts(layers, pp):
    if (layers, pp) == (28, 8):
        # The endpoint stages also hold the embedding or output projection.
        return PP8_LAYERS
    if pp < 2 or layers < pp or layers % pp:
        raise ValueError("use PP >= 2 with a positive uniform layer split, or 28 layers with PP=8")
    return (layers // pp,) * pp


def model_arguments(seq_length=1024, *, layer_counts=()):
    from pp2_equivalence import mcore_argv

    if seq_length < 1:
        raise ValueError("sequence length must be positive")
    argv = list(mcore_argv(seq_length))
    # The upstream text runner keeps Qwen2Tokenizer. Our multimodal embedding
    # calls require the image token IDs exposed by Qwen2VLTokenizer.
    argv[argv.index("--patch-tokenizer-type") + 1] = "Qwen2VLTokenizer"
    if layer_counts == PP8_LAYERS:
        argv.extend(["--decoder-first-pipeline-num-layers", str(layer_counts[0]),
                     "--decoder-last-pipeline-num-layers", str(layer_counts[-1])])
    return argv


def whole_encoder_config(qwen, args, language_config):
    config = qwen.get_vision_model_config(args, deepcopy(language_config))
    config.pipeline_model_parallel_size = 1
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None
    return config
