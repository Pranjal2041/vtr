"""vLLM general plugin (entry point `vllm.general_plugins`): registers the VTR model."""


def register():
    from vllm import ModelRegistry

    from vtr_inference.config import ARCHITECTURE

    ModelRegistry.register_model(ARCHITECTURE, f"vtr_inference.model:{ARCHITECTURE}")
