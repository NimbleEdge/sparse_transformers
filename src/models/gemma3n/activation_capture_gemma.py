from src.activation_capture import ActivationCaptureDefault


class ActivationCaptureGemma3n(ActivationCaptureDefault):
    """Helper class to capture activations from model layers."""

    def _register_gate_hook(self, layer_idx, layer):
        handle = layer.mlp.act_fn.register_forward_hook(
            self._create_mlp_hook(layer_idx, 'gate')
        )
        return handle
