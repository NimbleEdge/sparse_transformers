from src.activation_capture import ActivationCapture
import torch.nn.functional as F


class ActivationCaptureOPT(ActivationCapture):
    has_gate_proj: bool = True
    has_up_proj: bool = False   #potentially swap these later

    def get_layers(self, model):
        return model.model.decoder.layers

    def _create_mlp_hook(self, layer_idx, proj_type):
        def hook(module, input, output):
            key = f"{layer_idx}_{proj_type}"
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[key] = output.detach()
            return output
        return hook

    def _register_gate_hook(self, layer_idx, layer):
        handle = layer.mlp.gate_proj.register_forward_hook(
            self._create_mlp_hook(layer_idx, 'gate')
        )
        return handle
    
    def _register_up_hook(self, layer_idx, layer):
        pass
        
    def get_mlp_activations(self, layer_idx):
        """Get combined MLP activations for a layer."""
        gate_key = f"{layer_idx}_gate"
        
        if gate_key in self.mlp_activations:
            # Compute gated activations: gate(x) * up(x)
            gate_act = self.mlp_activations[gate_key]
            
            # Apply SwiGLU activation: silu(gate) * up
            gated_act = F.relu(gate_act) 
            return gated_act
        
        return None