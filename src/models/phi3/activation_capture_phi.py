from src.activation_capture import ActivationCapture
import torch.nn.functional as F



class ActivationCapturePhi3(ActivationCapture):
    """Helper class to capture activations from model layers."""
    has_gate_proj: bool = True
    has_up_proj: bool = True

    def get_layers(self, model):
        return model.model.layers

    def _register_gate_hook(self, layer_idx, layer):
        def hook(module, input, output):
            key1 = f"{layer_idx}_{'gate'}"
            key2 = f"{layer_idx}_{'up'}"
            # Just detach, don't clone or move to CPU yet
            gate_outputs, up_outputs = output.chunk(2, dim=1)
            self.mlp_activations[key1] = gate_outputs.detach()
            self.mlp_activations[key2] = up_outputs.detach()
            return output
        handle = layer.mlp.gate_up_proj.register_forward_hook(hook)
        return handle

    def _register_up_hook(self, layer_idx, layer):
        def hook(module, input, output):
            key = f"{layer_idx}_{'up'}"
            # Just detach, don't clone or move to CPU yet
            up_outputs = output.chunk(2, dim=1)[1]
            self.mlp_activations[key] = up_outputs.detach()
            return output
        handle = layer.mlp.gate_up_proj.register_forward_hook(hook)
        return handle
    
    def get_gate_activations(self, layer_idx):
        """Get combined MLP activations for a layer."""
        gate_key = f"{layer_idx}_gate"
        if gate_key in self.mlp_activations:
            gate_act = self.mlp_activations[gate_key]
            return F.silu(gate_act)
        return None
        
    def get_mlp_activations(self, layer_idx):
        """Get combined MLP activations for a layer."""
        gate_key = f"{layer_idx}_gate"
        up_key = f"{layer_idx}_up"
        
        if gate_key in self.mlp_activations and up_key in self.mlp_activations:
            # Compute gated activations: gate(x) * up(x)
            gate_act = self.mlp_activations[gate_key]
            up_act = self.mlp_activations[up_key]
            
            # Apply SwiGLU activation: silu(gate) * up
            gated_act = F.silu(gate_act) * up_act
            return gated_act
        
        return None