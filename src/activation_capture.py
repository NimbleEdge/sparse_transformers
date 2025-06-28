import torch.nn.functional as F


class ActivationCapture:
    """Helper class to capture activations from model layers."""
    
    def __init__(self):
        self.hidden_states = {}
        self.mlp_activations = {}
        self.handles = []
        
    def register_hooks(self, model):
        """Register forward hooks to capture activations."""
        # Clear any existing hooks
        self.remove_hooks()
        
        # Hook into each transformer layer
        for i, layer in enumerate(model.model.layers):

            # Capture hidden states before MLP
            handle = layer.register_forward_hook(
                self._create_hidden_state_hook(i),
                with_kwargs=True
            )
            self.handles.append(handle)
            
            # Capture MLP gate activations (after activation function)
            if hasattr(layer.mlp, 'gate_proj'):
                handle = layer.mlp.gate_proj.register_forward_hook(
                    self._create_mlp_hook(i, 'gate')
                )
                self.handles.append(handle)
            
            # Also capture up_proj activations
            if hasattr(layer.mlp, 'up_proj'):
                handle = layer.mlp.up_proj.register_forward_hook(
                    self._create_mlp_hook(i, 'up')
                )
                self.handles.append(handle)
    
    def _create_hidden_state_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            # args[0] is the input hidden states to the layer
            if len(args) > 0:
                # Just detach, don't clone or move to CPU yet
                self.hidden_states[layer_idx] = args[0].detach()
            return output
        return hook
    
    def _create_mlp_hook(self, layer_idx, proj_type):
        def hook(module, input, output):
            key = f"{layer_idx}_{proj_type}"
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[key] = output.detach()
            return output
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def clear_captures(self):
        """Clear captured activations."""
        self.hidden_states = {}
        self.mlp_activations = {}
    
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