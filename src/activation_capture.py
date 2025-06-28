import torch.nn.functional as F
from abc import ABC, abstractmethod


class ActivationCapture(ABC):
    """Helper class to capture activations from model layers."""
    has_gate_proj: bool
    has_up_proj: bool
    
    def __init__(self):
        self.hidden_states = {}
        self.mlp_activations = {}
        self.handles = []

    @abstractmethod
    def _register_gate_hook(self, layer_idx, layer):
        pass

    @abstractmethod
    def _register_up_hook(self, layer_idx, layer):
        pass

    @abstractmethod
    def get_layers(self, model):
        pass

    def _register_hidden_state_hook(self, layer_idx, layer):
        def hook(module, args, kwargs, output):
            # args[0] is the input hidden states to the layer
            if len(args) > 0:
                # Just detach, don't clone or move to CPU yet
                self.hidden_states[layer_idx] = args[0].detach()
            return output
        handle = layer.register_forward_hook(
            hook,
            with_kwargs=True
        )
        return handle

    def register_hooks(self, model):
        """Register forward hooks to capture activations."""
        # Clear any existing hooks
        self.remove_hooks()
        
        # Hook into each transformer layer
        for i, layer in enumerate(self.get_layers(model)):

            # Capture hidden states before MLP
            handle = self._register_hidden_state_hook(i, layer)
            if handle is not None:
                self.handles.append(handle)
            
            # Capture MLP gate activations (after activation function)
            if self.has_gate_proj:
                handle = self._register_gate_hook(i, layer)
                if handle is not None:
                    self.handles.append(handle)
            
            # Also capture up_proj activations
            if self.has_up_proj:
                handle = self._register_up_hook(i, layer)
                if handle is not None:
                    self.handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def clear_captures(self):
        """Clear captured activations."""
        self.hidden_states = {}
        self.mlp_activations = {}

    @abstractmethod
    def get_mlp_activations(self, layer_idx):
        """Get combined MLP activations for a layer."""
        pass

    @abstractmethod
    def get_gate_activations(self, layer_idx):
        """Get combined MLP activations for a layer."""
        return 


class ActivationCaptureDefault(ActivationCapture):
    """Helper class to capture activations from model layers."""
    has_gate_proj: bool = True
    has_up_proj: bool = True

    def get_layers(self, model):
        return model.model.layers

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
        handle = layer.mlp.up_proj.register_forward_hook(
            self._create_mlp_hook(layer_idx, 'up')
        )
        return handle
        
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
    
    def get_gate_activations(self, layer_idx):
        """Get combined MLP activations for a layer."""
        gate_key = f"{layer_idx}_gate"
        if gate_key in self.mlp_activations:
            gate_act = self.mlp_activations[gate_key]
            return F.silu(gate_act)
        return None
    

ACTIVATION_CAPTURE = {}

def register_activation_capture(model_name, activation_capture):
    ACTIVATION_CAPTURE[model_name] = activation_capture
