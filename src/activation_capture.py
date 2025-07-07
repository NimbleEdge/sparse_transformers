from typing_extensions import override
import torch.nn.functional as F
from abc import ABC, abstractmethod


class ActivationCapture(ABC):
    """Helper class to capture activations from model layers."""
    has_gate_proj: bool
    has_up_proj: bool
    
    def __init__(self, model):
        self.model = model
        self.mlp_activations = {}
        self.handles = []

    @abstractmethod
    def _register_gate_hook(self, layer_idx, layer):
        pass

    @abstractmethod
    def _register_up_hook(self, layer_idx, layer):
        pass

    @abstractmethod
    def get_layers(self):
        pass


    @abstractmethod
    def get_gate_activations(self, layer_idx):
        """Get combined MLP activations for a layer."""
        pass

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        # Clear any existing hooks
        self.remove_hooks()
        
        # Hook into each transformer layer
        for i, layer in enumerate(self.get_layers()):            
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
        self.mlp_activations = {}



class ActivationCaptureDefault(ActivationCapture):
    """Helper class to capture activations from model layers."""
    has_gate_proj: bool = True
    has_up_proj: bool = True

    def get_layers(self):
        return self.model.model.layers

    def _create_mlp_hook(self, layer_idx, proj_type):
        def hook(module, input, output):
            key = f"{layer_idx}_{proj_type}"
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[key] = output.clone().detach()
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
    
    def get_gate_activations(self, layer_idx):
        gate_key = f"{layer_idx}_gate"
        if gate_key in self.mlp_activations:
            gate_act = self.mlp_activations[gate_key]
            return F.silu(gate_act)
        return None

    def get_up_activations(self, layer_idx):
        up_key = f"{layer_idx}_up"
        if up_key in self.mlp_activations:
            up_act = self.mlp_activations[up_key]
            return up_act
        return None

class ActivationCaptureTraining(ActivationCaptureDefault):
    """Additional Hidden State capture for training dataset generation"""
    def __init__(self, model):
        super().__init__(model)
        self.hidden_states = {}
    
    def _create_hidden_state_hook(self, layer_idx, layer):
        def hook(module, args, kwargs, output):
            # args[0] is the input hidden states to the layer
            if len(args) > 0:
                # Just detach, don't clone or move to CPU yet
                self.hidden_states[layer_idx] = args[0].clone().detach()
            return output
        return hook
    
    def _register_hidden_state_hook(self, layer_idx, layer):
        handle = layer.register_forward_hook(
            self._create_hidden_state_hook(layer_idx, layer),
            with_kwargs=True
        )
        return handle

    @override
    def clear_captures(self):
        """Clear captured activations."""
        super().clear_captures()
        self.hidden_states = {}

    @override
    def register_hooks(self):
        """Register forward hooks to capture activations."""
        # Clear any existing hooks
        super().register_hooks()
        # Hook into each transformer layer
        for i, layer in enumerate(self.get_layers()):            
            # Capture hidden states before MLP
            handle = self._register_hidden_state_hook(i, layer)
            if handle is not None:
                self.handles.append(handle)
    
    def get_hidden_states(self, layer_idx):
        """Get hidden states for a layer."""
        return self.hidden_states[layer_idx]
