

import torch

from src.activation_capture import ActivationCapture, Hook

def calculate_threshold_one_token(neuron_outputs, cett_target, n_quantiles=1000):
    norms = neuron_outputs.norm(dim=0)
    quantiles = norms.quantile(torch.linspace(0,1,n_quantiles))
    tot_norm = neuron_outputs.sum(dim=1).norm()

    def CETT(threshold):
        threshold_norm = ((norms < threshold) * neuron_outputs).sum(dim=1).norm()
        return threshold_norm / tot_norm

    left = 0
    right = quantiles.size(0)
    threshold = 0
    while left < right:
        mid = (left + right) // 2
        cett = CETT(quantiles[mid])
        if cett <= cett_target:
            left = mid + 1
            threshold = quantiles[mid]
        else:
            right = mid - 1
    return threshold


def find_threshold(model, dataloader, layer_idx, cett_target=0.2, n_quantiles=500):
    model.activation_capture = model.ACTIVATION_CAPTURE(model)
    model.activation_capture.register_hooks(hooks=[Hook.UP])

    thresholds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            model.activation_capture.clear_captures()
        
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            activations = model.activation_capture.mlp_activations[Hook.UP][layer_idx]
            activations = activations.view(-1, activations.size(-1))

            for i in range(activations.size(0)):
                neuron_outputs = activations[i] * model.model.layers[0].mlp.down_proj.weight
                threshold = calculate_threshold_one_token(neuron_outputs, cett_target=cett_target, n_quantiles=n_quantiles)
                thresholds.append(threshold)

    return sum(thresholds)/len(thresholds)
                