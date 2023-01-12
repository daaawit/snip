import copy
import types

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def print_scores(keep_masks, names):
    """Printing in style of my master's thesis for sanity checking

    Args:
        keep_masks (list): List of masks produced by SNIP
        names (_type_): List of names of pruned layers (for printing style only)
    """
    head_str = f"| {'Layer':<32}| {'Before':<14}| {'After':<14}| {'Ratio':<10}|"
    head_sep = "=" * len(head_str)
    print(head_sep)
    print(head_str)
    print(head_sep)
    
    full_numel = 0
    full_numprune = 0
    for name, mask in zip(names, keep_masks): 

        numel = torch.numel(mask)
        numprune = torch.numel((mask == 1))
        ratio = round(numprune / numel, 4)
        
        layer_info = f"| - {name:<30}| {numel:<14}| {numprune:<14}| {ratio:<10}|"
        print(layer_info)
        
        full_numel += numel
        full_numprune += numprune
    
    print(head_sep, "\n")
    return full_numel, full_numprune

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False # TODO: This might be the difference to my implementation? 

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()

    grads_abs = []
    names = [] # Store names for later printing
    for name, layer in net.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            grads_abs.append(torch.abs(layer.weight_mask.grad))
            names.append(name)

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    numel, numprune = print_scores(keep_masks, names)
    
    print(f"- Intended prune ratio:\t{keep_ratio}")
    print(f"- Actual prune ratio:\t{numprune / numel}")
    print(f"- Threshold:           {acceptable_score}")

    return(keep_masks)

