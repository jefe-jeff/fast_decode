import torch
import numpy as np

def toeplitz(x, window):
    return x.unfold(1, window, 1)

def signal_window_dot(x, w, b):
    return (x.unsqueeze(3) * w).sum(2) + b

def random_weight(shape, device, dtype):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape, device, dtype):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

def three_layer_decode(x, params, window = 31):
    w1, b1, w2, b2, w3, b3 = params
    
    x.squeeze(0)
    data = toeplitz(x, window)

    
    data = signal_window_dot(data, w1, b1).clamp(0)
    data = signal_window_dot(data, w2, b2).clamp(0)
    data = signal_window_dot(data, w3, b3).squeeze().sum(2)
    return data