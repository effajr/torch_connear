from typing import Optional, Tuple

import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CoNNear(nn.Module):
    """Backbone U-Net architecture used throughout the CoNNear framework.

    This class implements the logic shared by every CoNNear submodule:
    input padding, encoding, decoding with skip connections, and output
    cropping. Subclasses are expected to populate `encoder` and `decoder`
    with the appropriate layers.

    According to the original work, during training and evaluation, the
    input signals are segmented into fixed-size windows that areprocessed
    independently. Because CoNNear resets its internal state at the
    start of each simulation, naively concatenating the outputs of
    consecutive windows can introduce discontinuities at the window
    boundaries. To mitigate this, the model can be given extra context 
    samples on both sides of each window (see `pad_context`), and the
    corresponding region can later be removed from the output with a
    final cropping layer (see `crop` and `cropping`).

    Separately, since the encoder is made of a fixed number of strided
    convolutional layers, the input length must be a multiple of
    2^(number of encoder layers). When enabled, `pad_multiple` zero-pads
    the input to satisfy this constraint and removes the corresponding
    padding from the output.

    Attributes:
        encoder_is_1d (bool): Whether the encoder operates on 1D tensors, as in
            the original TensorFlow implementation of the CoNNear cochlea
            encoder. Determines whether tensors are reshaped between the
            encoder and the decoder.
        pad_context (bool): Whether to pad the input with zeros over the context
            region so that the input and output have matching lengths.
        cropping (Tuple([int, int])): Number of context samples, `(left, right)`, 
            padded to the input and/or cropped from the output.
        crop (bool): Whether to crop the context region from the output before
            returning it.
        pad_multiple (bool): Whether to zero-pad the input so that its length is
            a multiple of 2^(number of encoder layers), as required by the
            strided convolutional encoder.
        encoder (nn.ModuleList): Module list of encoder layers, defined by subclasses.
        decoder (nn.ModuleList): Module list of decoder layers, defined by subclasses.

    Args:
        encoder_is_1d (bool): See `Attributes`. Defaults to `False`.
        pad_context (bool): See `Attributes`. Defaults to `True`.
        cropping (Tuple[int, int]): See `Attributes`. Defaults to `(256, 256)`.
        crop (bool): See `Attributes`. Defaults to `True`.
        pad_multiple (bool): See `Attributes`. Defaults to `True`.
    """
    def __init__(self,
                 encoder_is_1d: bool=False,
                 pad_context: bool=True,
                 cropping: Tuple[int, int]=(256, 256),
                 crop: bool=True,
                 pad_multiple: bool=True) -> None:
        super().__init__()
        
        self.encoder_is_1d = encoder_is_1d
        self.cropping = cropping
        self.crop = crop
        self.pad_context = pad_context
        self.pad_multiple = pad_multiple

        self.encoder: nn.ModuleList
        self.decoder: nn.ModuleList
        
    def pad_to_context(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.cropping)
        return x

    def pad_to_multiple(self, x: Tensor) -> Tuple[Tensor, int]:
        n = 2**len(self.encoder)
        length = x.size(-1)
        remainder = length % n
        # Return x if no padding needed
        if remainder == 0:
            return x, None
        pad_amount = n - remainder
        x = F.pad(x, (0, pad_amount))
        return x, -pad_amount

    def forward(self, x: Tensor) -> Tensor:
        # Reshape input
        if x.ndim < 3:
            x = x.unsqueeze(dim=1)
        if x.shape[-1] == 1:
            x = x.transpose(1, 2)

        # Pad with zeros in the context region if specified
        if self.pad_context:
            x = self.pad_to_context(x)
        # Pad input to target length
        if self.pad_multiple:
            x, stop_index = self.pad_to_multiple(x)
        else: 
            stop_index = -1

        # Store encoder outputs for skip connections
        encoder_outputs = []
        y = x
        for i, encoder_layer in enumerate(self.encoder):
            y, x = encoder_layer(y)
            if i < len(self.encoder) - 1:  # Don't store last encoder output
                encoder_outputs.append(x)

        # Reshape before decoder if needed
        if self.encoder_is_1d:
            y = y.unsqueeze(2)
            encoder_outputs = [skip.unsqueeze(2) for skip in encoder_outputs]

        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            if i > 0:
                skip = encoder_outputs.pop()
                # If needed, adjust lengths before concatenation
                if skip.size(-1) != y.size(-1):
                    min_len = min(skip.size(-1), y.size(-1))
                    y = y[..., :min_len]
                    skip = skip[..., :min_len]
                y = torch.cat([y, skip], dim=1)
            y = decoder_layer(y)

        if self.encoder_is_1d:
            y = y.swapaxes(1, 2)

        # Crop the output (equivalent to Keras Cropping1D)
        if self.crop and y.size(-1) > sum(self.cropping):
            y = y[..., self.cropping[0]:-self.cropping[1]]
        
        # Crop to remove the padding added by pad_to_multiple
        y = y[..., :stop_index]
        return y


def make_activation(name: str, num_params: Optional[int]=None) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    elif name == "prelu":
        return nn.PReLU(num_parameters=num_params, init=0.0)
    raise ValueError(f"Unsupported activation: {name}")


class BaseEncoderLayer(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride,) if isinstance(stride, int) else stride
        
        self.conv: nn.Module
        self.activation: nn.Module
        
    def apply_padding(self, x: Tensor, kernel_size: int, stride: int, dilation: int=1) -> Tensor:
        w_in = x.shape[-1]
        w_out = math.ceil(w_in/2)
        padding = max(0, (w_out-1)*stride[-1] + dilation*(kernel_size[-1]-1) + 1 - w_in)
        padding_left = padding // 2
        padding_right = padding - padding_left
        x = F.pad(x, (padding_left, padding_right))     # apply asymmetric padding to reproduce tensorflow "same"
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.apply_padding(x, self.kernel_size, self.stride)
        x = self.conv(x)
        a = self.activation(x)
        return a, x     # return both post and pre-activation for skip connections


class Encoder1DLayer(BaseEncoderLayer):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=64, stride=2) -> None:
        super().__init__(kernel_size, stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.activation = make_activation(name="tanh")


class Encoder2DLayer(BaseEncoderLayer):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=(1, 16), stride=(1, 2), activation="tanh") -> None:
        super().__init__(kernel_size, stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=True)
        self.activation = make_activation(name=activation, num_params=out_channels)


class Decoder2DLayer(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=(1, 16), stride=(1, 2), use_bias=True, activation="sigmoid") -> None:
        super().__init__()
        padding = tuple((k-s)//2 for k, s in zip(kernel_size, stride))
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        self.activation = make_activation(name=activation, num_params=out_channels)

    def forward(self, x) -> Tensor:
        return self.activation(self.conv_t(x))
