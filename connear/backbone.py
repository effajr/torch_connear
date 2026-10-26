from typing import Optional, Tuple

import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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

    To reduce memory usage when gradients are needed, the forward pass can
    use activation checkpointing (see `use_checkpoint`): the intermediate
    activations of the U-Net are not stored during the forward pass and are
    recomputed during the backward pass, at the cost of roughly one extra
    forward pass. For modules with a 2D encoder, the input can additionally
    be split along the frequency axis into several checkpointed segments
    (see `checkpoint_chunks`), which further lowers the peak memory. All 2D
    layers use kernels of height 1 with unit stride and no padding along
    that axis, and the other operations act along the time or channel axes
    only, so every frequency channel is processed independently and the
    split does not change the result (up to floating-point rounding).

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
        use_checkpoint (bool): Whether to use activation checkpointing
            (`torch.utils.checkpoint.checkpoint`, non-reentrant variant).
            Only active when gradients are recorded, i.e. when grad mode is
            enabled and the input or a parameter requires gradients;
            otherwise the regular forward pass is used.
        checkpoint_chunks (int): Number of segments the input is split into
            along the frequency axis (dim 2 of a `(B, C, n_cf, T)` input)
            when checkpointing is active. The peak memory of the backward
            pass is roughly divided by this number. Only used by modules
            with a 2D encoder receiving 4D inputs; ignored otherwise. `1`
            disables the split.
        encoder (nn.ModuleList): Module list of encoder layers, defined by subclasses.
        decoder (nn.ModuleList): Module list of decoder layers, defined by subclasses.

    Args:
        encoder_is_1d (bool): See `Attributes`. Defaults to `False`.
        pad_context (bool): See `Attributes`. Defaults to `True`.
        cropping (Tuple[int, int]): See `Attributes`. Defaults to `(256, 256)`.
        crop (bool): See `Attributes`. Defaults to `True`.
        pad_multiple (bool): See `Attributes`. Defaults to `True`.
        use_checkpoint (bool): See `Attributes`. Defaults to `False`.
        checkpoint_chunks (int): See `Attributes`. Defaults to `1`.

    Raises:
        ValueError: If `checkpoint_chunks` is lower than 1.
    """
    def __init__(self,
                 encoder_is_1d: bool=False,
                 pad_context: bool=True,
                 cropping: Tuple[int, int]=(256, 256),
                 crop: bool=True,
                 pad_multiple: bool=True,
                 use_checkpoint: bool=False,
                 checkpoint_chunks: int=1) -> None:
        super().__init__()

        if checkpoint_chunks < 1:
            raise ValueError(f"checkpoint_chunks must be >= 1, got {checkpoint_chunks} instead.")

        self.encoder_is_1d = encoder_is_1d
        self.cropping = cropping
        self.crop = crop
        self.pad_context = pad_context
        self.pad_multiple = pad_multiple
        self.use_checkpoint = use_checkpoint
        self.checkpoint_chunks = checkpoint_chunks

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
        # Regular forward pass when no gradient is recorded (inference, frozen
        # model on an input that does not require gradients) or when disabled
        if not (self.use_checkpoint and self._records_grad(x)):
            return self._forward(x)

        n_chunks = 1
        if self._can_split(x):
            n_chunks = min(self.checkpoint_chunks, x.size(2))
        if n_chunks == 1:
            return checkpoint(self._forward, x, use_reentrant=False)

        # Split along the frequency axis: each segment is checkpointed separately,
        # so that only one segment's activations are recomputed at a time
        self._check_frequency_independence()
        chunks = torch.tensor_split(x, n_chunks, dim=2)
        return torch.cat([checkpoint(self._forward, chunk, use_reentrant=False) for chunk in chunks], dim=2)

    def _records_grad(self, x: Tensor) -> bool:
        if not torch.is_grad_enabled():
            return False
        return x.requires_grad or any(p.requires_grad for p in self.parameters())

    def _can_split(self, x: Tensor) -> bool:
        # The frequency axis is dim 2 of the 4D inputs (B, C, n_cf, T) of 2D modules.
        # Inputs whose last dimension is 1 are transposed in _forward, so they are not split.
        return not self.encoder_is_1d and x.ndim == 4 and x.size(-1) != 1

    def _check_frequency_independence(self) -> None:
        # Splitting along the frequency axis is only exact if no layer mixes frequency channels
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                padding = 0 if isinstance(module.padding, str) else module.padding[0]
                independent = (module.kernel_size[0] == 1
                               and module.stride[0] == 1
                               and module.dilation[0] == 1
                               and padding == 0)
                if isinstance(module, nn.ConvTranspose2d):
                    independent = independent and module.output_padding[0] == 0
                if not independent:
                    raise ValueError(f"Cannot split the input along the frequency axis: {module} "
                                     "mixes frequency channels. Use checkpoint_chunks=1.")

    def _forward(self, x: Tensor) -> Tensor:
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
