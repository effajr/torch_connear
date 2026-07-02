import os
from typing import Tuple
import warnings

import torch
from torch import nn, Tensor

from connear.backbone import CoNNear, Encoder1DLayer, Encoder2DLayer, Decoder2DLayer
from config.cf import CF
from config.path import CP_DIR


class CoNNearPeriphery(nn.Module):
    """Wrapper module implementing the full cochlear processing pipeline.

    CoNNearPeriphery combines three neural network submodules to simulate
    the auditory periphery: the cochlea, inner hair cells (IHC), and
    auditory nerve fibers (ANF). The cochlea simulates mechanical
    displacement of the basilar membrane. The IHC stage simulates synaptic
    transduction and neural adaptation. The ANF stage simulates responses
    of three fiber types with different spontaneous firing rates (high,
    medium, and low).

    All submodules can be selectively enabled or disabled, and their
    pretrained weights from the original publication can be automatically
    loaded. This implementation uses the original sequential model.
    The more efficient branched variant, with a shared ANF encoder and three
    separate ANF decoders, may be added in a future release.

    When `load_weights` is `True`, the class loads pretrained parameters and
    computes the center frequencies of the simulated cochlear channels from
    the predefined configuration.

    Attributes:
        n_cf (int): Number of simulated cochlear frequency channels.
            Determines the dimensionality of the cochlear representation.
            Must be 21 or 201.
        use_ihc (bool): Whether to include the inner hair cell submodule
            in the processing pipeline.
        use_anf (bool): Whether to include the auditory nerve fiber
            submodules in the processing pipeline. Requires `use_ihc=True`.
        sum_anf (bool): If `True` and `use_anf=True`, the outputs of the
            three ANF submodules are summed with weights from
            `anf_distribution`. If `False`, outputs are concatenated along
            the channel dimension.
        anf_distribution (Tuple[int, int, int]): Weights applied to the
            high, medium, and low spontaneous-rate ANF outputs before
            summation. Defaults to `(3, 3, 13)`, corresponding to the
            natural distribution of innervation types (L, M and H fiber 
            counts) to a single IHC.
        pad_context (bool): Whether to pad the input with zeros over the
            context region so that the input and output have matching
            lengths. Propagated to all submodules.
        load_weights (bool): Whether to load pretrained weights from the
            original publication. If `False`, a warning is issued. Defaults
            to `True`.
        cochlea (CoNNearCochlea): The cochlear submodule.
        ihc (CoNNearIHC): The inner hair cell submodule. Only set if
            `use_ihc=True`.
        anf_l, anf_m, anf_h (CoNNearANF): Low, medium, and high SR
            auditory nerve fiber submodules. Only set if `use_anf=True`.
        cfs (Tensor): Center frequencies (in Hz) of the simulated cochlear
            channels. Shape `(n_cf,)`. Only set if `load_weights=True`.

    Args:
        n_cf (int): See `Attributes`. Must be 21 or 201. Defaults to `201`.
        use_ihc (bool): See `Attributes`. Defaults to `True`.
        use_anf (bool): See `Attributes`. Defaults to `True`.
        sum_anf (bool): See `Attributes`. Defaults to `True`.
        anf_distribution (Tuple[int, int, int]): See `Attributes`.
            Defaults to `(3, 3, 13)`.
        load_weights (bool): See `Attributes`. Defaults to `True`.
        pad_context (bool): See `Attributes`. Defaults to `True`.

    Raises:
        ValueError: If `n_cf` is not 21 or 201, or if `use_anf=True` but
            `use_ihc=False`.
    """
    def __init__(self,
                 n_cf: int=201,
                 use_ihc: bool=True,
                 use_anf: bool=True, 
                 sum_anf: bool=True,
                 anf_distribution: Tuple[int, int, int]=(3, 3, 13),
                 load_weights: bool=True,
                 pad_context: bool=True) -> None:
        super().__init__()
        
        if n_cf not in {21, 201}:
            raise ValueError("Invalid number of cochlear frequency channels.")
        if use_anf and not use_ihc:
            raise ValueError("Cannot use ANF without IHC.")
        
        # Pad with zeros in the context region if specified
        self.pad_context = pad_context
        # Number of frequency channels
        self.n_cf = n_cf
        # Summation of the ANR fibre responses
        self.sum_anf = sum_anf
        self.anf_distribution = anf_distribution
        # Flags to activate/deactivate submodules
        self.use_ihc = use_ihc
        self.use_anf = use_anf
        self.cochlea = CoNNearCochlea(n_cf=self.n_cf, pad_context=self.pad_context)
        if load_weights:
            self._load_weights()
            self._define_cf()
        else:
            warnings.warn("Warning: It is not recommended to use CoNNearPeriphery without loading the learned parameters.")
            
    def _load_weights(self) -> None:
        self.cochlea.load_state_dict(torch.load(os.path.join(CP_DIR, f"cochlea_{self.n_cf}cf.pth"), weights_only=True))
        if self.use_ihc:
            # Disable output cropping if no context was padded, to preserve input/output alignment
            if not self.pad_context:
                self.cochlea.crop = False
            self.ihc = CoNNearIHC(pad_context=self.pad_context)
            self.ihc.load_state_dict(torch.load(os.path.join(CP_DIR,"ihc.pth"), weights_only=True))
        if self.use_anf:
            if not self.pad_context:
                self.ihc.crop = False
            for anf_type in ["l", "m", "h"]:
                anf = CoNNearANF(anf_type=anf_type, pad_context=self.pad_context)
                anf.load_state_dict(torch.load(os.path.join(CP_DIR, f"anf{anf_type}.pth"), weights_only=True))
                setattr(self, f"anf_{anf_type}", anf)

    def _define_cf(self) -> None:
        self.cfs = torch.tensor(CF) * 1e3
        if self.n_cf == 21:
            self.cfs = self.cfs[::10]

    def forward(self, x: Tensor) -> Tensor:
        y = self.cochlea(x)
        if self.use_ihc:
            y = self.ihc(y)
        if self.use_anf:
            y_l = self.anf_l(y)
            y_m = self.anf_m(y)
            y_h = self.anf_h(y)
            if not self.sum_anf:
                return torch.cat([y_l, y_m, y_h], dim=1)
            y = sum(factor*resp for factor, resp in zip(self.anf_distribution, (y_l, y_m, y_h)))
        return y
    

class CoNNearCochlea(CoNNear):
    def __init__(self, n_cf=201, encoder_is_1d: bool=True, crop: bool=True, pad_context: bool=False) -> None:
        super().__init__(encoder_is_1d=encoder_is_1d, crop=crop, pad_context=pad_context)
        self.n_cf = n_cf
        self.encoder = nn.ModuleList([Encoder1DLayer(in_channels=1),
                                      Encoder1DLayer(),
                                      Encoder1DLayer(),
                                      Encoder1DLayer()])
        self.decoder = nn.ModuleList([Decoder2DLayer(kernel_size=(1, 64), activation="tanh", use_bias=False),
                                      Decoder2DLayer(in_channels=256, kernel_size=(1, 64), activation="tanh", use_bias=False),
                                      Decoder2DLayer(in_channels=256, kernel_size=(1, 64), activation="tanh", use_bias=False),
                                      Decoder2DLayer(in_channels=256, out_channels=self.n_cf, kernel_size=(1, 64), activation="identity", use_bias=False)])


class CoNNearIHC(CoNNear):
    def __init__(self, crop: bool=True, pad_context: bool=False) -> None:
        super().__init__(crop=crop, pad_context=pad_context)
        self.encoder = nn.ModuleList([Encoder2DLayer(in_channels=1),
                                      Encoder2DLayer(),
                                      Encoder2DLayer()])
        self.decoder = nn.ModuleList([Decoder2DLayer(),
                                      Decoder2DLayer(in_channels=256),
                                      Decoder2DLayer(in_channels=256, out_channels=1, activation="identity")])


class CoNNearANF(CoNNear):
    def __init__(self, anf_type: str, pad_context: bool=False) -> None:
        super().__init__(cropping=(7936, 256), pad_context=pad_context)
        if anf_type == "l":
            self.encoder = nn.ModuleList([Encoder2DLayer(in_channels=1 if i == 0 else 64, 
                                                        out_channels=64, 
                                                        kernel_size=(1, 8),
                                                        activation="tanh")
                                        for i in range(14)])
            self.decoder = nn.ModuleList([Decoder2DLayer(in_channels=64 if i==0 else 128, 
                                                        out_channels=1 if i == 13 else 64, 
                                                        kernel_size=(1, 8),
                                                        activation="identity" if i == 13 else "sigmoid")
                                        for i in range(14)])
        elif anf_type == "m" or anf_type == "h":
            self.encoder = nn.ModuleList([Encoder2DLayer(in_channels=1 if i == 0 else 64, 
                                                        out_channels=64, 
                                                        kernel_size=(1, 8),
                                                        activation="prelu")
                                        for i in range(14)])
            self.decoder = nn.ModuleList([Decoder2DLayer(in_channels=64 if i==0 else 128, 
                                                        out_channels=1 if i == 13 else 64, 
                                                        kernel_size=(1, 8),
                                                        activation="identity" if i == 13 else "prelu")
                                        for i in range(14)])
        else:
            raise ValueError(f"Variable anf_type should be 'l', 'm' or 'h', got {anf_type} instead.")
