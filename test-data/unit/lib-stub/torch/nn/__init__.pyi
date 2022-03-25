from typing import Tuple
from torch import Tensor
from typing_extensions import Annotated
from refinement import RefinementVar, RSelf

class Module:
    pass

IC = RefinementVar('IC')
OC = RefinementVar('OC')
KS = RefinementVar('KS')
SD = RefinementVar('SD')
PD = RefinementVar('PD')
DL = RefinementVar('DL')

T = RefinementVar('T')
S = RefinementVar('S')

SELF = RefinementVar('SELF')

class Conv2d:
    def __init__(
            self,
            in_channels: Annotated[int, IC],
            out_channels: Annotated[int, OC],
            kernel_size: Annotated[Tuple[int, int], KS],
            stride: Annotated[Tuple[int, int], SD],
            padding: Annotated[Tuple[int, int], PD],
            dilation: Annotated[Tuple[int, int], DL],
    ) -> Annotated[None,
            RSelf.in_channels == IC,
            RSelf.out_channels == OC,
            RSelf.kernel_size[0] == KS[0],
            RSelf.kernel_size[1] == KS[1],
            RSelf.stride[0] == SD[0],
            RSelf.stride[1] == SD[1],
            RSelf.padding[0] == PD[0],
            RSelf.padding[1] == PD[1],
            RSelf.dilation[0] == DL[0],
            RSelf.dilation[1] == DL[1]]:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(
            self: Annotated['Conv2d', SELF],
            t: Annotated[Tensor, T, T.shape[1] == SELF.in_channels]
    ) -> Annotated[Tensor, S,
            S.shape[0] == T.shape[0],
            S.shape[1] == SELF.out_channels,
            S.shape[2] == (T.shape[2] + 2
                * SELF.padding[0]
                - SELF.dilation[0]
                * (SELF.kernel_size[0] - 1)
                - 1) // SELF.stride[0] + 1,
            S.shape[3] == (T.shape[3] + 2
                * SELF.padding[1]
                - SELF.dilation[1]
                * (SELF.kernel_size[1] - 1)
                - 1) // SELF.stride[1] + 1,
            ]: ...

class Dropout4:
    def __init__(self, p: float): ...

    def __call__(self, t: Annotated[Tensor, T]) -> Annotated[Tensor, S,
            T.shape[0] == S.shape[0],
            T.shape[1] == S.shape[1],
            T.shape[2] == S.shape[2],
            T.shape[3] == S.shape[3],
            ]: ...

class Dropout2:
    def __init__(self, p: float): ...

    def __call__(self, t: Annotated[Tensor, T]) -> Annotated[Tensor, S,
            T.shape[0] == S.shape[0],
            T.shape[1] == S.shape[1],
            ]: ...

IF = RefinementVar('IF')
OF = RefinementVar('OF')

class Linear:
    def __init__(
            self,
            in_features: Annotated[int, IF],
            out_features: Annotated[int, OF]
    ) -> Annotated[None,
            RSelf.in_features == IF,
            RSelf.out_features == OF]:
        self.in_features = in_features
        self.out_features = out_features

    def __call__(
            self: Annotated['Linear', SELF],
            t: Annotated[Tensor, T, T.shape[1]==SELF.in_features]
    ) -> Annotated[Tensor, S,
            S.shape[0] == T.shape[0],
            S.shape[1] == SELF.out_features]: ...