from typing import Tuple
from torch import Tensor
from typing_extensions import Annotated
from refinement import RefinementVar, RSelf, Const

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
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]

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
            RSelf.kernel_size == KS,
            RSelf.stride == SD,
            RSelf.padding == PD,
            RSelf.dilation == DL]: ...

    def __call__(
            self: Annotated['Conv2d', SELF],
            t: Annotated[Tensor, T,
                len(T.shape) == 4,
                T.shape[1] == SELF.in_channels]
    ) -> Annotated[Tensor, S,
            S.shape == (T.shape[0],
                SELF.out_channels,
                (T.shape[2] + 2
                    * SELF.padding[0]
                    - SELF.dilation[0]
                    * (SELF.kernel_size[0] - 1)
                    - 1) // SELF.stride[0] + 1,
                (T.shape[3] + 2
                    * SELF.padding[1]
                    - SELF.dilation[1]
                    * (SELF.kernel_size[1] - 1)
                    - 1) // SELF.stride[1] + 1),
            ]: ...

class Dropout:
    def __init__(self, p: float): ...

    def __call__(self, t: Annotated[Tensor, T]) -> Annotated[Tensor, S,
            T.shape == S.shape]: ...

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
            t: Annotated[Tensor, T,
                len(T.shape) == 2,
                T.shape[1]==SELF.in_features]
    ) -> Annotated[Tensor, S,
            S.shape == (T.shape[0], SELF.out_features)]: ...
