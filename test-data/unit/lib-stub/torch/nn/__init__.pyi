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

class Conv2d:
    def __init__(
            self,
            in_channels: Annotated[int, IC],
            out_channels: Annotated[int, OC],
            kernel_size: Annotated[Tuple[int, int], KS],
            stride: Annotated[Tuple[int, int], SD],
            padding: Annotated[Tuple[int, int], PD]=(0,0),
            dilation: Annotated[Tuple[int, int], DL]=(0,0),
    ) -> Annotated[None,
            RSelf.in_channels == IC,
            RSelf.out_channels == OC,
            RSelf.kernel_size == KS,
            RSelf.stride == SD]:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, t: Annotated[Tensor, T]
            ) -> Annotated[Tensor, S,
                    S.shape[0] == (T.shape[0] + 2
                        * RSelf.padding[0]
                        - RSelf.dilation[0]
                        * (RSelf.kernel_size[0] - 1)
                        - 1) // RSelf.stride[0] + 1,
                    S.shape[1] == (T.shape[1] + 2
                        * RSelf.padding[1]
                        - RSelf.dilation[1]
                        * (RSelf.kernel_size[1] - 1)
                        - 1) // RSelf.stride[1] + 1,
                    ]: ...

class Dropout:
    def __init__(self, p: float): ...

    def __call__(self, t: Annotated[Tensor, T]) -> Annotated[Tensor, S,
            T.shape[0] == S.shape[0], T.shape[1] == S.shape[1]]: ...

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
