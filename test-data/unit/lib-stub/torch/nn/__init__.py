from torch import Tensor
from refinement import RefinementVar, RSelf

class Module:
    pass

IC = RefinementVar('IC')
OC = RefinementVar('OC')
KS = RefinementVar('KS')
SD = RefinementVar('SD')
T = RefinementVar('T')
S = RefinementVar('S')

class Conv2d:
    def __init__(
            self,
            in_channels: Annotated[int, IC],
            out_channels: Annotated[int, OC],
            kernel_size: Annotated[int, KS],
            stride: Annotated[int, SD]
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
                    # TODO: a bunch of fucking math
                    ]: ...
