from refinement import RefinementVar
from typing import Tuple
from typing_extensions import Annotated
from torch import Tensor

T = RefinementVar('T')
S = RefinementVar('S')

def relu(t: Annotated[Tensor, T]) -> Annotated[Tensor, S,
        T.shape == S.shape]: ...

KS = RefinementVar('KS')
SD = RefinementVar('SD')
PD = RefinementVar('PD')
DL = RefinementVar('DL')

def max_pool2d(
        t: Annotated[Tensor, T],
        kernel_size: Annotated[Tuple[int, int], KS],
        stride: Annotated[Tuple[int, int], SD],
        padding: Annotated[Tuple[int, int], PD],
        dilation: Annotated[Tuple[int, int], DL]
) -> Annotated[Tensor, S,
        T.shape[0] == S.shape[0],
        T.shape[1] == S.shape[1],
        S.shape[2] == (T.shape[2] + 2*PD[0] - (KS[0] - 1) - 1)//SD[0] + 1,
        S.shape[3] == (T.shape[3] + 2*PD[1] - (KS[1] - 1) - 1)//SD[1] + 1,
        ]: ...

def log_softmax(
        t: Annotated[Tensor, T],
        dim: int
) -> Annotated[Tensor, S,
        T.shape == S.shape]: ...
