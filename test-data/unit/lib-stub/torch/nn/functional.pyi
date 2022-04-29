from refinement import RefinementVar
from typing import Tuple, Union
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
        t: Annotated[Tensor, T, len(T.shape) == 4],
        kernel_size: Annotated[Union[int, Tuple[int, int]], KS[Expand]],
        stride: Annotated[Union[int, Tuple[int, int]], SD[Expand]],
        padding: Annotated[Union[int, Tuple[int, int]], PD[Expand]]=(0,0),
        dilation: Annotated[Union[int, Tuple[int, int]], DL[Expand]]=(1,1)
) -> Annotated[Tensor, S,
        S.shape == (T.shape[0],
            T.shape[1],
            (T.shape[2] + 2 * PD[0] - (KS[0] - 1) - 1) // SD[0] + 1,
            (T.shape[3] + 2 * PD[1] - (KS[1] - 1) - 1) // SD[1] + 1),
        ]: ...

def log_softmax(
        t: Annotated[Tensor, T],
        dim: int
) -> Annotated[Tensor, S,
        T.shape == S.shape]: ...
