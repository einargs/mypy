from typing import Tuple
from typing_extensions import Annotated
from refinement import RefinementVar

class Tensor:
    shape: Tuple[int, ...]

T = RefinementVar('T')
S = RefinementVar('S')

SD = RefinementVar('SD')
ED = RefinementVar('ED')

def flatten1(
        t: Annotated[Tensor, T],
) -> Annotated[Tensor, S,
        S.shape[0] == T.shape[0],
        S.shape[1] == T.shape[1] * T.shape[2] * T.shape[3]]: ...
