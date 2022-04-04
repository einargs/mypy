from typing import Tuple
from typing_extensions import Annotated
from refinement import RefinementVar

class Tensor:
    shape: Tuple[int, ...]

T = RefinementVar('T')
S = RefinementVar('S')

SD = RefinementVar('SD')
ED = RefinementVar('ED')

def flatten(
        t: Annotated[Tensor, T],
        start: Annotated[int, SD],
        end: Annotated[int, ED]
) -> Annotated[Tensor, S,
        S.shape == fold(lambda acc, cur: acc * cur, T.shape[SD:ED])]: ...
