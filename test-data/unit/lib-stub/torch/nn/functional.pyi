from refinement import RefinementVar
from typing_extensions import Annotated

T = RefinementVar('T')
S = RefinementVar('S')

def relu(t: Annotated[Tensor, T]) -> Annotated[Tensor, S,
        S.shape[0] == T.shape[0], S.shape[1] == T.shape[1]]: ...
