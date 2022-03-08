from typing import TypeVar, Generic, Annotated, Literal
from typing_extensions import TypeAlias
from refinement import RefinementVar
# import torch


class array:
    __slots__ = ('shape', 'dtype', 'device')

    def __init__(self, shape: tuple[int, ...], dtype: str, device: str) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device


V = RefinementVar('V')
X = RefinementVar('X')
Y = RefinementVar('Y')
Z = RefinementVar('Z')
A = RefinementVar('A')
B = RefinementVar('B')
C = RefinementVar('C')

T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
T4 = TypeVar('T4')


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride


c = Conv2d(1, 1, 2, 3)
reveal_type(c)
