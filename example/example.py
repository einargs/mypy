from typing import TypeVar, Generic, Annotated, Literal, Tuple
from typing_extensions import TypeAlias
from refinement import RefinementVar
# import torch

X = RefinementVar('X')

x: Annotated[int, X, 1 < 5 < 6] = 0


class array:
    __slots__ = ('shape', 'dtype', 'device')

    def __init__(self, shape: tuple[int, ...], dtype: str, device: str) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device


V = RefinementVar('V')
Y = RefinementVar('Y')
Z = RefinementVar('Z')
A = RefinementVar('A')
B = RefinementVar('B')
C = RefinementVar('C')

T = TypeVar('T')
S = TypeVar('S')


def f0(a: T) -> T:
    return a


def f1(a: T, b: S) -> Annotated[Tuple[T, S], V[Expand]]:
    return (a, f0(b))


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride


# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         # self.conv1: nn.Conv2d
#         # havoc self.conv1
#         # assume self.conv1.in_channels == 1 and ...
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)
# 
#     def forward(
#             self,
#             input: Annotated[Tensor, T, T.shape == (64, 1, 28, 28)]
#     ) -> Annotated[Tensor, R, R.shape == (64, 10)]:
#         # self: Net
#         # havoc self
#         x = self.conv1(input)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, (2,2), (2,2), (0,0), (1,1))
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1, 4)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
# 
