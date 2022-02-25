from typing import TypeVar, Generic, Annotated
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


class Container:
    def __init__(self, value: int):
        self.value = value


def minimal(m: Annotated[Container, A, A.value > 2]) -> Annotated[int, B, B > 1]:
    return m.value

#def test(m: Annotated[int, V]) -> None:
#    return None
#
#
#def matmul(m1: Annotated[array, V, V.shape == (A, B)],
#        m2: Annotated[array, V, V.shape == (B, C)]
#        ) -> Annotated[array, V, V.shape == (A, C)]:
#    return array((m1.shape[0], m2.shape[1]), m1.dtype, m1.device)
#
#
#def reshape(m: Annotated[array, A],
#        shape: Annotated[tuple[int, ...], B]
#        ) -> Annotated[None, V, A.shape == B]:
#    return None
#
#
#def other(m: Annotated[int, V, V == 4]):
#    return None


#def upsample_m(signal: torch.Tensor, y: Annotated[int, X]) -> Annotated[torch.Tensor, Y]:
#    ''' signal is a batch of 1D tensors; M is an integer '''
#    B = signal.shape[0]
#    L = signal.shape[1]
#
#    up = torch.zeros((B, int(y*L)), dtype=signal.dtype, device=signal.device)
#    up[:, ::y] = signal
#
#    return up


# def downsample_m(signal, M):
#     return signal[:, ::M]
# 
# 
# def batch_to_vol(signal, D):
#     ''' signal is a batch of 1D tensors; BxL '''
#     L = signal.shape[1]
#     return signal.reshape(D, D, L).numpy()
# 
# 
# def vol_to_batch(signal):
#     return torch.from_numpy(signal).flatten(start_dim=0, end_dim=1)
# 
# 
# def batch_2d_to_1d(signal):
#     return signal.flatten(start_dim=0, end_dim=1)
# 
# 
# def batch_1d_to_2d(signal, D):
#     L = signal.shape[1]
#     B = signal.shape[0] // D
#     return signal.reshape(B, D, L)
# 
# 
# def advance(signal, amt):
#     ''' signal is a batch of 1D Torch tensors; BxL '''
#     if amt == 0:
#         return signal
#     return F.pad(signal[:, amt:], (0, amt), mode='constant', value=0.0)
# 
# 
# def delay(signal, amt):
#     ''' signal is a batch of 1D Torch tensors; BxL '''
#     if amt == 0:
#         return signal
#     return F.pad(signal[:, :-amt], (amt, 0), mode='constant', value=0.0)
