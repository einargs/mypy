from typing import TypeVar, Generic, Annotated, Literal
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


def test_literal() -> Annotated[int, A, A > 5]:
    return 9


def test_tuple(
        m1: Annotated[array, V, V.shape == (1, 1)],
        m2: Annotated[array, A, A.shape == (2, 2)]
) -> Annotated[array, B, B.shape == (2, 2)]:
    return m2


def test_tuple_member(
        v: Annotated[array, V, V.shape == (1, 5)]
) -> Annotated[int, A, A > 2]:
    return v.shape[1]


def test_assignment(
    v: Annotated[array, V, V.shape == (1, 5)]
) -> Annotated[int, A, A > 2]:
    a: Annotated[int, B, B > 3] = v.shape[1]
    b = a
    return b


def test_call_returns() -> Annotated[int, A, A > 0]:
    a: Annotated[int, V, V > 3] = test_literal()
    print(a)
    return test_literal()


def dummy_call1(v: Annotated[int, V, V > 5]) -> Annotated[int, B, B > 4]:
    return v


def dummy_call2(m: Annotated[int, A, A > 7]) -> Annotated[int, B, B > 6]:
    return m


def test_call_args(m: Annotated[int, V, V > 9]) -> Annotated[int, A, A > 0]:
    other: Annotated[int, Y, Y > 8] = 9
    a: Annotated[int, B, B > 3] = dummy_call1(dummy_call2(8))
    b: Annotated[int, C, C > 3] = other
    c: Annotated[int, X, X > 3] = m
    print(b, c)
    return a


# Should have failures.
def test_assignment_fails(
    v: Annotated[array, V, V.shape == (1, 5)]
) -> Annotated[int, A, A > 0]:
    a: Annotated[int, B, B > 3] = v.shape[1]
    b = a

    c: Annotated[int, C, C > 4] = v.shape[1]  # should fail
    d: Annotated[int, X, X > 1] = a  # should fail
    e = d
    # f type checks even though d doesn't, because the errors are isolated.
    f: Annotated[int, Y, Y > 0] = e
    print(a, b, c, d, f)  # just using them to stop complaints

    return v.shape[0]


def minimal(m: Annotated[Container, A, A.value > 2]) -> Annotated[int, B, B > 1]:
    return m.value


# def test(m: Annotated[int, V]) -> None:
#     return None
#
#
# def matmul(m1: Annotated[array, V, V.shape == (A, B)],
#         m2: Annotated[array, V, V.shape == (B, C)]
#         ) -> Annotated[array, V, V.shape == (A, C)]:
#     return array((m1.shape[0], m2.shape[1]), m1.dtype, m1.device)
#
#
# def reshape(m: Annotated[array, A],
#         shape: Annotated[tuple[int, ...], B]
#         ) -> Annotated[None, V, A.shape == B]:
#     return None
#
#
# def other(m: Annotated[int, V, V == 4]):
#     return None


# def upsample_m(signal: torch.Tensor, y: Annotated[int, X]) -> Annotated[torch.Tensor, Y]:
#     ''' signal is a batch of 1D tensors; M is an integer '''
#     B = signal.shape[0]
#     L = signal.shape[1]
#
#     up = torch.zeros((B, int(y*L)), dtype=signal.dtype, device=signal.device)
#     up[:, ::y] = signal
#
#     return up


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
