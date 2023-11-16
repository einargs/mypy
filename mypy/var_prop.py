"""Simple utility for dealing with lists of strings and integers that
correspond to property accesses.
"""

from enum import Enum
import enum
from typing import Union
from typing_extensions import TypeAlias


class MetaProp(Enum):
    """Special variable properties.

    Currently only includes len, which refers to the len function wrapping the
    previous properties.
    """
    len = enum.auto()

class DupCall:
    def __init__(self, size: int):
        self.size = size

VarProp: TypeAlias = Union[str, int, MetaProp, DupCall]

def prop_list_str(base: str, props: list[VarProp]) -> str:
    out = base
    for prop in props:
        if isinstance(prop, int):
            out += f"[{prop}]"
        elif isinstance(prop, str):
            out += f".{prop}"
        elif isinstance(prop, MetaProp):
            if prop == MetaProp.len:
                out = f"len({out})"
            else:
                assert False, "impossible"
        elif isinstance(prop, DupCall):
            out = f"dup({out}, {prop.size})"
        else:
            assert False, "impossible"
    return out