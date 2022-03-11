"""Simple utility for dealing with lists of strings and integers that
correspond to property accesses.
"""

from typing import Union
from typing_extensions import TypeAlias

VarProp: TypeAlias = Union[str, int]

def prop_list_str(base: str, props: list[VarProp]) -> str:
    return "".join([base] + [f"[{v}]" if isinstance(v, int) else f".{v}" for v in props])
