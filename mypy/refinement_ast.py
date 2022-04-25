"""
A restricted syntax tree specifically dealing with relevant information.
"""
from typing import (
    Dict, Any, Union, Optional, Tuple, overload, TypeVar, Callable, Type, Final,
)
from typing_extensions import TypeAlias
from abc import abstractmethod
import enum

from mypy.nodes import Context, TypeInfo


T = TypeVar('T')
JsonDict: TypeAlias = Dict[str, Any]


def make_deserialize_map(ty: Type[T]) -> Dict[str, Callable[[JsonDict], T]]:
    deserialize_map = {
        key: obj.deserialize
        for key, obj in globals().items()
        if isinstance(obj, type) and issubclass(obj, ty) and obj is not ty
    }

    return deserialize_map


class RType(Context):
    @abstractmethod
    def serialize(self) -> Union[JsonDict, str]:
        raise NotImplementedError('Cannot serialize {} instance'.format(self.__class__.__name__))

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RType':
        classname = data['.class']
        deserialize = rtype_deserialize_map.get(classname)
        if deserialize is None:
            raise NotImplementedError('Cannot deserialize {} instance'.format(cls.__name__))

        return deserialize(data)

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("Cannot get representation of {}"
                .format(self.__class__.__name__))


class RTupleType(RType):
    # Any size ambiguities should be resolved during translation.
    def __init__(
            self,
            size: Optional[int],
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.size = size

    def serialize(self) -> JsonDict:
        return {'.class': 'RTupleType',
                'size': self.size
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RTupleType':
        assert data['.class'] == 'RTupleType'
        return RTupleType(data['size'])

    def __repr__(self) -> str:
        return f"tuple({self.size})"


class RIntType(RType):
    def serialize(self) -> JsonDict:
        return {'.class': 'RIntType'}

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RIntType':
        assert data['.class'] == 'RIntType'
        return RIntType()

    def __repr__(self) -> str:
        return "int"


class RBoolType(RType):
    def serialize(self) -> JsonDict:
        return {'.class': 'RBoolType'}

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RBoolType':
        assert data['.class'] == 'RBoolType'
        return RBoolType()

    def __repr__(self) -> str:
        return "bool"


class RNoneType(RType):
    def serialize(self) -> JsonDict:
        return {'.class': 'RNoneType'}

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RNoneType':
        assert data['.class'] == 'RNoneType'
        return RNoneType()

    def __repr__(self) -> str:
        return "None#"


class RClassType(RType):
    def __init__(
            self,
            fullname: str,
            fields: Dict[str, RType],
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        # The full, unique name
        self.fullname: Final = fullname
        self.fields: Final = fields

    def serialize(self) -> JsonDict:
        return {'.class': 'RClassType',
                'fields': {k: v.serialize() for k, v in self.fields.items()},
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RClassType':
        assert data['.class'] == 'RClassType'
        fields = {
            key: RType.deserialize(value)
            for key, value in data['fields'].items()
        }
        return RIntType(data['fullname'], fields)

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}: {ty}" for name, ty in self.fields.items())
        return f"{self.fullname}({fields})"


class RClassHoleType(RType):
    # TODO: i'm positive that RSelf was working.
    def __init__(
            self,
            type: TypeInfo,
            line: int = -1,
            column: int = -1,
            ) -> None:
        super().__init__(line, column)
        self.type = type

    def serialize(self) -> JsonDict:
        return {'.class': 'RClassHoleType',
                'type': self.type.serialize()
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RClassHoleType':
        assert data['.class'] == 'RClassHoleType'
        return RClassHoleType(TypeInfo.deserialize(data['type']))

    def __repr__(self) -> str:
        return f"ClassHole({self.type.fullname})"


class RDupUnionType(RType):
    """Similar to RClassHoleType, this is only used in translation.
    """
    def __init__(
            self,
            size: int,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.size = size

    def serialize(self) -> JsonDict:
        return {'.class': 'RUnionType',
                'size': self.size,
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RUnionType':
        assert data['.class'] == "RUnionType"
        return RUnionType(data['size'])

    def __repr__(self) -> str:
        return f"DupUnion({self.size})"


rtype_deserialize_map: Final = make_deserialize_map(RType)


class RAST(Context):
    pass


Key: TypeAlias = Tuple[Any, ...]


@overload
def rexpr_key(expr: 'RExpr') -> Key: ...


def rexpr_key(expr: Optional['RExpr']) -> Optional[Key]:
    if expr is None:
        return None

    if isinstance(expr, RName):
        return ('RName', expr.name)
    elif isinstance(expr, RFree):
        return ('RFree', expr.id)
    elif isinstance(expr, RMember):
        return ('RMember', expr.attr, rexpr_key(expr.base))
    elif isinstance(expr, RIndex):
        return ('RIndex', expr.index, rexpr_key(expr.base))
    elif isinstance(expr, RArith):
        return ('RArith', rexpr_key(expr.lhs), expr.op, rexpr_key(expr.rhs))
    elif isinstance(expr, RLogic):
        return ('RLogic', expr.op, list(map(rexpr_key, expr.args)))
    elif isinstance(expr, RCmp):
        return ('RCmp', rexpr_key(expr.lhs), expr.op, rexpr_key(expr.rhs))
    elif isinstance(expr, RIntLiteral):
        return ('RIntLiteral', expr.value)
    elif isinstance(expr, RLenExpr):
        return ('RLenExpr', rexpr_key(expr.expr))
    elif isinstance(expr, RDupExpr):
        return ('RDupExpr', rexpr_key(expr.expr), expr.size)
    elif isinstance(expr, RFoldExpr):
        return ('RFoldExpr',
                expr.acc_var,
                expr_cur_var,
                rexpr_key(expr.fold_expr),
                rexpr_key(expr.folded_var),
                rexpr_key(expr.start),
                rexpr_key(expr.end))
    elif isinstance(expr, RTupleExpr):
        return ('RTupleExpr', list(map(rexpr_key, expr.members)))
    elif isinstance(expr, RNoneExpr):
        return ('RNoneExpr',)
    else:
        assert False


class RExpr(RAST):
    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError("__eq__ not implemented for {}".format(self.__class__.__name__))

    def __hash__(self) -> int:
        # Has to be abstract bc __eq__ and __hash__ have to come from the same
        # class.
        raise NotImplementedError("__hash__ not implemented for {}".format(self.__class__.__name__))

    @abstractmethod
    def serialize(self) -> Union[JsonDict, str]:
        raise NotImplementedError('Cannot serialize {} instance'.format(self.__class__.__name__))

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RExpr':
        classname = data['.class']
        deserialize = rexpr_deserialize_map.get(classname)
        if deserialize is None:
            raise NotImplementedError('Cannot deserialize {} instance'.format(cls.__name__))

        return deserialize(data)

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("Cannot get representation of {}"
                .format(self.__class__.__name__))


# An RExpr only containing RName, RFree, RMember, RIndex throughout the entire
# structure.
RLoc: TypeAlias = RExpr


class RName(RExpr):
    def __init__(
            self,
            name: str,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.name = name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RName):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RName',
                'name': self.name,
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RName':
        assert data['.class'] == 'RName'
        return RName(data['name'])

    def __repr__(self) -> str:
        assert isinstance(self.name, str), f"name was {type(self.name)}"
        return self.name


class RFree(RExpr):
    """A free variable generated by the translation to the intermediary
    refinement representation.
    """
    def __init__(
            self,
            name: str,
            id: int,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.name = name
        self.id = id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RFree):
            return NotImplemented
        return self.name == other.name and self.id == other.id

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RFree',
                'name': self.name,
                'id': self.id,
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RFree':
        assert data['.class'] == 'RFree'
        return RFree(data['name'], data['id'])

    def __repr__(self) -> str:
        return f"{self.name}#{self.id}"


RVar: TypeAlias = Union[RFree, RName]


class RMember(RExpr):
    def __init__(
            self,
            base: RExpr,
            attr: str,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.base = base
        self.attr = attr

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RMember):
            return NotImplemented
        return self.base == other.base and self.attr == other.attr

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RMember',
                'base': self.base.serialize(),
                'attr': self.attr,
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RMember':
        assert data['.class'] == 'RMember'
        return RMember(RExpr.deserialize(data['base']), data['attr'])

    def __repr__(self) -> str:
        return f"{self.base}.{self.attr}"


class RIndex(RExpr):
    def __init__(
            self,
            base: RExpr,
            index: int,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.base = base
        self.index = index

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RIndex):
            return NotImplemented
        return self.base == other.base and self.index == other.index

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RIndex',
                'base': self.base.serialize(),
                'index': self.index,
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RIndex':
        assert data['.class'] == 'RIndex'
        return RIndex(RExpr.deserialize(data['base']), data['index'])

    def __repr__(self) -> str:
        return f"{self.base}[{self.index}]"


class RArithOp(enum.Enum):
    plus = "+"
    minus = "-"
    mult = "*"
    div = "/"


class RArith(RExpr):
    def __init__(
            self,
            lhs: RExpr,
            op: RArithOp,
            rhs: RExpr,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RArith):
            return NotImplemented
        return (self.lhs == other.lhs
                and self.op == other.op
                and self.rhs == other.rhs)

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RArith',
                'lhs': self.lhs.serialize(),
                'op': op.value,
                'rhs': self.rhs.serialize(),
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RArith':
        assert data['.class'] == 'RArith'
        return RArith(
                RExpr.deserialize(data['lhs']),
                RArithOp(data['op']),
                RExpr.deserialize(data['rhs']))

    def __repr__(self) -> str:
        return f"({self.lhs} {self.op.value} {self.rhs})"


class RLogicOp(enum.Enum):
    and_op = "and"
    or_op = "or"


class RLogic(RExpr):
    def __init__(
            self,
            op: RLogicOp,
            args: list[RExpr],
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.op = op
        self.args = args

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RLogic):
            return NotImplemented
        return self.op == other.op and self.args == other.args

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RLogic',
                'lhs': self.lhs.serialize(),
                'op': op.value,
                'rhs': self.rhs.serialize(),
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RLogic':
        assert data['.class'] == 'RLogic'
        return RLogic(
                RExpr.deserialize(data['lhs']),
                RLogicOp(data['op']),
                RExpr.deserialize(data['rhs']))

    def __repr__(self) -> str:
        arg_str = ", ".join(map(str, self.args))
        return f"{self.op.value}({arg_str})"


class RCmpOp(enum.Enum):
    eq = "=="
    not_eq = "!="
    lt = "<"
    lt_eq = "<="
    gt = ">"
    gt_eq = ">="


class RCmp(RExpr):
    def __init__(
            self,
            lhs: RExpr,
            op: RCmpOp,
            rhs: RExpr,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RCmp):
            return NotImplemented
        return (self.lhs == other.lhs
                and self.op == other.op
                and self.rhs == other.rhs)

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RCmp',
                'lhs': self.lhs.serialize(),
                'op': op.value,
                'rhs': self.rhs.serialize(),
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RCmp':
        assert data['.class'] == 'RCmp'
        return RCmp(
                RExpr.deserialize(data['lhs']),
                RCmpOp(data['op']),
                RExpr.deserialize(data['rhs']))

    def __repr__(self) -> str:
        return f"{self.lhs} {self.op.value} {self.rhs}"


class RIntLiteral(RExpr):
    def __init__(
            self,
            value: int,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RIntLiteral):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RIntLiteral',
                'value': self.value,
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RIntLiteral':
        assert data['.class'] == 'RIntLiteral'
        return RIntLiteral(data['value'])

    def __repr__(self) -> str:
        return str(self.value)


class RLenExpr(RExpr):
    def __init__(
            self,
            expr: RExpr,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.expr = expr

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RLenExpr):
            return NotImplemented
        return self.expr == other.expr

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RLenExpr',
                'expr': self.expr.serialize(),
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RLenExpr':
        assert data['.class'] == 'RLenExpr'
        return RLenExpr(RExpr.deserialize(data['expr']))

    def __repr__(self) -> str:
        return f"len({self.expr})"


class RDupExpr(RExpr):
    def __init__(
            self,
            expr: RExpr,
            size: int,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.expr = expr
        self.size = size

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RDupExpr):
            return NotImplemented
        return self.expr == other.expr and self.size == other.size

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RDupExpr',
                'expr': self.expr.serialize(),
                'size': self.size,
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RDupExpr':
        assert data['.class'] == 'RDupExpr'
        return RDupExpr(RExpr.deserialize(data['expr']), data['size'])

    def __repr__(self) -> str:
        return f"dup({self.expr}, {self.size})"


class RFoldExpr(RExpr):
    def __init__(
            self,
            acc_var: str,
            cur_var: str,
            fold_expr: RExpr,
            folded_var: RLoc,
            start: Optional[RExpr],
            end: Optional[RExpr],
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.acc_var = acc_var
        self.cur_var = cur_var
        self.fold_expr = fold_expr
        self.folded_var = folded_var
        self.start = start
        self.end = end

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RFoldExpr):
            return NotImplemented
        return (self.acc_var == other.acc_var
                and self.cur_var == other.cur_var
                and self.fold_expr == other.fold_expr
                and self.folded_var == other.folded_var
                and self.start == other.start
                and self.end == other.end)

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RFoldExpr',
                'acc_var': self.acc_var,
                'cur_var': self.cur_var,
                'fold_expr': self.fold_expr.serialize(),
                'folded_var': self.folded_var.serialize(),
                'start': None if self.start is None else self.start.serialize(),
                'end': None if self.end is None else self.end.serialize(),
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RFoldExpr':
        assert data['.class'] == 'RFoldExpr'
        return RFoldExpr(
                data['acc_var'],
                data['cur_var'],
                RExpr.deserialize(data['fold_expr']),
                RExpr.deserialize(data['folded_var']),
                None if data['start'] is None else RExpr.deserialize(data['start']),
                None if data['end'] is None else RExpr.deserialize(data['end']))

    def __repr__(self) -> str:
        start = "" if self.start is None else str(self.start)
        end = "" if self.end is None else str(self.end)
        return (f"fold(lambda {self.acc_var}, {self.cur_var}: {self.fold_expr}, "
            f"{self.folded_var}[{start}:{end}])")


class RTupleExpr(RExpr):
    # TODO add to other enumerations
    def __init__(
            self,
            members: list[RExpr],
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.members = members

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RTupleExpr):
            return NotImplemented
        return self.members == other.members

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RTupleExpr',
                'members': [m.serialize() for m in self.members],
                }

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RTupleExpr':
        assert data['.class'] == 'RTupleExpr'
        return RTupleExpr(list(map(RExpr.deserialize, data['expr'])))

    def __repr__(self) -> str:
        members = ", ".join(map(str, self.members))
        return "({})".format(members)


class RNoneExpr(RExpr):
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RNoneExpr):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        key = rexpr_key(self)
        return hash(key)

    def serialize(self) -> JsonDict:
        return {'.class': 'RNoneExpr'}

    @classmethod
    def deserialize(cls, data: JsonDict) -> 'RNoneExpr':
        assert data['.class'] == 'RNoneExpr'
        return RNoneExpr()

    def __repr__(self) -> str:
        return "None#"


rexpr_deserialize_map: Final = make_deserialize_map(RExpr)


class RCond(Context):
    """A wrapper to indicate that an expression is a refinement condition.
    Further, the context informmation should specifically point back to where
    the error should be triggered if the condition fails.
    """
    def __init__(
            self,
            expr: RExpr,
            line: int,
            column: int
            ) -> None:
        super().__init__(line, column)
        self.expr = expr

    def __repr__(self) -> str:
        return str(self.expr)


class RStmt(RAST):
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("Cannot get representation of {}"
                .format(self.__class__.__name__))


class RHavoc(RStmt):
    """Indicates that the given variable can have any value. In other
    words, it invalidates any existing constraints.
    """
    def __init__(
            self,
            var: RExpr,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.var = var

    def __repr__(self) -> str:
        return f"havoc {self.var}"


class RDecl(RStmt):
    """Once something is declared it needs to be assigned to or have a Havoc
    statement used on it.
    """
    def __init__(
            self,
            var: RExpr,
            type: RType,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.var = var
        self.type = type

    def __repr__(self) -> str:
        return f"{self.var}: {self.type}"


class RExprAssign(RStmt):
    def __init__(
            self,
            var: RExpr,
            ty: RType,
            expr: RExpr,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.var = var
        self.ty = ty
        self.expr = expr

    def __repr__(self) -> str:
        return f"{self.var}: {self.ty} = {self.expr}"


class RExit(RStmt):
    """Rather than having a return statement, this representation instead has
    exit statements that serve to indicate termination of control flow.

    All post conditions are instead checked by separately generated assertions.

    Any return values are translated into separate variables that are then used
    in the post conditon.
    """
    # TODO: am I sure I don't want to just have a temp variable and expression
    # indicated in this? Or note down an optional variable to substitute the
    # expression with? (And then as an optimization I can instead substitute
    # for a variable equal to the return value?)
    def __init__(
            self,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)

    def __repr__(self) -> str:
        return "exit"


class RAssert(RStmt):
    def __init__(
            self,
            cond: RCond,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.cond = cond

    def __repr__(self) -> str:
        return f"assert {self.cond}"


class RAssume(RStmt):
    def __init__(
            self,
            expr: RExpr,
            line: int = -1,
            column: int = -1
            ) -> None:
        super().__init__(line, column)
        self.expr = expr

    def __repr__(self) -> str:
        return f"assume {self.expr}"


def rexpr_substitute(
        term: RExpr,
        substitutions: Dict[RVar, RExpr]
        ) -> RExpr:
    """Probably not the final form, but will do for now.

    The substitution pairs are (from, to) ordering.
    """
    if len(substitutions) == 0:
        return term

    sub_map = substitutions
    assert isinstance(sub_map, dict)

    def sub(term: RExpr) -> RExpr:
        assert isinstance(term, RExpr)
        if isinstance(term, RName):
            return sub_map.get(term, term)
        elif isinstance(term, RFree):
            return sub_map.get(term, term)
        elif isinstance(term, RMember):
            base = sub(term.base)
            return RMember(base, term.attr).set_line(term)
        elif isinstance(term, RIndex):
            base = sub(term.base)
            return RIndex(base, term.index).set_line(term)
        elif isinstance(term, RArith):
            lhs = sub(term.lhs)
            rhs = sub(term.rhs)
            return RArith(lhs, term.op, rhs).set_line(term)
        elif isinstance(term, RLogic):
            return RLogic(term.op, list(map(sub, term.args))).set_line(term)
        elif isinstance(term, RCmp):
            lhs = sub(term.lhs)
            rhs = sub(term.rhs)
            return RCmp(lhs, term.op, rhs).set_line(term)
        elif isinstance(term, RIntLiteral):
            return term
        elif isinstance(term, RLenExpr):
            return RLenExpr(sub(term.expr)).set_line(term)
        elif isinstance(term, RDupExpr):
            return RDupExpr(sub(term.expr), term.size).set_line(term)
        elif isinstance(term, RFoldExpr):
            return RFoldExpr(
                    term.acc_var,
                    term.cur_var,
                    # Currently I'm not going down into this bc I don't
                    # want to deal with shadowing or removing variables
                    # from the substitutions, etc.
                    term.fold_expr,
                    sub(term.folded_var),
                    None if term.start is None else sub(term.start),
                    None if term.end is None else sub(term.end)).set_line(term)
        elif isinstance(term, RTupleExpr):
            return RTupleExpr(list(map(sub, term.members))).set_line(term)
        elif isinstance(term, RNoneExpr):
            return term
        else:
            assert False

    val = sub(term)
    assert val is not None
    return val

def rexpr_uses_self(term: RExpr) -> bool:
    if isinstance(term, RName):
        return term.name == "self"
    elif isinstance(term, (RMember, RIndex)):
        return rexpr_uses_self(term.base)
    elif isinstance(term, RArith):
        return rexpr_uses_self(term.lhs) or rexpr_uses_self(term.rhs)
    elif isinstance(term, RLogic):
        return any(map(rexpr_uses_self, term.args))
    elif isinstance(term, RCmp):
        return rexpr_uses_self(term.lhs) or rexpr_uses_self(term.rhs)
    elif isinstance(term, RIntLiteral):
        return False
    elif isinstance(term, (RLenExpr, RDupExpr)):
        return rexpr_uses_self(term.expr)
    elif isinstance(term, RFoldExpr):
        return (rexpr_uses_self(term.folded_var)
            or (False if term.start is None else rexpr_uses_self(term.start))
            or (False if term.end is None else rexpr_uses_self(term.end)))
    elif isinstance(term, RTupleExpr):
        return any(map(rexpr_uses_self, term.members))
    elif isinstance(term, (RFree, RIntLiteral, RNoneExpr)):
        return False
    else:
        assert False
