from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import (
    Dict, List, Set, Iterator, Union, Optional, Tuple, cast, Any,
)
from typing_extensions import TypeAlias

from mypy.types import (
    Type, AnyType, PartialType, UnionType, TypeOfAny, NoneType, get_proper_type,
    BaseType, RefinementConstraint, RefinementExpr, RefinementConstraintKind,
    RefinementLiteral, RefinementVar, RefinementTuple,
)
from mypy.nodes import (
    Expression, Var, RefExpr, IndexExpr, MemberExpr, AssignmentExpr, NameExpr,
    Context, IntExpr,
)
from mypy.literals import Key, literal, literal_hash, subkeys
from mypy.messages import MessageBuilder
import z3


def vc_access(e: Expression) -> bool:
    if isinstance(e, NameExpr):
        return True
    elif isinstance(e, MemberExpr):
        return vc_access(e.expr)
    else:
        return False


class VerificationVar:
    """Indicates the state of refinement variables.
    """

    @abstractmethod
    def __eq__(self, other: Any) -> bool: pass

    @abstractmethod
    def __hash__(self) -> int: pass

    @abstractmethod
    def subvars(self) -> 'list[VerificationVar]': pass

    @abstractmethod
    def __repr__(self) -> str: pass

    @abstractmethod
    def extend(self, props: list[str]) -> 'VerificationVar':
        """Extends the variable with a list of properties.
        """
        pass


class RealVar(VerificationVar):
    """A real var is present in the actual source, either as a term variable or
    a refinement variable.
    """
    def __init__(
            self,
            name: str,
            props: list[str] = []) -> None:
        self.name = name
        self.props = props

    def subvars(self) -> 'list[VerificationVar]':
        vars = [RealVar(self.name)]
        props = []
        for p in self.props[:-1]:
            props.append(p)
            vars.append(RealVar(self.name, props.copy()))
        return vars

    def extend(self, props: list[str]) -> 'VerificationVar':
        return RealVar(self.name, self.props + props)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RealVar):
            return NotImplemented
        return self.name == other.name and self.props == other.props

    def __hash__(self) -> int:
        fullname = ".".join([self.name] + self.props)
        return hash(("real_var", fullname))

    def __repr__(self) -> str:
        fullname = ".".join([self.name] + self.props)
        return "var({})".format(fullname)


class FreshVar(VerificationVar):
    """A type of verification var that can be introduced as fresh in a context.
    """
    def __init__(self, id: int, props: list[str] = []):
        self.id = id
        self.props = props

    def subvars(self) -> 'list[VerificationVar]':
        return []

    def extend(self, props: list[str]) -> 'VerificationVar':
        return FreshVar(self.id, self.props + props)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FreshVar):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(("fresh_var", self.id))

    def __repr__(self) -> str:
        return f"var(id={self.id})"


def to_real_var(e: Expression, props=[]) -> Optional[RealVar]:
    if isinstance(e, NameExpr):
        props.reverse()
        return RealVar(e.name, props=props)
    elif isinstance(e, MemberExpr):
        props.append(e.name)
        return to_real_var(e.expr, props)
    else:
        return None


class VCExpr:
    __slots__ = ()


class VCLiteral(VCExpr):
    __slots__ = ('value',)

    def __init__(self, value: int):
        self.value = value


class VCVar(VCExpr):
    __slots__ = ('id',)

    def __init__(self, id: int):
        self.id = id


class VCConstraint:
    __slots__ = ('left', 'kind', 'right')

    def __init__(self, left: VCExpr, kind: RefinementConstraintKind, right: VCExpr):
        self.left = left
        self.kind = kind
        self.right = right


SMTVar: TypeAlias = z3.ArithRef
SMTExpr: TypeAlias = Union[SMTVar, int]
SMTConstraint: TypeAlias = z3.BoolRef


class VerificationBinder:
    """Deals with generation verification conditions.
    """

    def __init__(self, msg: MessageBuilder):
        self.msg = msg

        self.next_id = 0

        # Smt stuff
        self.smt_context: z3.Context = z3.Context()
        self.smt_solver: z3.Solver = z3.Solver(ctx=self.smt_context)

        # This will contain all of the constraints on smt variables.
        self.constraints: list[SMTConstraint] = []

        # Maps variable names to the smt variables
        self.var_versions: Dict[VerificationVar, SMTVar] = {}

        # Maps variable names to the variable names containing them.
        # Used to track what variables to invalidate if another variable is
        # invalidated.
        self.dependencies: Dict[VerificationVar, Set[VerificationVar]] = {}

        # Maps bound refinement variable names to term variables.
        self.bound_var_to_name: Dict[str, VerificationVar] = {}

    def fresh_verification_var(self) -> VerificationVar:
        self.next_id += 1
        return FreshVar(self.next_id)

    def fresh_smt_var(self) -> SMTVar:
        self.next_id += 1
        return z3.Int(f"int-{self.next_id}", ctx=self.smt_context)

    def get_smt_var(self, var: VerificationVar) -> SMTVar:
        if var in self.var_versions:
            return self.var_versions[var]
        else:
            fresh_var = self.fresh_smt_var()
            self.var_versions[var] = fresh_var
            return fresh_var

    def invalidate_var(self, var: VerificationVar) -> None:
        for dep in self.dependencies[var]:
            self.invalidate_var(dep)
        fresh_var = self.fresh_smt_var()
        self.var_versions[var] = fresh_var

    def add_bound_var(self, term_var: str, ref_var: str, context: Context) -> None:
        if ref_var in self.bound_var_to_name:
            self.fail("Tried to bind already bound refinement variable", context)
            return

        self.bound_var_to_name[ref_var] = RealVar(term_var)
        return

    def translate_expr(self, expr: RefinementExpr) -> SMTExpr:
        if isinstance(expr, RefinementLiteral):
            return expr.value
        elif isinstance(expr, RefinementTuple):
            assert False, "Have not yet implemented tuple handling"
        elif isinstance(expr, RefinementVar):
            # Resolve anything where we have a term variable m, but we use the
            # refinement variable R to refer to it in constraints.
            default_var = RealVar(expr.name)
            base = self.bound_var_to_name.get(expr.name, default_var)
            var = base.extend(expr.props)
            for sv in var.subvars():
                self.dependencies.setdefault(sv, set()).add(var)

            return self.get_smt_var(var)
        else:
            assert False, "should be impossible"

    @contextmanager
    def var_binding(self, ref_var: str, term_var: VerificationVar) -> Iterator[None]:
        """Temporary binds a refinement variable to a given base term
        variable.
        """
        if ref_var in self.bound_var_to_name:
            old = self.bound_var_to_name[ref_var]
            self.bound_var_to_name[ref_var] = term_var
            yield None
            self.bound_var_to_name[ref_var] = old
        else:
            self.bound_var_to_name[ref_var] = term_var
            yield None
            del self.bound_var_to_name[ref_var]

    def translate_constraint(self,
            con: RefinementConstraint,
            *,
            ctx: Context) -> SMTConstraint:
        left = self.translate_expr(con.left)
        right = self.translate_expr(con.right)

        if not isinstance(left, z3.ArithRef) and not isinstance(right, z3.ArithRef):
            self.fail("A refinement constraint must include at least one "
                    "refinement variable", ctx)

        if con.kind == RefinementConstraint.EQ:
            return left == right
        elif con.kind == RefinementConstraint.NOT_EQ:
            return left != right
        elif con.kind == RefinementConstraint.LT:
            return left < right
        elif con.kind == RefinementConstraint.LT_EQ:
            return left <= right
        elif con.kind == RefinementConstraint.GT:
            return left > right
        elif con.kind == RefinementConstraint.GT_EQ:
            return left >= right

    def add_var(self, var_name: str, typ: Type) -> None:
        if not isinstance(typ, BaseType) or typ.refinements is None:
            return
        print("works")
        info = typ.refinements

        if info.var is not None:
            self.add_bound_var(var_name, info.var.name, typ)

        for c in info.constraints:
            con = self.translate_constraint(c, ctx=typ)
            self.constraints.append(con)

    def fail(self, msg: str, context: Context) -> None:
        self.msg.fail(msg, context)

    def var_from_expr(self, expr: Expression) -> Optional[VerificationVar]:
        if isinstance(expr, IntExpr):
            fresh_var = self.fresh_verification_var()
            smt_var = self.get_smt_var(fresh_var)
            self.constraints.append(smt_var == expr.value)
            return fresh_var
        else:
            return to_real_var(expr)

    def check_implication(self, constraints: list[SMTConstraint]) -> bool:
        """Checks if the given constraints are implied by already known
        constraints.

        Returns false if the implication is not satisfiable.
        """
        variables = list(self.var_versions.values())
        print("smt variables", variables)
        cond = z3.ForAll(variables,
                z3.Implies(z3.And(self.constraints), z3.And(constraints)))
        return self.smt_solver.check(cond) == z3.sat

    def check_subsumption(self,
            expr: Optional[Expression],
            target: BaseType,
            ctx: Context) -> None:
        """This is the meat of the refinement type checking.

        Here we check whether an expression's type subsumes the type it needs
        to be. This happens when passing arguments to refinement functions and
        when returning from refinement functions.

        We check this by generating a vc constraint that says, for all x given
        the associated `VCConstraint`s, do the `VCConstraint`s of the new type
        hold?
        """
        info = target.refinements
        if info is None:
            return

        FAIL_MSG = "refinement type check failed"

        if expr is None or info.var is None:
            # This only applies to return statement checking. This case should
            # be prevented by checks to ensure we aren't giving a None type
            # a refinement variable.
            # TODO: implement those checks.
            assert expr is not None or info.var is not None

            constraints = [self.translate_constraint(c, ctx=ctx)
                    for c in info.constraints]
            if not self.check_implication(constraints):
                self.fail(FAIL_MSG, target)
            return
        else:
            var = self.var_from_expr(expr)
            if var is None:
                self.fail("could not type check expression against "
                    "refinement type", target)
                return
            with self.var_binding(info.var.name, var):
                constraints = [self.translate_constraint(c, ctx=ctx)
                        for c in info.constraints]
                print("var_versions", self.var_versions)
                print("constraints", self.constraints)
                if not self.check_implication(constraints):
                    self.fail(FAIL_MSG, target)
                return
