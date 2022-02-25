from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import (
    Dict, List, Set, Iterator, Union, Optional, Tuple, cast, Any,
)
from typing_extensions import TypeAlias

from mypy.types import (
    Type, AnyType, PartialType, UnionType, TypeOfAny, NoneType, get_proper_type,
    BaseType, RefinementConstraint, RefinementExpr, RefinementConstraintKind,
    RefinementLiteral, RefinementVar, RefinementTuple, RefinementValue,
)
from mypy.nodes import (
    Expression, Var, RefExpr, IndexExpr, MemberExpr, AssignmentExpr, NameExpr,
    Context, IntExpr,
)
from mypy.literals import Key, literal, literal_hash, subkeys
from mypy.messages import MessageBuilder
import z3


VarProp: TypeAlias = Union[str, int]


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
    def extend(self, props: list[VarProp]) -> 'VerificationVar':
        """Extends the variable with a list of properties.
        """
        pass


def prop_list_str(base: str, props: list[VarProp]) -> str:
    return "".join([base] + [f"[{v}]" if isinstance(v, int) else f".{v}" for v in props])


class RealVar(VerificationVar):
    """A real var is present in the actual source, either as a term variable or
    a refinement variable.
    """
    def __init__(
            self,
            name: str,
            props: list[VarProp] = []) -> None:
        self.name = name
        self.props = props

    def subvars(self) -> 'list[VerificationVar]':
        vars: list[VerificationVar] = [RealVar(self.name)]
        props = []
        for p in self.props[:-1]:
            props.append(p)
            vars.append(RealVar(self.name, props.copy()))
        return vars

    def extend(self, props: list[VarProp]) -> 'VerificationVar':
        return RealVar(self.name, self.props + props)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RealVar):
            return NotImplemented
        return self.name == other.name and self.props == other.props

    def __hash__(self) -> int:
        fullname = prop_list_str(self.name, self.props)
        return hash(("real_var", fullname))

    def __repr__(self) -> str:
        fullname = prop_list_str(self.name, self.props)
        return "var({})".format(fullname)


class FreshVar(VerificationVar):
    """A type of verification var that can be introduced as fresh in a context.
    """
    def __init__(self, id: int, props: list[VarProp] = []):
        self.id = id
        self.props = props

    def subvars(self) -> 'list[VerificationVar]':
        vars: list[VerificationVar] = [FreshVar(self.id)]
        props = []
        for p in self.props[:-1]:
            props.append(p)
            vars.append(FreshVar(self.id, props.copy()))
        return vars

    def extend(self, props: list[VarProp]) -> 'VerificationVar':
        return FreshVar(self.id, self.props + props)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FreshVar):
            return NotImplemented
        return self.id == other.id and self.props == other.props

    def __hash__(self) -> int:
        fullname = prop_list_str(str(self.id), self.props)
        return hash(("fresh_var", fullname))

    def __repr__(self) -> str:
        fullname = prop_list_str(str(self.id), self.props)
        return f"var_id({fullname})"


def to_real_var(e: Expression, props=[]) -> Optional[RealVar]:
    if isinstance(e, NameExpr):
        props.reverse()
        return RealVar(e.name, props=props)
    elif isinstance(e, MemberExpr):
        props.append(e.name)
        return to_real_var(e.expr, props)
    else:
        return None


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

    def translate_expr(self, expr: RefinementExpr,
            *, ext_props: list[VarProp] = []) -> SMTExpr:
        if isinstance(expr, RefinementLiteral):
            return expr.value
        elif isinstance(expr, RefinementVar):
            # Resolve anything where we have a term variable m, but we use the
            # refinement variable R to refer to it in constraints.
            default_var = RealVar(expr.name)
            base = self.bound_var_to_name.get(expr.name, default_var)
            var = base.extend(cast(list[VarProp], expr.props) + ext_props)
            for sv in var.subvars():
                self.dependencies.setdefault(sv, set()).add(var)

            return self.get_smt_var(var)
        else:
            assert False, "Should not be passed"

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

    def translate_to_constraints(self,
            con: RefinementConstraint,
            *, ctx: Context) -> list[SMTConstraint]:
        """Translate a refinement constraint to smt solver constraints,
        splitting it into multiple constraints if it contains a refinement
        tuple.
        """
        left_tuple = isinstance(con.left, RefinementTuple)
        right_tuple = isinstance(con.right, RefinementTuple)
        if left_tuple and right_tuple:
            self.fail("Should not compare two tuple expressions", ctx)
            return []
        elif left_tuple or right_tuple:
            tuple_expr: RefinementTuple = cast(RefinementTuple,
                    con.left if left_tuple else con.right)
            var_expr = con.right if left_tuple else con.left

            if not isinstance(var_expr, RefinementVar):
                self.fail("Can only compare a tuple expression to a "
                    "refinement variable", ctx)
                return []

            if con.kind != RefinementConstraint.EQ:
                self.fail("Can only use == with tuple expressions", ctx)
                return []

            results: list[SMTConstraint] = []
            for i, v in enumerate(tuple_expr.items):
                tuple_item = self.translate_expr(v)
                indexed_var = self.translate_expr(var_expr, ext_props=[i])
                # We don't technically need to figure out left and right,
                # but it's helpful.
                left = tuple_item if left_tuple else indexed_var
                right = indexed_var if left_tuple else tuple_item
                results.append(left == right)
            return results
        else:
            left = self.translate_expr(con.left)
            right = self.translate_expr(con.right)
            return [self.translate_single_constraint(
                left, con.kind, right, ctx=ctx)]

    def translate_single_constraint(self,
            left: SMTExpr,
            kind: RefinementConstraintKind,
            right: SMTExpr,
            *, ctx: Context) -> SMTConstraint:
        if not isinstance(left, z3.ArithRef) and not isinstance(right, z3.ArithRef):
            self.fail("A refinement constraint must include at least one "
                    "refinement variable", ctx)

        if kind == RefinementConstraint.EQ:
            return left == right
        elif kind == RefinementConstraint.NOT_EQ:
            return left != right
        elif kind == RefinementConstraint.LT:
            return left < right
        elif kind == RefinementConstraint.LT_EQ:
            return left <= right
        elif kind == RefinementConstraint.GT:
            return left > right
        elif kind == RefinementConstraint.GT_EQ:
            return left >= right

    def add_var(self, var_name: str, typ: Type) -> None:
        if not isinstance(typ, BaseType) or typ.refinements is None:
            return
        print("works")
        info = typ.refinements

        if info.var is not None:
            self.add_bound_var(var_name, info.var.name, typ)

        for c in info.constraints:
            cons = self.translate_to_constraints(c, ctx=typ)
            self.constraints += cons

    def fail(self, msg: str, context: Context) -> None:
        self.msg.fail(msg, context)

    def var_from_expr(self, expr: Expression) -> Optional[VerificationVar]:
        """Get a verification var for an expression. Will create verification
        vars and constraints for integers, etc.
        """
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
        print("constraints", self.constraints, constraints)
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

            constraints = [smt_c
                    for c in info.constraints
                    for smt_c in self.translate_to_constraints(c, ctx=ctx)]
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
                constraints = [smt_c
                        for c in info.constraints
                        for smt_c in self.translate_to_constraints(c, ctx=ctx)]
                print("var_versions", self.var_versions)
                print("constraints", self.constraints)
                if not self.check_implication(constraints):
                    self.fail(FAIL_MSG, target)
                return
