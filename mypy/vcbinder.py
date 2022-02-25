from contextlib import contextmanager
from typing import Dict, List, Set, Iterator, Union, Optional, Tuple, cast, Any
from typing_extensions import TypeAlias

from mypy.types import (
    Type, AnyType, PartialType, UnionType, TypeOfAny, NoneType, get_proper_type,
    BaseType, RefinementConstraint, RefinementExpr, RefinementConstraintKind,
    RefinementLiteral, RefinementVar, RefinementTuple,
)
from mypy.nodes import (
    Expression, Var, RefExpr, IndexExpr, MemberExpr, AssignmentExpr, NameExpr,
    Context,
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

    All of them are uniquely identified by their ids. The `meta` property
    indicates whether they are a meta variable or bound to specific term
    variable.
    """

    def __init__(
            self,
            name: str,
            props: list[str] = []) -> None:
        self.name = name
        self.props = props

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VerificationVar):
            return NotImplemented
        return self.name == other.name and self.props == other.props

    def __hash__(self) -> int:
        fullname = ".".join([self.name] + self.props)
        h = hash(fullname)
        print("verification var hash", h, "for", fullname)
        return h

    def subvars(self) -> 'list[VerificationVar]':
        vars = [VerificationVar(self.name)]
        props = []
        for p in self.props[:-1]:
            props.append(p)
            vars.append(VerificationVar(self.name, props.copy()))
        return vars

    def __repr__(self) -> str:
        fullname = ".".join([self.name] + self.props)
        return "var({})".format(fullname)


def to_verification_var(e: Expression, props=[]) -> Optional[VerificationVar]:
    if isinstance(e, NameExpr):
        props.reverse()
        return VerificationVar(e.name, props=props)
    elif isinstance(e, MemberExpr):
        props.append(e.name)
        return to_verification_var(e.expr, props)
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


class VerificationBinder:
    """Deals with generation verification conditions.
    """

    def __init__(self, msg: MessageBuilder):
        self.msg = msg

        self.next_id = 0

        # Maps ids to relevant constraints
        self.constraints: Dict[int, Set[VCConstraint]] = {}

        self.smt_context: z3.Context = z3.Context()
        self.smt_solver: z3.Solver = z3.Solver(ctx=self.smt_context)

        self.smt_vars: Dict[int, z3.Int] = {}

        # Maps variable names to the ids
        self.var_versions: Dict[VerificationVar, int] = {}

        # Maps variable names to the variable names containing them
        self.dependencies: Dict[VerificationVar, Set[VerificationVar]] = {}

        # Maps bound refinement variable names to term variables base names.
        self.bound_var_to_name: Dict[str, str] = {}

    def _get_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def invalidate_var(self, var: VerificationVar) -> None:
        for dep in self.dependencies[var]:
            self.invalidate_var(dep)
        self.var_versions[var] = self._get_id()

    def get_smt_var(self, id: int) -> z3.Int:
        return self.smt_vars.setdefault(id, z3.Int(f"int-{id}",
            ctx=self.smt_context))

    def meta_var(self, name: str) -> VerificationVar:
        return VerificationVar(name)

    def add_bound_var(self, term_var: str, ref_var: str, context: Context) -> None:
        if ref_var in self.bound_var_to_name:
            self.fail("Tried to bind already bound refinement variable", context)
            return

        self.bound_var_to_name[ref_var] = term_var
        return

    def translate_expr(self, expr: RefinementExpr) -> VCExpr:
        if isinstance(expr, RefinementLiteral):
            return VCLiteral(expr.value)
        elif isinstance(expr, RefinementTuple):
            print("Have not yet implemented tuple handling")
            assert False
        elif isinstance(expr, RefinementVar):
            base_name = self.bound_var_to_name.get(expr.name, expr.name)
            print("expr.name", expr.name, "base_name", base_name)
            var = VerificationVar(base_name, expr.props)
            for sv in var.subvars():
                self.dependencies.setdefault(sv, set()).add(var)

            if var in self.var_versions:
                id = self.var_versions[var]
            else:
                id = self._get_id()
                self.var_versions[var] = id

            return VCVar(id)
        else:
            assert False

    @contextmanager
    def var_binding(self, ref_var: str, term_var: str) -> Iterator[None]:
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

    def translate_constraint(self, con: RefinementConstraint) -> VCConstraint:
        kind = con.kind
        left = self.translate_expr(con.left)
        right = self.translate_expr(con.right)
        return VCConstraint(left, kind, right)

    def add_constraint_to_smt(self, con: VCConstraint) -> None:
        """This adds the constraint to the smt solver.

        This is separate from just adding the constraint, because this can
        be done just for checking a type.
        """
        def to_smt_expr(e: VCExpr) -> Union[int, z3.Int]:
            if isinstance(e, VCVar):
                return self.get_smt_var(e.id)
            elif isinstance(e, VCLiteral):
                return e.value
            else:
                raise RuntimeError("Tuples not yet supported")

        right = to_smt_expr(con.right)
        left = to_smt_expr(con.left)

        print("smt sides", right, left)

        if con.kind == RefinementConstraint.EQ:
            self.smt_solver.add(right == left)
        elif con.kind == RefinementConstraint.NOT_EQ:
            self.smt_solver.add(right != left)
        elif con.kind == RefinementConstraint.LT:
            self.smt_solver.add(right > left)
        elif con.kind == RefinementConstraint.LT_EQ:
            self.smt_solver.add(right >= left)
        elif con.kind == RefinementConstraint.GT:
            self.smt_solver.add(right < left)
        elif con.kind == RefinementConstraint.GT_EQ:
            self.smt_solver.add(right <= left)

    def add_constraint(self, con: VCConstraint) -> None:
        # Index the vc constraint
        # TODO: may not be needed if the smt solver works the way I think
        # it does
        def add_con(e: VCExpr) -> None:
            """Adds the constraint to the index of constraints related to
            this expression if it is a VCExpr.
            """
            # TODO: does not currently handle VCTuples.
            if not isinstance(e, VCVar):
                return
            self.constraints.setdefault(e.id, set()).add(con)
        add_con(con.right)
        add_con(con.left)

        # Here we actually add this to the smt solver
        self.add_constraint_to_smt(con)

    def add_var(self, var_name: str, typ: Type) -> None:
        if not isinstance(typ, BaseType) or typ.refinements is None:
            return
        print("works")
        info = typ.refinements

        if info.var is not None:
            self.add_bound_var(var_name, info.var.name, typ)

        for c in info.constraints:
            con = self.translate_constraint(c)
            if not isinstance(con.left, VCVar) and not isinstance(con.right, VCVar):
                self.fail("A refinement constraint must include at least one "
                        "refinement variable", typ)
            self.add_constraint(con)

    def fail(self, msg: str, context: Context) -> None:
        self.msg.fail(msg, context)

    def check_subsumption(self, expr: Optional[Expression], target: BaseType) -> bool:
        """This is the meat of the refinement type checking.

        Here we check whether an expression's type subsumes the type it needs
        to be. This happens when passing arguments to refinement functions and
        when returning from refinement functions.

        We check this by generating a vc constraint that says, for all x given
        the associated `VCConstraint`s, do the `VCConstraint`s of the new type
        hold?

        We return false if the _actual refinement check_ is wrong.
        """
        info = target.refinements
        if info is None:
            return True

        if expr is None:
            self.smt_solver.push()
            for con in info.constraints:
                self.add_constraint_to_smt(self.translate_constraint(con))
            result = self.smt_solver.check()
            self.smt_solver.pop()
            return result == z3.sat

        var = to_verification_var(expr)
        if var is None:
            return True

        # We generate the substitution parameters if they exist.
        implied_constraints: list[VCConstraint]
        if info.var is None:
            constraints = [self.translate_constraint(c) for c in info.constraints]
        else:
            with self.var_binding(info.var.name, var.name):
                print("binding temp var", self.bound_var_to_name)
                print("smt_vars", self.smt_vars)
                print("var_versions", self.var_versions)
                constraints = [self.translate_constraint(c)
                        for c in info.constraints]
                print("2 binding temp var", self.bound_var_to_name)
                print("2 smt_vars", self.smt_vars)
                print("2 var_versions", self.var_versions)

        # Here we set a breakpoint that we'll jump back to after we've checked
        # things.
        self.smt_solver.push()

        for c in constraints:
            self.add_constraint_to_smt(c)

        result = self.smt_solver.check()

        print("smt solver", self.smt_solver)
        print("smt result", result)

        self.smt_solver.pop()

        return result == z3.sat
