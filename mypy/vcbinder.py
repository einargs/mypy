from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import (
    Dict, List, Set, Iterator, Union, Optional, Tuple, cast, Any, Iterable,
)
from typing_extensions import TypeAlias, TypeGuard

from mypy.types import (
    Type, AnyType, PartialType, UnionType, TypeOfAny, NoneType, get_proper_type,
    BaseType, RefinementConstraint, RefinementExpr, ConstraintKind,
    RefinementLiteral, RefinementVar, RefinementTuple, RefinementValue,
    RefinementBinOpKind, RefinementBinOp, Instance, ProperType,
)
from mypy.nodes import (
    Expression, ComparisonExpr, OpExpr, MemberExpr, UnaryExpr, StarExpr, IndexExpr,
    NameExpr, LITERAL_TYPE, IntExpr, FloatExpr, ComplexExpr, StrExpr, BytesExpr,
    UnicodeExpr, ListExpr, TupleExpr, SetExpr, DictExpr, CallExpr, SliceExpr, CastExpr,
    ConditionalExpr, EllipsisExpr, YieldFromExpr, YieldExpr, RevealExpr, SuperExpr,
    TypeApplication, LambdaExpr, ListComprehension, SetComprehension, DictionaryComprehension,
    GeneratorExpr, BackquoteExpr, TypeVarExpr, TypeAliasExpr, NamedTupleExpr, EnumCallExpr,
    TypedDictExpr, NewTypeExpr, PromoteExpr, AwaitExpr, TempNode, AssignmentExpr, ParamSpecExpr,
    RefinementVarExpr, Context, Lvalue,
)
import mypy.checker
from mypy.visitor import ExpressionVisitor
from mypy.literals import Key, literal, literal_hash, subkeys
from mypy.messages import MessageBuilder
import z3


VarProp: TypeAlias = Union[str, int]


def is_refined_type(typ: Type) -> TypeGuard[BaseType]:
    """Checks if a type is a base type and has refinement info.
    """
    return isinstance(typ, BaseType) and typ.refinements is not None


class VerificationVar:
    """Indicates the state of refinement variables.
    """

    @abstractmethod
    def __eq__(self, other: Any) -> bool: pass

    @abstractmethod
    def __hash__(self) -> int: pass

    @property
    @abstractmethod
    def props(self) -> list[VarProp]: pass

    @abstractmethod
    def copy_base(self, props: Optional[list[VarProp]] = None) -> 'VerificationVar':
        """Create a new copy with the props as an empty list or the passed
        props list.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str: pass

    @abstractmethod
    def fullpath(self) -> str: pass

    @abstractmethod
    def extend(self, props: list[VarProp]) -> 'VerificationVar':
        """Extends the variable with a list of properties.
        """
        pass

    def subvars(self) -> 'list[VerificationVar]':
        if len(self.props) == 0:
            return []

        vars: list[VerificationVar] = [self.copy_base()]
        props = []
        for p in self.props[:-1]:
            props.append(p)
            vars.append(self.copy_base(props.copy()))
        return vars


def prop_list_str(base: str, props: list[VarProp]) -> str:
    return "".join([base] + [f"[{v}]" if isinstance(v, int) else f".{v}" for v in props])


class RealVar(VerificationVar):
    """A real var is present in the actual source, either as a term variable or
    a refinement variable.
    """
    def __init__(
            self,
            name: str,
            props: Optional[list[VarProp]] = None) -> None:
        if props is None:
            props = []
        self.name = name
        self._props = props

    @property
    def props(self) -> list[VarProp]:
        return self._props

    def copy_base(self, props: Optional[list[VarProp]] = None) -> 'RealVar':
        if props is None:
            props = []
        return RealVar(self.name, props)

    def extend(self, props: list[VarProp]) -> 'VerificationVar':
        return RealVar(self.name, self.props + props)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RealVar):
            return NotImplemented
        return self.name == other.name and self.props == other.props

    def fullpath(self) -> str:
        return prop_list_str(self.name, self.props)

    def __hash__(self) -> int:
        return hash(("real_var", self.fullpath()))

    def __repr__(self) -> str:
        return "var({})".format(self.fullpath())


class FreshVar(VerificationVar):
    """A type of verification var that can be introduced as fresh in a context.
    """
    def __init__(self, id: int, props: Optional[list[VarProp]] = None):
        if props is None:
            props = []
        self.id = id
        self._props = props

    @property
    def props(self) -> list[VarProp]:
        return self._props

    def copy_base(self, props: Optional[list[VarProp]] = None) -> 'FreshVar':
        if props is None:
            props = []
        return FreshVar(self.id, props)

    def extend(self, props: list[VarProp]) -> 'VerificationVar':
        return FreshVar(self.id, self.props + props)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FreshVar):
            return NotImplemented
        return self.id == other.id and self.props == other.props

    def fullpath(self) -> str:
        return prop_list_str(str(self.id), self.props)

    def __hash__(self) -> int:
        return hash(("fresh_var", self.fullpath()))

    def __repr__(self) -> str:
        return f"var_id({self.fullpath()})"


def to_real_var(e: Expression, props: Optional[list[VarProp]] = None) -> Optional[RealVar]:
    if props is None:
        props = []
    if isinstance(e, NameExpr):
        props.reverse()
        return RealVar(e.name, props=props)
    elif isinstance(e, MemberExpr):
        props.append(e.name)
        return to_real_var(e.expr, props)
    elif isinstance(e, IndexExpr) and isinstance(e.index, IntExpr):
        props.append(e.index.value)
        return to_real_var(e.base, props)
    else:
        return None


SMTVar: TypeAlias = z3.ArithRef
SMTExpr: TypeAlias = Union[SMTVar, int]
SMTConstraint: TypeAlias = z3.BoolRef


class VerificationBinder:
    """Deals with generation verification conditions.
    """

    def __init__(self, chk: 'mypy.checker.TypeChecker'):
        # The parent type checker
        self.chk = chk

        # The message builder
        self.msg = chk.msg

        self.next_id = 0

        # Smt stuff
        self.smt_context: z3.Context = z3.Context()
        self.smt_solver: z3.Solver = z3.Solver(ctx=self.smt_context)

        # This will contain all of the constraints on smt variables.
        self.constraints: list[SMTConstraint] = []

        # This contains a list of all of the smt variables.
        self.smt_variables: list[SMTVar] = []

        # Maps variable names to the smt variables
        self.var_versions: Dict[VerificationVar, SMTVar] = {}

        # Tracks what expressions have had constraints "loaded" from their
        # types.
        self.has_been_touched: Set[VerificationVar] = set()

        # Maps variable names to the variable names containing them.
        # Used to track what variables to invalidate if another variable is
        # invalidated.
        self.dependencies: Dict[VerificationVar, Set[VerificationVar]] = {}

        # Maps bound refinement variable names to term variables.
        self.bound_var_to_name: Dict[str, VerificationVar] = {}

    def add_constraints(self, constraints: list[SMTConstraint]) -> None:
        self.constraints += constraints
        self.smt_solver.add(constraints)

    def fresh_verification_var(self) -> FreshVar:
        self.next_id += 1
        return FreshVar(self.next_id)

    def fresh_smt_var(self, var: VerificationVar) -> SMTVar:
        self.next_id += 1
        name = var.fullpath()
        smt_var = z3.Int(f"{name}#{self.next_id}", ctx=self.smt_context)
        self.smt_variables.append(smt_var)
        return smt_var

    def get_smt_var(self, var: VerificationVar) -> SMTVar:
        if var in self.var_versions:
            return self.var_versions[var]
        else:
            fresh_var = self.fresh_smt_var(var)
            self.var_versions[var] = fresh_var
            self.has_been_touched.add(var)
            return fresh_var

    def invalidate_var(self, var: VerificationVar) -> None:
        """Invalidate a variable, forcing the creation of a new smt variable
        with no associated constraints the next time it is used.
        """
        if var in self.var_versions:
            print("Invalidating:", self.var_versions[var])
            del self.var_versions[var]
        else:
            print("Invalidating:", var)
        if var in self.dependencies:
            for dep in self.dependencies[var]:
                self.invalidate_var(dep)

    def invalidate_vars_in_expr(self, expr: Expression) -> None:
        """Invalidate all mentions of `RealVar`s in an expression.
        """
        invalidator = Invalidator(self)
        expr.accept(invalidator)

    def translate_expr(self, expr: RefinementExpr,
            *, ext_props: Optional[list[VarProp]] = None) -> SMTExpr:
        if ext_props is None:
            ext_props = []
        if isinstance(expr, RefinementLiteral):
            return expr.value
        elif isinstance(expr, RefinementBinOp):
            left = self.translate_expr(expr.left)
            right = self.translate_expr(expr.right)
            if expr.kind == RefinementBinOpKind.add:
                return left + right
            elif expr.kind == RefinementBinOpKind.sub:
                return left - right
            elif expr.kind == RefinementBinOpKind.mult:
                return left * right
            elif expr.kind == RefinementBinOpKind.floor_div:
                return left / right
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
    def var_binding(self, ref_var: Optional[str], term_var: VerificationVar) -> Iterator[None]:
        """Temporary binds a refinement variable to a given base term
        variable.
        """
        if ref_var is None:
            yield None
        elif ref_var in self.bound_var_to_name:
            old = self.bound_var_to_name[ref_var]
            self.bound_var_to_name[ref_var] = term_var
            yield None
            self.bound_var_to_name[ref_var] = old
        else:
            self.bound_var_to_name[ref_var] = term_var
            yield None
            del self.bound_var_to_name[ref_var]

    @contextmanager
    def var_bindings(self, bindings: list[tuple[str, VerificationVar]]) -> Iterator[None]:
        """Temporarily binds multiple refinement variables to base verification
        variables.
        """
        old_vars: list[Optional[VerificationVar]] = []
        for ref_var, term_var in bindings:
            old_vars.append(self.bound_var_to_name.get(ref_var))
            self.bound_var_to_name[ref_var] = term_var
        yield None
        for i, (ref_var, _) in enumerate(bindings):
            old = old_vars[i]
            if old is None:
                del self.bound_var_to_name[ref_var]
            else:
                self.bound_var_to_name[ref_var] = old

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

            if con.kind != ConstraintKind.EQ:
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
            kind: ConstraintKind,
            right: SMTExpr,
            *, ctx: Context) -> SMTConstraint:
        if not isinstance(left, z3.ArithRef) and not isinstance(right, z3.ArithRef):
            self.fail("A refinement constraint must include at least one "
                    "refinement variable", ctx)

        if kind == ConstraintKind.EQ:
            return left == right
        elif kind == ConstraintKind.NOT_EQ:
            return left != right
        elif kind == ConstraintKind.LT:
            return left < right
        elif kind == ConstraintKind.LT_EQ:
            return left <= right
        elif kind == ConstraintKind.GT:
            return left > right
        elif kind == ConstraintKind.GT_EQ:
            return left >= right

    def add_bound_var(self,
            term_var: VerificationVar,
            ref_var: str,
            context: Context) -> None:
        if ref_var in self.bound_var_to_name:
            var = self.bound_var_to_name[ref_var]
            if var in self.var_versions:
                self.fail('Tried to bind already bound refinement variable "{}"'.format(ref_var),
                        context)
                return

        self.bound_var_to_name[ref_var] = term_var
        return

    def add_var(self, var: VerificationVar, typ: Type, *, ctx: Context) -> None:
        if not isinstance(typ, BaseType) or typ.refinements is None:
            return
        info = typ.refinements

        if info.var is not None:
            self.add_bound_var(var, info.var.name, typ)

        for c in info.constraints:
            cons = self.translate_to_constraints(c, ctx=ctx)
            self.add_constraints(cons)

    def add_argument(self, arg_name: str, typ: Type, *, ctx: Context) -> None:
        var = RealVar(arg_name)
        self.add_var(var, typ, ctx=ctx)

    def add_lvalue(self, lvalue: Lvalue, typ: Type) -> None:
        var = to_real_var(lvalue)
        if var is None:
            return
        self.add_var(var, typ, ctx=lvalue)

    def fail(self, msg: str, context: Context) -> None:
        self.msg.fail(msg, context)

    def var_for_call_expr(
            self,
            ret_type: BaseType,
            bindings: list[Tuple[str, Expression, Type]],
            ctx: Context
    ) -> VerificationVar:
        assert ret_type.refinements is not None

        fresh_var = self.fresh_verification_var()
        var_bindings: list[tuple[str, VerificationVar]] = []
        for ref_var, expr, expr_type in bindings:
            expr_var, was_error = self.var_from_expr(expr, expr_type)
            if expr_var is not None:
                var_bindings.append((ref_var, expr_var))
        # TODO: I think I need to generalize the idea of a bound variable,
        # because currently this won't catch overlapping refinement variables.
        # OTOH maybe that's okay because it'll be caught when it's defined?
        if ret_type.refinements.var is not None:
            var_bindings.append((ret_type.refinements.var.name, fresh_var))
        with self.var_bindings(var_bindings):
            cons = []
            for c in ret_type.refinements.constraints:
                cons += self.translate_to_constraints(c, ctx=ctx)
            self.add_constraints(cons)
        return fresh_var

    def real_var_from_expr(
            self,
            expr: Expression,
            expr_type: Optional[Type]
    ) -> Optional[VerificationVar]:
        """Attempt to directly convert an expression to a real verification
        variable.
        """
        var = to_real_var(expr)

        if var is None:
            return None

        # TODO: how do I deal with other variables mentioned in this, e.g.,
        # other properties of an object? Thinking about it maybe uniquing
        # the refinement variables in semantic analysis would help with
        # that. That way refinement variables would know they're talking
        # about the other properties when those come in...

        # TODO check that other stuff doesn't somehow get added before type
        # constraints can be added.
        if is_refined_type(expr_type) and var not in self.has_been_touched:
            assert expr_type.refinements, "guarenteed by is_refined_type"
            if expr_type.refinements.var is None:
                ref_var = None
            else:
                ref_var = expr_type.refinements.var.name

            with self.var_binding(ref_var, var):
                for c in expr_type.refinements.constraints:
                    cons = self.translate_to_constraints(c, ctx=expr)
                    self.add_constraints(cons)

        return var

    def smt_expr_from(
            self,
            expr: Expression,
            expr_type: ProperType
    ) -> Optional[SMTExpr]:
        """See if an expression is either an integer literal or has a tracked
        smt variable.
        """
        if isinstance(expr, IntExpr):
            return expr.value
        elif (is_refined_type(expr_type)
                and expr_type.refinements
                and expr_type.refinements.verification_var):
            smt_var = self.get_smt_var(expr_type.refinements.verification_var)
            print("smt_var", smt_var)
            return smt_var
        else:
            var = self.real_var_from_expr(expr, expr_type)
            if var:
                return self.get_smt_var(var)

        return None

    def var_for_bin_op(
            self,
            expr: OpExpr,
            left_type: ProperType,
            right_type: ProperType
    ) -> Optional[VerificationVar]:
        left = self.smt_expr_from(expr.left, left_type)
        right = self.smt_expr_from(expr.right, right_type)
        if left is None or right is None:
            return None

        if expr.op == '+':
            smt_expr = left + right
        elif expr.op == '-':
            smt_expr = left - right
        elif expr.op == '*':
            smt_expr = left * right
        elif expr.op == '//':
            smt_expr = left / right
        else:
            return None

        fresh_var = self.fresh_verification_var()
        smt_var = self.get_smt_var(fresh_var)
        self.add_constraints([smt_var == smt_expr])

        return fresh_var

    def var_from_expr(
            self,
            expr: Expression,
            expr_type: Optional[Type]
    ) -> tuple[Optional[VerificationVar], bool]:
        """Convert an expression verification var with the appropriate
        constraints.

        This currently handles converting expressions to RealVars, integer
        literals, and call expressions.

        This returns the optional verification var and a boolean that indicates
        whether a specific error message has already been returned.
        """
        if isinstance(expr, IntExpr):
            fresh_var = self.fresh_verification_var()
            smt_var = self.get_smt_var(fresh_var)
            self.add_constraints([smt_var == expr.value])
            return fresh_var, False
        elif isinstance(expr, CallExpr):
            assert expr_type is not None, "I don't think this should happen?"

            if is_refined_type(expr_type) and expr_type.refinements is not None:
                return expr_type.refinements.verification_var, False
            else:
                return None, False
        elif isinstance(expr, TupleExpr):
            self.fail("Tuple expressions are not yet implemented", expr)
            return None, True
        elif (is_refined_type(expr_type)
                and expr_type.refinements
                and expr_type.refinements.verification_var):
            print("triggered")
            return expr_type.refinements.verification_var, False
        else:
            var = self.real_var_from_expr(expr, expr_type)
            return var, False

    def check_implication(self, constraints: list[SMTConstraint]) -> bool:
        """Checks if the given constraints are implied by already known
        constraints.

        Returns false if the implication is not satisfiable.
        """

        # If there are no constraints to be checked it is defacto true.
        if constraints == []:
            return True

        # print("Variables:", self.smt_variables)
        print("Given:", self.constraints)
        print("Goal:", constraints)

        # Basically, in order to prove that the constraints are "valid" --
        # evaluates to true for all possible variable values -- we put a not
        # around the condition and then check that conditions is unsatisfiable
        # -- that we have no way it can be *untrue*.
        cond = z3.Not(z3.And(constraints))
        result = self.smt_solver.check(cond)
        print("Result:", "valid" if result == z3.unsat else "invalid")
        print()
        return result == z3.unsat

    def check_subsumption(self,
            expr: Optional[Expression],
            expr_type: Optional[Type],
            target: Type,
            *, ctx: Context) -> None:
        """This is the meat of the refinement type checking.

        Here we check whether an expression's type subsumes the type it needs
        to be. This happens when passing arguments to refinement functions and
        when returning from refinement functions.

        We check this by generating a vc constraint that says, for all x given
        the associated `VCConstraint`s, do the `VCConstraint`s of the new type
        hold?
        """
        if not (isinstance(expr_type, BaseType)
                and isinstance(target, BaseType)):
            return

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
                self.fail(FAIL_MSG, ctx)
            return
        else:
            var, has_sent_error = self.var_from_expr(expr, expr_type)
            if var is None:
                if not has_sent_error:
                    self.fail('could not understand expression', ctx)
                return
            print("var binding", info.var.name, "to", var)
            with self.var_binding(info.var.name, var):
                constraints = [smt_c
                        for c in info.constraints
                        for smt_c in self.translate_to_constraints(c, ctx=ctx)]
                if not self.check_implication(constraints):
                    self.fail(FAIL_MSG, ctx)
                return


class Invalidator(ExpressionVisitor[None]):
    def __init__(self, parent: 'VerificationBinder'):
        self.vc_binder = parent

    def invalidate(self, expr: Expression) -> None:
        """Invalidate the associated `RealVar` (if it exists).
        """
        var = to_real_var(expr)
        if var is None:
            return
        self.vc_binder.invalidate_var(var)

    def visit_int_expr(self, e: IntExpr) -> None:
        return None

    def visit_str_expr(self, e: StrExpr) -> None:
        return None

    def visit_bytes_expr(self, e: BytesExpr) -> None:
        return None

    def visit_unicode_expr(self, e: UnicodeExpr) -> None:
        return None

    def visit_float_expr(self, e: FloatExpr) -> None:
        return None

    def visit_complex_expr(self, e: ComplexExpr) -> None:
        return None

    def visit_star_expr(self, e: StarExpr) -> None:
        e.expr.accept(self)

    def visit_name_expr(self, e: NameExpr) -> None:
        self.invalidate(e)

    def visit_member_expr(self, e: MemberExpr) -> None:
        self.invalidate(e)

    def visit_op_expr(self, e: OpExpr) -> None:
        e.left.accept(self)
        e.right.accept(self)

    def visit_comparison_expr(self, e: ComparisonExpr) -> None:
        for o in e.operands:
            o.accept(self)

    def visit_unary_expr(self, e: UnaryExpr) -> None:
        e.expr.accept(self)

    def seq(self, iterable: Iterable[Expression]) -> None:
        for i in iterable:
            i.accept(self)

    def opt(self, expr: Optional[Expression]) -> None:
        if expr is not None:
            expr.accept(self)

    def visit_list_expr(self, e: ListExpr) -> None:
        self.seq(e.items)

    def visit_dict_expr(self, e: DictExpr) -> None:
        for k, v in e.items:
            v.accept(self)

    def visit_tuple_expr(self, e: TupleExpr) -> None:
        self.seq(e.items)

    def visit_set_expr(self, e: SetExpr) -> None:
        self.seq(e.items)

    def visit_index_expr(self, e: IndexExpr) -> None:
        self.invalidate(e)

    def visit_assignment_expr(self, e: AssignmentExpr) -> None:
        e.target.accept(self)
        e.value.accept(self)

    def visit_call_expr(self, e: CallExpr) -> None:
        e.callee.accept(self)
        self.seq(e.args)

    def visit_slice_expr(self, e: SliceExpr) -> None:
        self.opt(e.begin_index)
        self.opt(e.end_index)
        self.opt(e.stride)

    def visit_cast_expr(self, e: CastExpr) -> None:
        e.expr.accept(self)

    def visit_conditional_expr(self, e: ConditionalExpr) -> None:
        e.cond.accept(self)
        e.if_expr.accept(self)
        e.else_expr.accept(self)

    def visit_ellipsis(self, e: EllipsisExpr) -> None:
        return None

    def visit_yield_from_expr(self, e: YieldFromExpr) -> None:
        e.expr.accept(self)

    def visit_yield_expr(self, e: YieldExpr) -> None:
        if e.expr is not None:
            e.expr.accept(self)

    def visit_reveal_expr(self, e: RevealExpr) -> None:
        return None

    def visit_super_expr(self, e: SuperExpr) -> None:
        e.call.accept(self)

    def visit_type_application(self, e: TypeApplication) -> None:
        return None

    def visit_lambda_expr(self, e: LambdaExpr) -> None:
        e.expr().accept(self)

    # TODO: right now we're just ignoring generator stuff, so
    # in the future I should implement that if needed.
    def visit_list_comprehension(self, e: ListComprehension) -> None:
        return None

    def visit_set_comprehension(self, e: SetComprehension) -> None:
        return None

    def visit_dictionary_comprehension(self, e: DictionaryComprehension) -> None:
        return None

    def visit_generator_expr(self, e: GeneratorExpr) -> None:
        return None

    def visit_backquote_expr(self, e: BackquoteExpr) -> None:
        e.expr.accept(self)
        return None

    def visit_refinement_var_expr(self, e: RefinementVarExpr) -> None:
        return None

    def visit_type_var_expr(self, e: TypeVarExpr) -> None:
        return None

    def visit_paramspec_expr(self, e: ParamSpecExpr) -> None:
        return None

    def visit_type_alias_expr(self, e: TypeAliasExpr) -> None:
        return None

    def visit_namedtuple_expr(self, e: NamedTupleExpr) -> None:
        return None

    def visit_enum_call_expr(self, e: EnumCallExpr) -> None:
        return None

    def visit_typeddict_expr(self, e: TypedDictExpr) -> None:
        return None

    def visit_newtype_expr(self, e: NewTypeExpr) -> None:
        return None

    def visit__promote_expr(self, e: PromoteExpr) -> None:
        return None

    def visit_await_expr(self, e: AwaitExpr) -> None:
        e.expr.accept(self)

    def visit_temp_node(self, e: TempNode) -> None:
        return None
