from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import (
    Dict, List, Set, Iterator, Union, Optional, Tuple, cast, Any, Iterable,
)
from typing_extensions import TypeAlias, TypeGuard

from mypy.types import (
    Type, AnyType, PartialType, UnionType, TypeOfAny, NoneType, get_proper_type,
    BaseType, RefinementConstraint, RefinementExpr, ConstraintKind,
    RefinementLiteral, RefinementVar, RefinementTuple, RefinementBinOpKind,
    RefinementBinOp, Instance, ProperType, TupleType, TypeVarType,
    RefinementSelf,
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
from mypy.checkmember import analyze_member_access
from mypy.visitor import ExpressionVisitor
from mypy.literals import Key, literal, literal_hash, subkeys
from mypy.messages import MessageBuilder
from mypy.var_prop import VarProp, prop_list_str
import z3


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
    def copy(self) -> 'VerificationVar': pass

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
        vars.reverse()
        return vars


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

    def copy(self) -> 'RealVar':
        return RealVar(self.name, self.props.copy())

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
        return self.fullpath()


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

    def copy(self) -> 'FreshVar':
        return FreshVar(self.id, self.props.copy())

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
        return self.fullpath()


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

        # This contains a list of all of the smt variables.
        self.smt_variables: list[SMTVar] = []

        # Maps variable names to the smt variables.
        self.var_versions: Dict[VerificationVar, SMTExpr] = {}

        # Maps variable names to refinement types.
        self.var_types: Dict[VerificationVar, Type] = {}

        # Tracks what expressions have had constraints "loaded" from their
        # types.
        self.has_been_touched: Set[VerificationVar] = set()

        # Maps variable names to the variable names containing them.
        # Used to track what variables to invalidate if another variable is
        # invalidated.
        self.dependencies: Dict[VerificationVar, Set[VerificationVar]] = {}

        # Maps bound refinement variable names to term variables.
        # TODO: change from string to a tagged union, as for now we're using
        # the "RSelf" string to deal with RSelf, which should be safe, but we'd
        # rather not.
        self.bound_var_to_name: Dict[str, VerificationVar] = {}

        # Are we currently loading info from a variable and thus don't need to
        # recurse deeper?
        self.currently_loading_from_var: bool = False

    def add_smt_constraints(
            self,
            constraints: list[SMTConstraint]
    ) -> None:
        self.smt_solver.add(constraints)

    def add_constraints(
            self,
            constraints: list[RefinementConstraint],
            *, ctx: Context
    ) -> None:
        cons = self.translate_all(constraints, ctx=ctx)
        self.add_smt_constraints(cons)

    def save_dependencies(self, var: VerificationVar) -> None:
        # This accumulates the sub variables as we go, so that if we have:
        # a.b.c.d, then for a.b.c we'll add d, and a.b we'll add c and d, etc.
        acc_vars = set((var,))
        for sv in var.subvars():
            sv_deps = self.dependencies.setdefault(sv, set())
            sv_deps |= acc_vars
            acc_vars.add(sv)

    def touch(self, var: VerificationVar) -> None:
        self.has_been_touched.add(var)

    def is_active(self, var: VerificationVar) -> bool:
        """Checks if a variable or any sub variables are in var_versions.
        """
        if var in self.var_versions:
            return True
        for dep in self.dependencies.get(var, set()):
            if self.is_active(dep):
                return True
        return False

    def store_type(self, var: VerificationVar, var_type: Type) -> None:
        self.var_types[var] = var_type

    def type_for(self, var: VerificationVar) -> Optional[Type]:
        """Possibly this will fallback to the literal hash stuff?
        """
        return self.var_types.get(var)

    def fresh_verification_var(self) -> FreshVar:
        self.next_id += 1
        return FreshVar(self.next_id)

    def fresh_smt_var(self, var: VerificationVar) -> SMTVar:
        self.next_id += 1
        name = var.fullpath()
        smt_var = z3.Int(f"{name}#{self.next_id}", ctx=self.smt_context)
        self.smt_variables.append(smt_var)
        return smt_var

    def get_smt_expr(
            self,
            var: VerificationVar
    ) -> SMTExpr:
        self.save_dependencies(var)
        if var in self.var_versions:
            return self.var_versions[var]
        else:
            fresh_var = self.fresh_smt_var(var)
            self.var_versions[var] = fresh_var
            return fresh_var

    def invalidate_var(self, var: VerificationVar) -> None:
        """Invalidate a variable, forcing the creation of a new smt variable
        with no associated constraints the next time it is used.
        """
        # TODO: do I need to delete from self.var_types? I don't think so,
        # because has_been_touched fulfills that role.
        if var in self.var_versions:
            self.has_been_touched.add(var)
            del self.var_versions[var]
        for dep in self.dependencies.get(var, set()):
            self.invalidate_var(dep)

    def invalidate_vars_in_expr(self, expr: Expression) -> None:
        """Invalidate all mentions of `RealVar`s in an expression.
        """
        invalidator = Invalidator(self)
        expr.accept(invalidator)

    def load_from_type(
            self,
            expr_var: VerificationVar,
            expr_type: Type,
            *, ctx: Context
    ) -> None:
        # TODO: how do I deal with other variables mentioned in this, e.g.,
        # other properties of an object? Thinking about it maybe uniquing
        # the refinement variables in semantic analysis would help with
        # that. That way refinement variables would know they're talking
        # about the other properties when those come in...

        # So fully explaining this: first we check that neither the variable nor
        # any dependencies are in var_versions. This is a heuristic to determine
        # whether or not we've loaded from something and that's currently
        # active, in which case another load would duplicate stuff. (This may
        # need revision.)
        # We then check that it hasn't been loaded from before -- so that we
        # only have one load for non-const stuff -- or that the type we'd be
        # loading from is const.
        # TODO: I think there's a better way to do this that involves marking
        # variables as constant in a separate set/map. A problem with this is
        # that stuff that's const could be invalidated by being assigned to.
        # TODO: currently I think there are problems, because e.g. something
        # depending on another value might be unloaded, have that value change,
        # and then be reloaded with a new condition. So that's a question.
        # TODO: further, the declaration of a parent as Const should probably
        # influence the child properties? E.g. declaring self as Const in
        # forward should make any Conv2ds Const inside forward.
        # NOTE: also, I kind of want to make ints automatically const, but that
        # can wait. Also if they're properties they can change? Hmmmmm. Yeah,
        # dealing with that is a headache that can wait. (The parent might be
        # passed as an argument and invalidate them as children, which I'd have
        # to distinguish from them being invalidated for being used on their
        # own.)
        if (isinstance(expr_type, BaseType)
                and expr_type.refinements
                and not self.is_active(expr_var)
                and (expr_var not in self.has_been_touched
                    or expr_type.refinements.is_const)):
            self.touch(expr_var)
            var_bindings: list[tuple[str, VerificationVar]] = [
                    # TODO: this is kind of a hack. Ideally I would
                    # translate RSelf to some kind of local refinement
                    # variable or something in check_assignment when
                    # the type is inferred.
                    ("RSelf", expr_var)
                    ]
            if (ref_var := expr_type.refinements.var) is not None:
                var_bindings.append((ref_var.name, expr_var))

            with self.var_bindings(var_bindings):
                self.add_constraints(expr_type.refinements.constraints,
                        ctx=ctx)

    def load_from_var(
            self,
            var: VerificationVar,
            *, ctx: Context
    ) -> None:
        if self.currently_loading_from_var:
            return
        self.currently_loading_from_var = True
        base = var.copy_base()
        base_type: Optional[Type] = None
        props = var.props.copy()
        
        while True:
            if (var_type := self.type_for(base)):
                base_type = var_type

            if base_type:
                var_copy = base.copy()
                self.load_from_type(var_copy, base_type, ctx=ctx)
            
            if props == []:
                break
            prop = props.pop(0)
            if base_type:
                if isinstance(prop, int):
                    base_type = None
                elif isinstance(prop, str):
                    base_type = analyze_member_access(
                            prop, base_type, ctx, is_lvalue=False,
                            is_super=False, is_operator=True, msg=self.msg,
                            original_type=base_type, chk=self.chk,
                            in_literal_context=False)
                else:
                    assert False, "impossible by type of prop"
            base.props.append(prop)
        self.currently_loading_from_var = False

    def load_from_sub_exprs(
            self,
            expr: Expression,
            *, ctx: Context
    ) -> None:
        """This function goes through all sub expressions, checking their types
        and then seeing if those types are refinement types and then trying to
        load from them.

        The expression should successfully parse as a RealVar.
        """

        def sub_expr(e: Expression) -> Optional[Expression]:
            if isinstance(acc_expr, NameExpr):
                return None
            elif isinstance(acc_expr, MemberExpr):
                return acc_expr.expr
            elif isinstance(acc_expr, IndexExpr):
                return acc_expr.base
            else:
                assert False, "should be impossible"

        acc_expr = expr
        # This skips the first one, since this is just for sub expressions.
        while (acc_expr := sub_expr(acc_expr)):
            var = to_real_var(acc_expr)
            assert var is not None
            expr_type = self.chk.expr_checker.accept(acc_expr)
            self.store_type(var, expr_type)
            self.load_from_type(var, expr_type, ctx=ctx)

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

    def translate_expr(self, expr: RefinementExpr,
            *, ext_props: Optional[list[VarProp]] = None) -> SMTExpr:
        """Tranlsate a refinement expression into an SMT expression the smt
        solver can use.
        """
        if ext_props is None:
            ext_props = []
        if isinstance(expr, RefinementLiteral):
            return z3.IntVal(expr.value, ctx=self.smt_context)
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
        elif isinstance(expr, RefinementSelf):
            if "RSelf" in self.bound_var_to_name:
                # This means that we've bound RSelf to a specific variable, so
                # we're checking a call site usage or something.
                var = self.bound_var_to_name["RSelf"].extend(
                        expr.props + ext_props)
            else:
                # Otherwise we're checking the body of the function containing
                # this.
                var = RealVar("self", expr.props + ext_props)
            self.save_dependencies(var)
            self.load_from_var(var, ctx=expr)
            return self.get_smt_expr(var)
        elif isinstance(expr, RefinementVar):
            # Resolve anything where we have a term variable m, but we use the
            # refinement variable R to refer to it in constraints.
            default_var = RealVar(expr.name)
            base = self.bound_var_to_name.get(expr.name, default_var)
            var = base.extend(expr.props + ext_props)
            self.save_dependencies(var)
            self.load_from_var(var, ctx=expr)

            return self.get_smt_expr(var)
        else:
            assert False, "Should not be passed"

    def translate_all(
            self,
            constraints: list[RefinementConstraint],
            *, ctx: Context
    ) -> list[SMTConstraint]:
        return [smt_c
                for c in constraints
                for smt_c in self.translate_to_constraints(c, ctx=ctx)]

    def translate_to_constraints(
            self,
            con: RefinementConstraint,
            *, ctx: Context
    ) -> list[SMTConstraint]:
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
        # if not isinstance(left, z3.ArithRef) and not isinstance(right, z3.ArithRef):
        #     self.fail("A refinement constraint must include at least one "
        #             "refinement variable", ctx)

        if kind == ConstraintKind.EQ:
            result = left == right
        elif kind == ConstraintKind.NOT_EQ:
            result = left != right
        elif kind == ConstraintKind.LT:
            result = left < right
        elif kind == ConstraintKind.LT_EQ:
            result = left <= right
        elif kind == ConstraintKind.GT:
            result = left > right
        elif kind == ConstraintKind.GT_EQ:
            result = left >= right
        else:
            assert False, "Impossible!"

        if isinstance(result, bool):
            return z3.BoolVal(result, ctx=self.smt_context)
        else:
            return result

    def add_bound_var(
            self,
            term_var: VerificationVar,
            ref_var: str,
            *, ctx: Context,
            ) -> None:
        if ref_var in self.bound_var_to_name:
            var = self.bound_var_to_name[ref_var]
            if var in self.var_versions:
                # We only throw an error if we're using the same refinement
                # variable for a different term variable.
                if term_var != var:
                    self.fail('Tried to bind already bound refinement '
                            'variable "{}"'.format(ref_var), ctx)
                return

        self.bound_var_to_name[ref_var] = term_var
        return

    def add_var(
            self,
            var: VerificationVar,
            typ: Type,
            *, ctx: Context
            ) -> None:
        if not isinstance(typ, BaseType) or typ.refinements is None:
            return
        info = typ.refinements

        if info.var is not None:
            self.add_bound_var(var, info.var.name, ctx=ctx)

        self.add_constraints(info.constraints, ctx=ctx)

    def add_argument(self, arg_name: str, typ: Type, *, ctx: Context) -> None:
        var = RealVar(arg_name)
        self.add_var(var, typ, ctx=ctx)

    def alias_as(self, source: VerificationVar, alias: VerificationVar) -> None:
        if source in self.var_versions:
            self.var_versions[alias] = self.var_versions[source]
            del self.var_versions[source]
        for dep in self.dependencies.get(source, set()):
            prop_len = len(source.props)
            assert source.props == dep.props[:prop_len]
            new_alias = alias.extend(dep.props[prop_len:])
            self.dependencies.setdefault(alias, set()).add(new_alias)
            self.alias_as(dep, new_alias)

    def add_lvalue(self, lvalue: Lvalue, typ: Type) -> None:
        var = to_real_var(lvalue)
        if var is None:
            return
        self.add_var(var, typ, ctx=lvalue)

    def add_inferred_lvalue(self, lvalue: Lvalue, typ: Type) -> None:
        """Add a verification var with constraints based on the inferred type.

        Unlike `add_lvalue`, this doesn't bind the refinement variable.
        """
        if not isinstance(typ, BaseType) or typ.refinements is None:
            return
        info = typ.refinements
        var = to_real_var(lvalue)
        if var is None:
            return
        self.touch(var)

        if info.verification_var:
            print("aliasing")
            self.alias_as(info.verification_var, var)
        else:
            print("rebinding")
            bindings: list[tuple[str, VerificationVar]] = [
                    ("RSelf", var)
                    ]
            if info.var:
                bindings.append((info.var.name, var))
            with self.var_bindings(bindings):
                self.add_constraints(info.constraints, ctx=lvalue)

    def overwrite_inferred_lvalue(
            self,
            lvalue: Lvalue,
            rvalue: Expression,
            rvalue_type: Type
    ) -> None:
        """This is used to overwrite an lvalue that was not declared with a
        type.
        """
        rvar, _ = self.var_from_expr(rvalue, rvalue_type)
        lvar = to_real_var(lvalue)
        if not (rvar and lvar):
            return

        intermediary = self.fresh_verification_var()
        self.alias_as(rvar, intermediary)
        self.invalidate_vars_in_expr(lvalue)
        self.invalidate_vars_in_expr(rvalue)

        self.alias_as(intermediary, lvar)

    def overwrite_lvalue(
            self,
            lvalue: Lvalue,
            lvalue_type: Type,
            rvalue: Expression,
            rvalue_type: Type
    ) -> None:
        """The goal is to have this be a toggle that switches between checking
        the value against a declared type and just overwritting an lvalue with
        an inferred type.
        """
        # TODO: I can use a flag on the type to say whether a refinement type
        # was inferred or not, and only set it for the end thing amabobber.

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
        else:
            # TODO: find better way of doing this (when I switch to using proper
            # tagged enums for bound vars/otherwise overhaul bound vars). Right now
            # we just bind RSelf because it won't be used unless it needs to be.
            var_bindings.append(("RSelf", fresh_var))
        with self.var_bindings(var_bindings):
            self.add_constraints(ret_type.refinements.constraints, ctx=ctx)
        return fresh_var

    def real_var_from_expr(
            self,
            expr: Expression,
            expr_type: Type
    ) -> Optional[VerificationVar]:
        """Attempt to directly convert an expression to a real verification
        variable.
        """
        var = to_real_var(expr)

        if var is None:
            return None

        assert expr_type is not None, "Just a quick check"

        self.store_type(var, expr_type)
        self.load_from_type(var, expr_type, ctx=expr)
        self.load_from_sub_exprs(expr, ctx=expr)

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
            smt_var = self.get_smt_expr(expr_type.refinements.verification_var)
            return smt_var
        else:
            var = self.real_var_from_expr(expr, expr_type)
            if var:
                return self.get_smt_expr(var)

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
        self.var_versions[fresh_var] = smt_expr

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
            self.var_versions[fresh_var] = expr.value
            return fresh_var, False
        elif isinstance(expr, CallExpr):
            assert expr_type is not None, "I don't think this should happen?"

            if is_refined_type(expr_type) and expr_type.refinements is not None:
                return expr_type.refinements.verification_var, False
            else:
                return None, False
        elif isinstance(expr, TupleExpr):
            var = self.fresh_verification_var()
            has_sent_error = False
            for i, item in enumerate(expr.items):
                if isinstance(expr_type, TupleType):
                    idx_type = expr_type.items[i]
                else:
                    idx_type = None
                var_idx = var.extend([i])
                item_var, idx_error = self.var_from_expr(item, idx_type)
                has_sent_error = has_sent_error or idx_error
                if item_var is None:
                    continue
                item_smt = self.get_smt_expr(item_var)
                idx_smt = self.get_smt_expr(var_idx)
                self.add_smt_constraints([idx_smt == item_smt])
            return var, has_sent_error
        elif (is_refined_type(expr_type)
                and expr_type.refinements
                and expr_type.refinements.verification_var):
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

        SHOULD_LOG = False

        if SHOULD_LOG:
            print("var_versions:", {k: v
                for k, v in self.var_versions.items()
                if isinstance(k, RealVar)})
            print("Given:", self.smt_solver)
            print("Goal:", constraints)

        # Basically, in order to prove that the constraints are "valid" --
        # evaluates to true for all possible variable values -- we put a not
        # around the condition and then check that conditions is unsatisfiable
        # -- that we have no way it can be *untrue*.
        cond = z3.Not(z3.And(constraints))
        try:
            result = self.smt_solver.check(cond)
        except z3.Z3Exception as exc:
            print("exception:", exc)
            return False
        print("Result:", "valid" if result == z3.unsat else "invalid")
        if result == z3.sat and SHOULD_LOG:
            print("Counter example:", self.smt_solver.model())
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
        # TODO: this means that we never deal with expr_type being None. Is that
        # right? Changing it gives errors, so...
        if not (isinstance(expr_type, BaseType)
                and isinstance(target, BaseType)):
            return

        info = target.refinements
        if info is None:
            return

        def check():
            FAIL_MSG = "refinement type check failed"
            constraints = self.translate_all(info.constraints, ctx=ctx)
            if constraints != []:
                print(f"Location: {ctx.line}:{ctx.column}")
            if not self.check_implication(constraints):
                self.fail(FAIL_MSG, ctx)

        if expr is None or info.var is None:
            # assert expr is not None or info.var is not None

            check()
            return
        else:
            var, has_sent_error = self.var_from_expr(expr, expr_type)
            if var is None:
                if not has_sent_error:
                    self.fail('could not understand expression', ctx)
                return
            with self.var_binding(info.var.name, var):
                check()
                return

    def check_call_args(
            self,
            bindings: list[Tuple[Type, Expression, Type]],
            *, ctx: Context
    ) -> None:
        """In order to allow all of the bindings to exist for each of the
        different arguments to access each other we have to recursively add
        to the bindings.
        """
        if bindings == []:
            return

        expected_type, expr, expr_type = bindings[0]
        info = expected_type.refinements if isinstance(expected_type, BaseType) else None

        if info is None:
            self.check_call_args(bindings[1:], ctx=ctx)
            return

        def check():
            FAIL_MSG = "refinement type check failed"
            constraints = self.translate_all(info.constraints, ctx=ctx)
            if constraints != []:
                print(f"Location: {ctx.line}:{ctx.column}")
            if not self.check_implication(constraints):
                self.fail(FAIL_MSG, ctx)

        if info.var is None:
            check()
            self.check_call_args(bindings[1:], ctx=ctx)
        else:
            var, has_sent_error = self.var_from_expr(expr, expr_type)
            if var is None:
                if not has_sent_error:
                    self.fail('could not understand expression', ctx)
                return
            with self.var_binding(info.var.name, var):
                check()
                self.check_call_args(bindings[1:], ctx=ctx)

    def check_init(self, target: Type, *, ctx: Context) -> None:
        """This is a specialized version of check_subsumption for the
        ending of `__init__`.
        """
        self.check_subsumption(None, None, target, ctx=ctx)


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
        if e.analyzed:
            e.analyzed.accept(self)
        else:
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
