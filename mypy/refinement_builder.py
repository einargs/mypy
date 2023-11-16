from typing import (
    Dict, Any, Union, Optional, Tuple, Callable, overload, Iterator, Literal,
    FrozenSet, Sequence,
)
from typing_extensions import TypeAlias
from contextlib import contextmanager

from mypy.refinement_ast import (
    RType, RIntType, RTupleType, RClassType, RBoolType, RNoneType,
    RExpr, RName, RFree, RMember, RIndex, RArith, RLogic, RCmp,
    RIntLiteral, RArithOp, RLogicOp, RCmpOp, RCond, RVar, RLoc,
    RStmt, RDecl, RHavoc, RExprAssign, RExit, RAssert, RAssume,
    rexpr_substitute, RNoneExpr, RLenExpr, RFoldExpr,
    RTupleExpr, RClassHoleType, rexpr_uses_self, RDupUnionType,
)
from mypy.types import (
    BaseType, RefinementInfo, Type, Instance, TupleType, NoneType, UnionType,
    RefinementOptions, LiteralType,
)
from mypy.nodes import (
    Expression, Context, ComparisonExpr, Var, TypeInfo,
    OpExpr, MemberExpr, IndexExpr, RefExpr, LambdaExpr,
    NameExpr, IntExpr, TupleExpr, CallExpr, SliceExpr,
)
from mypy.messages import MessageBuilder


def parse_rexpr(
        expr: Expression,
        *, from_type: bool
        ) -> Optional[RExpr]:
    def parse(expr: Expression) -> Optional[RExpr]:
        """An internal function to avoid having to pass around options.
        """
        # TODO: improve error messages, e.g. for fold and dup.
        # TODO: add in the context stuff from the expressions
        if isinstance(expr, IntExpr):
            return RIntLiteral(expr.value).set_line(expr)

        elif isinstance(expr, OpExpr):
            if expr.op == '+':
                op = RArithOp.plus
            elif expr.op == '-':
                op = RArithOp.minus
            elif expr.op == '*':
                op = RArithOp.mult
            elif expr.op == '//':
                op = RArithOp.div
            else:
                return None

            return RArith(parse(expr.left), op, parse(expr.right)).set_line(expr)

        elif isinstance(expr, ComparisonExpr):
            def transform(tup: Tuple[str, Expression, Expression]) -> RExpr:
                op_str, left_e, right_e = tup
                op: RCmpOp
                if op_str == "==":
                    op = RCmpOp.eq
                elif op_str == "!=":
                    op = RCmpOp.not_eq
                elif op_str == "<":
                    op = RCmpOp.lt
                elif op_str == "<=":
                    op = RCmpOp.lt_eq
                elif op_str == ">":
                    op = RCmpOp.gt
                elif op_str == ">=":
                    op = RCmpOp.gt_eq
                else:
                    return None

                left = parse(left_e)
                right = parse(right_e)
                if left is None or right is None:
                    return None
                assert isinstance(right, Context), f"Unexpected type {type(right)} for {right_e}"
                return RCmp(left, op, right).set_line(left, end_line=right.end_line)

            pairs = list(map(transform, expr.pairwise()))

            if any(map(lambda e: e is None, pairs)):
                return None

            if len(pairs) == 1:
                return pairs[0]
            else:
                return RLogic(RLogicOp.and_op, pairs).set_line(expr)

        elif isinstance(expr, CallExpr):
            if not isinstance(expr.callee, NameExpr):
                return None

            if expr.callee.name == "len" and len(expr.args) == 1:
                base = parse(expr.args[0])
                return RLenExpr(base).set_line(expr)

            elif (from_type
                    and expr.callee.name == "fold"
                    and len(expr.args) == 2):
                fold_lambda = expr.args[0]
                if not (isinstance(fold_lambda, LambdaExpr)
                        and len(fold_lambda.arg_names) == 2):
                    return None
                [acc_var, cur_var] = fold_lambda.arg_names
                fold_expr = parse(fold_lambda.expr())

                if fold_expr is None:
                    return None

                subscript = expr.args[1]
                if not (isinstance(subscript, IndexExpr)
                        and isinstance(subscript.index, SliceExpr)
                        and subscript.index.stride is None):
                    return None
                
                folded_var = parse(subscript.base)
                if folded_var is None:
                    return None

                if subscript.index.begin_index is None:
                    start = None
                else:
                    start = parse(subscript.index.begin_index)
                    if start is None:
                        return None

                if subscript.index.end_index is None:
                    end = None
                else:
                    end = parse(subscript.index.end_index)
                    if end is None:
                        return None

                return RFoldExpr(
                        acc_var,
                        cur_var,
                        fold_expr,
                        folded_var,
                        start,
                        end).set_line(expr)
            else:
                return None
        elif isinstance(expr, NameExpr):
            if expr.fullname == "builtins.None":
                return RNoneExpr()
            if from_type and expr.name == "RSelf":
                return RName("self").set_line(expr)
            else:
                return RName(expr.name).set_line(expr)
        elif isinstance(expr, MemberExpr):
            base = parse(expr.expr)
            return RMember(base, expr.name).set_line(expr)
        elif isinstance(expr, IndexExpr) and isinstance(expr.index, IntExpr):
            base = parse(expr.base)
            return RIndex(base, expr.index.value).set_line(expr)
        elif isinstance(expr, TupleExpr):
            items = list(map(parse, expr.items))
            if any(map(lambda e: e is None, items)):
                return None
            return RTupleExpr(items).set_line(expr)
        else:
            return None

    return parse(expr)


class ParsedType:
    """An intermediary type that base types are converted to.

    `base` is the parsed type. If it's None, that means the type is foreign.
    `cond` is the RExpr for the refinement condition that might be attached.
    `var` is the possible variable representing the value of a term typed by
    this in the refinement condition.

    If it has a var, then it has a cond and base type. If it has a cond, it
    has a base type.

    A parsed type is basically just a RefinementInfo -- it's just that a
    ParsedType is meant to have accumulated all of the conditions for all
    sub-fields.
    """
    # TODO: consider replacing cond with a list for more accurate error
    # locations.
    def __init__(
            self,
            base: RType,
            cond: Optional[RExpr] = None,
            var: Optional[RExpr] = None,
            eval_expr: Optional[RExpr] = None):
        self.base = base
        self.cond = cond
        self.var = var
        self.eval_expr = eval_expr

    def cond_substitute(self, new_var: RExpr, substs: Dict[RVar, RExpr] = {}) -> Optional[RExpr]:
        if self.var and self.cond:
            return rexpr_substitute(self.cond, substs | {self.var: new_var})
        else:
            return self.cond

    def __repr__(self) -> str:
        var_str = f"{self.var}: " if self.var else ""
        par_str = f"{{{var_str}{self.cond}}}" if self.cond else ""
        return f"{self.base}{par_str}"


class RefBuildError(Exception):
    """User-facing errors that occur during construction of the refinement
    representation.
    """
    def __init__(self, msg: str, ctx: Context) -> None:
        self.msg = msg
        self.ctx = ctx


class RefinementBuilder:
    def __init__(self, msg: MessageBuilder) -> None:
        self.msg = msg

        self.next_id = 0

        self.stmts: list[RStmt] = []

        # Are we building a function?
        self.in_function = False

        # Can be None if not in a function or because we have no return
        # value.
        self.return_var: Optional[RFree] = None
        self.return_type: Optional[RType] = None

        # Can be None if not in a function or because we have no post
        # condition.
        self.post_condition: Optional[RExpr] = None

        # Holds the bindings for the refinement variables.
        self.ref_var_bindings: Dict[RVar, RExpr] = {}

        self.skip_types = frozenset((
            "builtins.list",
            "builtins.dict",
            "builtins.str",
            "builtins.set",
            "builtins.float",
        ))

    @contextmanager
    def log_errors(self) -> Iterator[None]:
        try:
            yield None
        except RefBuildError as err:
            if err.ctx.line != -1 and err.ctx.column != -1:
                self.msg.fail(err.msg, err.ctx)
            else:
                raise err

    def substitute(
            self,
            body: RExpr,
            substs: Optional[Dict[RVar, RExpr]] = None
            ) -> RExpr:
        """Substitution including all refinement variables declared inside
        in-scope definitions. Should be used for types defined in-scope.
        """
        if substs is None:
            substs = self.ref_var_bindings
        else:
            substs = self.ref_var_bindings | substs

        return rexpr_substitute(body, substs)

    def cond_substitute(self, ty: ParsedType, new_var: RExpr) -> Optional[RExpr]:
        return ty.cond_substitute(new_var, self.ref_var_bindings)

    def free_var(self, name: str) -> RFree:
        self.next_id += 1
        return RFree(name, self.next_id)

    def add(self, stmt: Union[RStmt, list[RStmt]]) -> None:
        if isinstance(stmt, list):
            self.stmts += stmt
        else:
            assert isinstance(stmt, RStmt)
            self.stmts.append(stmt)

    def assert_(
            self,
            expr: RExpr,
            *, ctx: Context
            ) -> None:
        assert ctx.line != -1 and ctx.column != -1, f"no location in {ctx}"
        self.add(RAssert(RCond(expr,
            line=ctx.line,
            column=ctx.column)))

    def define_func(
            self,
            ret_type: Optional[ParsedType],
            bindings: list[Tuple[str, ParsedType]]
            ) -> None:
        """This specifically means that we're building a function body here.
        """
        # print("define_func::bindings",
        #         list(map(lambda t: t[1].base, bindings)),
        #         "ret_type", ret_type.base if ret_type else None)

        self.in_function = True

        # Here we declare all of the argument variables.
        for name, ty in bindings:
            var = RName(name)
            self.add(RDecl(var, ty.base))
            self.add(RHavoc(var))
            if ty.var:
                self.ref_var_bindings[ty.var] = var
            if ty.cond:
                self.add(RAssume(self.substitute(ty.cond)))

        if ret_type is not None:
            self.return_type = ret_type.base

            if ret_type.var is not None:
                self.return_var = self.free_var("return")
                self.add(RDecl(self.return_var, self.return_type))

            if ret_type.cond is not None:
                subst = {}
                # Here we generate frozen versions of the first versions of the
                # variables for use in the post condition.
                for name, ty in bindings:
                    if ty.var:
                        var = RName(name)
                        frozen = self.free_var(f"{name}0")
                        subst[ty.var] = frozen
                        self.add(RDecl(frozen, ty.base))
                        self.add(RHavoc(frozen))
                        self.add(RAssume((RCmp(frozen, RCmpOp.eq, var))))
                if ret_type.var is not None:
                    subst[ret_type.var] = self.return_var
                self.post_condition = rexpr_substitute(
                        ret_type.cond,
                        subst)

    @overload
    def insert_call(
            self,
            ret_type: ParsedType,
            bindings: list[Tuple[RExpr, ParsedType]]
            ) -> RVar: ...

    def insert_call(
            self,
            ret_type: Optional[ParsedType],
            bindings: list[Tuple[RExpr, ParsedType]]
            ) -> Optional[RVar]:
        """Rather than having a specific representation of calls inside the
        refinement representation, we instead just insert the precondition
        checks and then the post condition assumptions with a havoc applied
        to a variable declared to be the result type.

        Only returns None if ret_type is None. (Overload?)
        """
        # print("insert_call::bindings",
        #         list(map(lambda t: t[1], bindings)),
        #         "ret_type", ret_type)

        # Here we handle declaring, assigning, and adding the precondition
        # assertions for all arguments.
        substs = {}
        for i, (val_expr, ty) in enumerate(bindings):
            arg_name = f"arg{i}"
            arg_var = self.free_var(arg_name)
            if ty.var:
                substs[ty.var] = arg_var

            self.add(RDecl(arg_var, ty.base))
            self.add(RExprAssign(arg_var, ty.base, val_expr))
            if ty.cond:
                self.assert_(rexpr_substitute(ty.cond, substs), ctx=val_expr)

        # Here we declare, havoc (bc return value can be anything), and state
        # post-condition assumptions.
        if ret_type is not None:
            ret_var: RFree
            if ret_type.var:
                ret_var = self.free_var(f"ret:{ret_type.var}")
                substs[ret_type.var] = ret_var
            else:
                ret_var = self.free_var("ret")
            self.add(RDecl(ret_var, ret_type.base))
            self.add(RHavoc(ret_var))
            if ret_type.cond:
                out = rexpr_substitute(ret_type.cond, substs)
                self.add(RAssume(out))
            return ret_var
        else:
            return None

    def insert_assignment(
            self,
            lhs: RExpr,
            left_ty: Optional[ParsedType],
            rhs: RExpr,
            right_ty: ParsedType,
            *, is_def: bool
            ) -> None:
        if left_ty is not None:
            typing = left_ty.base
        else:
            typing = right_ty.base

        if isinstance(lhs, (RName, RFree)) and (is_def or left_ty):
            self.add(RDecl(lhs, typing))

        self.add(RExprAssign(lhs, typing, rhs))
        if left_ty and left_ty.var:
            self.ref_var_bindings[left_ty.var] = lhs
        if left_ty and left_ty.cond:
            cond = self.cond_substitute(left_ty, lhs)
            self.assert_(cond, ctx=cond)

    def insert_return(
            self,
            value: Optional[RExpr],
            *, ctx: Context
            ) -> None:
        assert self.in_function, "return outside of a function"

        if value:
            # assert self.return_type is not None
            assert not self.post_condition or (self.post_condition and self.return_var)
            if self.return_var:
                self.add(RExprAssign(self.return_var,
                    self.return_type, value))

        if self.post_condition:
            self.assert_(self.post_condition, ctx=ctx)

    # Then this section below is the interface between the outside types
    # and the pure refinement logic above.

    def parse_rtype(
            self, ty: Type, *, allow_union: bool = False
            ) -> Optional[RType]:
        """Just parse an RType from a Type.
        """
        def all_ints(members: Sequence[Type]) -> bool:
            return all(isinstance(self.parse_rtype(t), RIntType) for t in members)
        if not isinstance(ty, BaseType):
            return None

        if ty.refinements:
            options = ty.refinements.options
        else:
            options = RefinementOptions.default()

        rtype: RType

        if isinstance(ty, Instance):
            fullname = ty.type.fullname
            if fullname == "builtins.int":
                rtype = RIntType()
            elif fullname == "builtins.bool":
                rtype = RBoolType()
            elif fullname == "builtins.None":
                rtype = RNoneType()
            elif fullname == "builtins.tuple":
                if all_ints(ty.args):
                    if len(ty.args) == 1:
                        rtype = RTupleType(None)
                    elif len(ty.args) == 0:
                        assert False, "I'm not sure this can happen"
                        return None
                    else:
                        rtype = RTupleType(len(ty.args))
                else:
                    return None
            elif fullname in self.skip_types:
                return None
            else:
                for info in ty.type.mro:
                    if info.fullname == "builtins.tuple":
                        rtype = RTupleType(None)
                        break
                else:
                    rtype = RClassHoleType(ty.type)
        elif isinstance(ty, NoneType):
            rtype = RNoneType()
        elif isinstance(ty, UnionType):
            # To accept a union as a valid thing, we need to have the expand_var
            # option enabled so that the value of the variable is consistent.
            if not (options['expand_var'] and allow_union):
                return None
            if not len(ty.items) == 2:
                return None
            ty1 = self.parse_rtype(ty.items[0])
            ty2 = self.parse_rtype(ty.items[1])

            # TODO: this discards any constraints on the components. Maybe
            # include a warning about that?

            if ty1 is None or ty2 is None:
                return None

            if isinstance(ty1, RIntType) and isinstance(ty2, RTupleType):
                if ty2.size is None:
                    return None
                rtype = RDupUnionType(ty2.size)
            elif isinstance(ty2, RIntType) and isinstance(ty1, RTupleType):
                if ty1.size is None:
                    return None
                rtype = RDupUnionType(ty1.size)
            else:
                return None
        elif isinstance(ty, TupleType):
            if all_ints(ty.items):
                rtype = RTupleType(len(ty.items))
            else:
                return None
        else:
            return None

        return rtype.set_line(ty)

    def elaborate_type(
            self,
            ty: RType,
            ) -> RType:
        """If `ty` is a RClassType, convert it into an RClassType.
        """
        cache: Dict[str, RType] = {}

        skip_fields = frozenset((
            "__hash__",
            "__dict__",
            "__module__",
            "__annotations__",
            "__doc__",
            "__constants__",
        ))

        def elab(ty: RType, parents: FrozenSet[str]) -> Optional[RType]:
            nonlocal cache
            if isinstance(ty, RClassHoleType):
                # We shouldn't have anything from skip_types inside even at
                # the start.
                assert ty.type.fullname not in self.skip_types
                assert ty.type.fullname not in parents
                if ty.type.fullname in cache:
                    rtype = cache[ty.type.fullname]
                    return RClassType(rtype.fullname, rtype.fields).set_line(ty)

                fields: Dict[str, RType] = {}
                rtype = RClassType(ty.type.fullname, fields)
                cache[ty.type.fullname] = rtype

                new_parents = parents | frozenset((ty.type.fullname,))

                for info in ty.type.mro:
                    for name, table_node in info.names.items():
                        if name in skip_fields:
                            continue
                        if name in fields:
                            continue
                        node = table_node.node
                        if isinstance(node, Var):
                            if node.type is None:
                                continue
                            if (isinstance(node.type, Instance) and
                                    (node.type.type.fullname in new_parents
                                        or node.type.type.fullname in self.skip_types)):
                                continue
                            parsed = self.parse_rtype(node.type)
                            if parsed:
                                if isinstance(parsed, RClassHoleType):
                                    fields[name] = elab(parsed, new_parents)
                                else:
                                    fields[name] = parsed
                return RClassType(ty.type.fullname, fields).set_line(ty)
            else:
                return ty

        return elab(ty, frozenset())

    def parse_type(self, ty: Type, *, allow_union: bool = False) -> Optional[ParsedType]:
        def parse(ty: Type, visited: FrozenSet[str]) -> Optional[ParsedType]:
            if not isinstance(ty, BaseType):
                return None
            info = ty.refinements or RefinementInfo(None, [])
            constraints = info.constraints.copy()

            if info.var is None:
                base_var = self.free_var("v")
            else:
                base_var = info.var

            used_base_var = False

            rtype = self.parse_rtype(ty, allow_union=allow_union)

            if rtype is None:
                return None

            def load_tuple_constraints(members: list[Type]) -> None:
                nonlocal constraints, used_base_var, visited
                tmp_constraints = []
                for i, member in enumerate(members):
                    parsed = parse(member, visited)
                    assert isinstance(parsed.base, RIntType)
                    if parsed.cond is not None:
                        if parsed.var is None:
                            cond = parsed.cond
                        else:
                            used_base_var = True
                            idx_var = RIndex(base_var, i)
                            cond = rexpr_substitute(parsed.cond,
                                    {parsed.var: idx_var})
                        tmp_constraints.append(cond)
                constraints += tmp_constraints

            if isinstance(rtype, RClassHoleType):
                visited |= frozenset((rtype.type.fullname,))
                def not_visited(type: Type) -> bool:
                    nonlocal visited
                    if isinstance(type, Instance):
                        return type.type.fullname not in visited
                    return True
                for name, table_node in rtype.type.names.items():
                    node = table_node.node
                    if (isinstance(node, Var)
                            and node.type is not None
                            and not_visited(node.type)):
                        parsed = parse(node.type, visited)
                        if parsed is not None:
                            if parsed.cond is not None:
                                substs: Dict[RName, RExpr] = {}
                                if parsed.var is not None:
                                    used_base_var = True
                                    mem_var = RMember(base_var, name)
                                    substs[parsed.var] = mem_var
                                # if rexpr_uses_self(parsed.cond):
                                #     used_base_var = True
                                #     mem_var = RMember(base_var, name)
                                #     substs[RName("self")] = mem_var
                                new_cond = rexpr_substitute(parsed.cond, substs)

                                constraints.append(new_cond)
            elif isinstance(rtype, RTupleType):
                if rtype.size is not None:
                    if isinstance(ty, Instance):
                        assert ty.type.fullname == "builtins.tuple"
                        assert len(ty.args) > 1
                        load_tuple_constraints(ty.args)
                    elif isinstance(ty, TupleType):
                        load_tuple_constraints(ty.items)
                    else:
                        # These should be the only forms that can produce an
                        # RTupleType.
                        assert False

            if info.var:
                out_var = info.var
            elif used_base_var:
                out_var = base_var
            else:
                out_var = None

            if len(constraints) > 1:
                full_cond = RLogic(RLogicOp.and_op, constraints).set_line(constraints[0])
            elif len(constraints) == 1:
                full_cond = constraints[0]
            else:
                full_cond = None

            return ParsedType(
                    rtype,
                    full_cond,
                    out_var,
                    info.eval_expr)

        return parse(ty, frozenset())

    @overload
    def parse_typed_expr(
            self,
            raw_expr: Expression,
            raw_type: Type,
            ) -> Tuple[RExpr, ParsedType]: ...

    @overload
    def parse_typed_expr(
            self,
            raw_expr: Expression,
            raw_type: Type,
            *, throw: Literal[False]
            ) -> Optional[Tuple[RExpr, ParsedType]]: ...

    def parse_typed_expr(
            self,
            raw_expr: Expression,
            raw_type: Type,
            *, throw: bool = True
            ) -> Optional[Tuple[RExpr, ParsedType]]:
        assert raw_type is not None
        try:
            ty = self.parse_type(raw_type)
            if ty is None:
                raise RefBuildError(f"could not parse type {raw_type}", raw_expr)
            if ty.eval_expr:
                return ty.eval_expr, ParsedType(ty.base)
            else:
                expr = parse_rexpr(raw_expr, from_type=False)
                if expr is None:
                    raise RefBuildError("could not parse", raw_expr)
                return expr, ty
        except Exception as err:
            if throw:
                raise err
            else:
                return None

    def build_assignment(
            self,
            raw_lhs: Expression,
            raw_left_ty: Optional[Type],
            raw_rhs: Expression,
            raw_right_ty: Type,
            *, is_def: bool
            ) -> None:
        """Build an assignment.
        """
        with self.log_errors():
            lhs = parse_rexpr(raw_lhs, from_type=False)
            if lhs is None:
                return
            if raw_left_ty is None:
                left_ty = None
            else:
                left_ty = self.parse_type(raw_left_ty)

                if left_ty is None:
                    return
            tup = self.parse_typed_expr(
                    raw_rhs, raw_right_ty, throw=False)
            if tup is None:
                return
            rhs, right_ty = tup

            self.insert_assignment(lhs, left_ty, rhs, right_ty, is_def=is_def)

    def build_func_def(
            self,
            raw_ret_type: Type,
            raw_bindings: list[Tuple[str, Type]]
            ) -> None:
        """Build a function definition.
        """
        with self.log_errors():
            ret_type = self.parse_type(raw_ret_type)
            bindings = [(name, ty)
                    for name, raw_ty in raw_bindings
                    if (ty := self.parse_type(raw_ty)) is not None]
            self.define_func(ret_type, bindings)

    def build_return(
            self,
            raw_expr: Expression,
            raw_type: Type,
            ) -> None:
        assert self.in_function
        with self.log_errors():
            expr, _ = self.parse_typed_expr(raw_expr, raw_type)
            self.insert_return(expr, ctx=raw_expr)

    def build_empty_return(self, *, ctx: Context) -> None:
        assert self.in_function
        with self.log_errors():
            self.insert_return(None, ctx=ctx)

    # def build_index_from_call(
    #         self,
    #         raw_ret_type: Type,
    #         raw_bindings: list[Tuple[Expression, Type, Type]],
    #         ) -> Optional[RIndex]:
    #     """This gets called when callable_name ends with __getitem__.
    #     """
    #     if not len(raw_bindings) == 2:
    #         return None

    #     raw_base, raw_base_type, _ = raw_bindings[0]
    #     raw_index, _, _ = raw_bindings[1]

    #     if not isinstance(raw_index, IntExpr):
    #         return None
    #     index = raw_index.value

    #     tup = self.parse_typed_expr(raw_base, raw_base_type, throw=False)
    #     if tup is None:
    #         return None
    #     base, _ = tup

    #     print("built index")
    #     return RIndex(base, index)

    def build_ref_expr(self, raw_expr: RefExpr, type: Type) -> Type:
        """Basically this just adds an rexpr as an eval_expr.
        """
        if not isinstance(type, BaseType):
            return type
        rexpr = parse_rexpr(raw_expr, from_type=False)
        if rexpr is None:
            return type
        if type.refinements:
            ref_info = type.refinements.substitute(rexpr)
        else:
            ref_info = RefinementInfo(None, [], rexpr)
        new_type = type.copy_with_refinements(ref_info)
        return new_type

    def build_call(
            self,
            raw_ret_type: Type,
            # (expr, expr_type, expected_type)
            raw_bindings: list[Tuple[Expression, Type, Type]],
            *, call_ctx: Context,
            callable_name: Optional[str],
            ) -> Optional[Type]:
        def package_eval_expr(e: RExpr, s: Dict[RVar, RExpr]) -> Type:
            e.set_line(call_ctx)
            if raw_ret_type.refinements:
                ref_info = raw_ret_type.refinements.substitute(e, s)
            else:
                ref_info = RefinementInfo(None, [], e)
            new_ret_type = raw_ret_type.copy_with_refinements(ref_info)
            return new_ret_type

        with self.log_errors():
            if callable_name is not None and callable_name.endswith("__getitem__"):
                return None

            ret_type = self.parse_type(raw_ret_type)

            bindings: list[Tuple[RExpr, ParsedType]] = []
            substs: Dict[RVar, RExpr] = {}
            for raw_expr, raw_expr_type, raw_expected_type in raw_bindings:
                expected_type = self.parse_type(raw_expected_type,
                        allow_union=True)
                if expected_type is None:
                    continue

                tup = self.parse_typed_expr(
                        raw_expr, raw_expr_type, throw=False)
                if tup is None:
                    continue
                expr, expr_type = tup

                # The expand option
                expand_enabled = (raw_expected_type.refinements and
                        raw_expected_type.refinements.options['expand_var'])
                if isinstance(expected_type.base, RDupUnionType):
                    assert expand_enabled, "parse_type should prevent this"
                    size = expected_type.base.size
                    expected_type.base = RTupleType(size)
                    if isinstance(expr_type.base, RIntType):
                        expr = RTupleExpr([expr] * size)
                    else:
                        assert (isinstance(expr_type.base, RTupleType)
                                and expr_type.base.size == size), \
                            "type checking should prevent this"
                elif expand_enabled:
                    self.fail("Expand option requires union of int and "
                            "tuple of ints", raw_expected_type)

                bindings.append((expr, expected_type))
                if expected_type.var:
                    substs[expected_type.var] = expr

            eval_expr = self.insert_call(ret_type, bindings)

            if eval_expr is not None:
                eval_expr.set_line(call_ctx)
                if raw_ret_type.refinements:
                    ref_info = raw_ret_type.refinements.substitute(eval_expr, substs)
                else:
                    ref_info = RefinementInfo(None, [], eval_expr)
                new_ret_type = raw_ret_type.copy_with_refinements(ref_info)
                return new_ret_type
            else:
                return None

    def build_arith_op(
            self,
            raw_expr: OpExpr,
            raw_result_type: Type,
            raw_left_type: Type,
            raw_right_type: Type
            ) -> Optional[Type]:
        with self.log_errors():
            if raw_expr.op == '+':
                op = RArithOp.plus
            elif raw_expr.op == '-':
                op = RArithOp.minus
            elif raw_expr.op == '*':
                op = RArithOp.mult
            elif raw_expr.op == '//':
                op = RArithOp.div
            else:
                return None

            result_type = self.parse_type(raw_result_type)
            if result_type is None:
                return None
            if not isinstance(result_type.base, RIntType):
                return None

            left_tup = self.parse_typed_expr(raw_expr.left, raw_left_type, throw=False)
            right_tup = self.parse_typed_expr(raw_expr.right, raw_right_type, throw=False)

            if left_tup is None or right_tup is None:
                return None

            left, _ = left_tup
            right, _ = right_tup

            result_expr = RArith(left, op, right)
            ref_info = RefinementInfo(None, [], result_expr)
            new_result = raw_result_type.copy_with_refinements(ref_info)
            return new_result

    def finalize_stmts(self) -> list[RStmt]:
        """Get the final form of the statements. Should only be invoked after
        all statements that will be used have been built.

        Convert any `RClassHoleType`s into `RClassType`s.

        May also be used for later things.
        """
        def transform(stmt: RStmt) -> RStmt:
            if isinstance(stmt, RDecl):
                return RDecl(stmt.var,
                        self.elaborate_type(stmt.type)).set_line(stmt)
            elif isinstance(stmt, RExprAssign):
                return RExprAssign(
                        stmt.var,
                        self.elaborate_type(stmt.ty),
                        stmt.expr).set_line(stmt)
            else:
                return stmt

        return list(map(transform, self.stmts))

