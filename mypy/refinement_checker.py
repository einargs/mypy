"""Checks the refinement representation.
"""
from typing import Dict, Any, Union, Optional, Tuple, Iterator, Set
from typing_extensions import TypeAlias
from contextlib import contextmanager

from mypy.nodes import Context
from mypy.refinement_ast import (
    RType, RTupleType, RIntType, RBoolType, RNoneType, RClassType,
    RExpr, RName, RFree, RMember, RIndex, RArith, RLogic, RCmp,
    RIntLiteral, RArithOp, RLogicOp, RCmpOp, RLoc, RVar, RCond,
    RStmt, RDecl, RHavoc, RExprAssign, RExit, RAssert, RAssume,
    RLenExpr, RDupExpr, RFoldExpr, RTupleExpr, RNoneExpr,
    rexpr_substitute,
)
from mypy.messages import MessageBuilder
import z3


SMTVar: TypeAlias = z3.ArithRef
SMTSort: TypeAlias = z3.SortRef
SMTSeq: TypeAlias = z3.SeqRef
SMTData: TypeAlias = z3.DatatypeRef
SMTDupUnionSort: TypeAlias = z3.DatatypeRef
SMTTuple: TypeAlias = z3.DatatypeRef
SMTClass: TypeAlias = z3.DatatypeRef
SMTDataSort: TypeAlias = z3.DatatypeSortRef
SMTDupUnionSort: TypeAlias = z3.DatatypeSortRef
SMTTupleSort: TypeAlias = z3.DatatypeSortRef
SMTClassSort: TypeAlias = z3.DatatypeSortRef
SMTExpr: TypeAlias = z3.ExprRef
SMTConstraint: TypeAlias = z3.BoolRef


def get_base_of(expr: RLoc) -> RVar:
    if isinstance(expr, (RName, RFree)):
        return expr
    elif isinstance(expr, RMember):
        return get_base_of(expr.base)
    elif isinstance(expr, RIndex):
        return get_base_of(expr.base)
    else:
        assert False, "Structure of RLoc was malformed"


class RefinementError(Exception):
    """These are user facing type errors.
    """
    def __init__(self, msg: str, ctx: Context):
        self.msg = msg
        self.ctx = ctx


class ContradictionError(RefinementError):
    """An error that means you fundamentally cannot type check past it.
    """
    pass


class RefinementChecker:
    def __init__(self) -> None:
        # Currently these both should only have RName and RFree
        self.types: Dict[RLoc, RType] = {}
        self.smt_vars: Dict[RLoc, SMTExpr] = {}

        # Smt stuff
        self.smt_context: z3.Context = z3.Context()
        self.smt_solver: z3.Solver = z3.Solver(ctx=self.smt_context)

        # Datatype stuff
        self.tuple_sorts: Dict[int, SMTTupleSort] = {}
        self.tuple_sort_sizes: Dict[SMTTupleSort, int] = {}
        self.class_sorts: Dict[str, SMTClassSort] = {}

        # Make the smt none datatype
        none_type = z3.Datatype("None", ctx=self.smt_context)
        none_type.declare("singleton")
        self.none_sort = none_type.create()

        self.seq_sort = z3.SeqSort(z3.IntSort(ctx=self.smt_context))

        # Constraints on the length of a sequence based on indexing into them.
        # Used to prevent duplicates.
        self.seq_len_constraints: Set[SMTExpr] = set()

        # self.wrapping_context: Optional[Context] = None

    def fail(self, msg: str, context: Context) -> None:
        self.msg.fail(msg, context)

    # NOTE: an interesting idea for a way to allow exceptions to be thrown
    # without having a context and getting them at various levels of
    # specificity. Not sure it's worth it yet.
    # @contextmanager
    # def with_ctx(ctx: Context) -> Iterator[None]:
    #     old = self.wrapping_context
    #     try:
    #         yield None
    #     except RefinementError as err:
    #         if err.ctx is None:
    #             err.ctx = ctx
    #         raise err
    #     finally:
    #         self.wrapping_context = old

    def tuple_to_seq(self, smt_expr: SMTExpr) -> SMTExpr:
        sort = smt_expr.sort()
        size = self.tuple_sort_sizes.get(sort)
        assert size is not None, "passed non-tuple argument"

        return z3.Concat(*(z3.Unit(sort.accessor(0, i)(smt_expr))
            for i in range(size)))

    def regularize_sorts(
            self,
            left: SMTExpr,
            right: SMTExpr
            ) -> Optional[Tuple[SMTExpr, SMTExpr]]:
        left_sort = left.sort()
        right_sort = right.sort()

        if left_sort == right_sort:
            return left, right

        def seq_and_tup(seq: SMTSort, tup: SMTSort) -> bool:
            return seq == self.seq_sort and tup in self.tuple_sort_sizes

        if seq_and_tup(left_sort, right_sort):
            return left, self.tuple_to_seq(right)
        elif seq_and_tup(right_sort, left_sort):
            return self.tuple_to_seq(left), right
        else:
            return None

    def build_tuple(self, members: list[SMTExpr]) -> SMTExpr:
        assert all(map(lambda e: isinstance(e, SMTExpr), members))
        size = len(members)
        sort = self.sort_for_type(RTupleType(size))
        return sort.init(*members)

    def sort_for_type(self, ty: RType) -> SMTSort:
        """Does not accept RNoneType.
        """
        assert not isinstance(ty, RNoneType)

        if isinstance(ty, RTupleType):
            if (sort := self.tuple_sorts.get(ty.size)) is not None:
                return sort

            if ty.size is None:
                return self.seq_sort
            else:
                tuple_type = z3.Datatype(f"{ty.size} tuple", ctx=self.smt_context)
                I = z3.IntSort(ctx=self.smt_context)
                tuple_type.declare("init", *[(f"project{i}", I) for i in range(ty.size)])
                sort = tuple_type.create()
                self.tuple_sorts[ty.size] = sort
                self.tuple_sort_sizes[sort] = ty.size
                return sort
        elif isinstance(ty, RIntType):
            return z3.IntSort(ctx=self.smt_context)
        elif isinstance(ty, RBoolType):
            return z3.BoolSort(ctx=self.smt_context)
        elif isinstance(ty, RNoneType):
            return self.none_sort
        elif isinstance(ty, RClassType):
            if (sort := self.class_sorts.get(ty.fullname)) is not None:
                return sort

            class_type = z3.Datatype(ty.fullname, ctx=self.smt_context)
            fields = [(attr, self.sort_for_type(ty.fields[attr]))
                    for attr in sorted(ty.fields.keys())]
            class_type.declare("init", *fields)
            sort = class_type.create()
            self.class_sorts[ty.fullname] = sort
            return sort
        else:
            assert False

    def declare(
            self,
            var: RExpr,
            ty: RType,
            ) -> None:
        if isinstance(var, (RName, RFree)):
            assert var not in self.types
            assert var not in self.smt_vars
            self.types[var] = ty
        # elif isinstance(var, RMember) and isinstance(var.base, RName):
        #     # This is specifically for new properties inside __init__

        #     # TODO: possibly add some kind of flag option that can be used to
        #     # communicate that we're inside an __init__ and inferred property
        #     # definitions are allowed.
        #     base_ty = self.type_of(var.base)
        #     if not isinstance(base_ty, RClassType):
        #         raise RefinementError("not an object", var.base)
        #     new_fields = base_ty.fields.copy()
        #     new_fields[var.attr] = ty
        #     new_ty = RClassType(base_ty.fullname, new_fields)
        #     print("new type", new_ty)
        #     if base_ty.fullname in self.class_sorts:
        #         del self.class_sorts[base_ty.fullname]
        #         new_sort = self.sort_for_type(new_ty)
        #         assert self.class_sorts[base_ty.fullname] == new_sort
        #         if var.base in self.smt_vars:
        #             del self.smt_vars[var.base]
        #     self.types[var.base] = new_ty
        else:
            raise RefinementError("did not understand declaration", var)


    def type_of(self, expr: RExpr) -> RType:
        # TODO: I may want to implement a cache for looking up types. It could
        # improve the complexity of some other stuff. OTOH, I don't think n
        # will be big enough for it to matter.
        if expr in self.types:
            return self.types[expr]

        if isinstance(expr, RIntLiteral):
            return RIntType()
        elif isinstance(expr, RCmp):
            return RBoolType()
        elif isinstance(expr, RLogic):
            return RBoolType()
        elif isinstance(expr, RArith):
            return RIntType()
        elif isinstance(expr, RIndex):
            ptype = self.type_of(expr.base)
            assert isinstance(ptype, RTupleType)
            assert expr.index < ptype.size
            return RIntType()
        elif isinstance(expr, RMember):
            ptype = self.type_of(expr.base)
            assert isinstance(ptype, RClassType)
            assert expr.attr in ptype.fields
            return ptype.fields[expr.attr]
        elif isinstance(expr, RFree):
            ty = self.types.get(expr)
            assert ty is not None, \
                f"Generated variable {expr} had no declared type"
            return ty
        elif isinstance(expr, RName):
            ty = self.types.get(expr)
            assert ty is not None, \
                f"{expr} had no declared type"
            return ty
        elif isinstance(expr, RLenExpr):
            return RIntType()
        elif isinstance(expr, RDupExpr):
            return RTupleType(expr.size)
        elif isinstance(expr, RFoldExpr):
            def read_static_int(e: RExpr) -> int:
                if isinstance(e, RIntLiteral):
                    return e.value
                else:
                    smt_var = self.var_for(e)
                    if not isinstance(smt_var, z3.IntNumRef):
                        raise RefinementError("expected static int", e)
                    return smt_var.as_long()

            var_type = self.type_of(expr.folded_var)
            expr.folded_var
            if expr.start is None:
                start = 0
            else:
                start = read_static_int(expr.start)
            if expr.end is None:
                assert isinstance(var_type, RTupleType)
                end = var_type.size
            else:
                end = read_static_int(expr.end)
            return RTupleType(end - start)
        elif isinstance(expr, RTupleExpr):
            return RTupleType(len(expr.members))
        elif isinstance(expr, RNoneExpr):
            return RNoneType()
        else:
            assert False

    def sort_of(self, expr: RExpr) -> SMTSort:
        ty = self.type_of(expr)
        return self.sort_for_type(ty)

    def fresh_const(self, expr: RLoc) -> SMTExpr:
        sort = self.sort_of(expr)
        return z3.FreshConst(sort, prefix=str(expr))

    def get_member_index(self, expr: Union[RMember, RIndex]) -> int:
        if isinstance(expr, RMember):
            base_ty = self.type_of(expr.base)
            assert isinstance(base_ty, RClassType)
            assert expr.attr in base_ty.fields, \
                f"{expr.attr} (expr: {expr}) was not a field in {base_ty.fullname} at {base_ty.line}:{base_ty.column}"
            # Iteration order is guarenteed
            return list(sorted(base_ty.fields.keys())).index(expr.attr)
        elif isinstance(expr, RIndex):
            base_ty = self.type_of(expr.base)
            assert base_ty.size is None or expr.index < base_ty.size
            return expr.index
        else:
            assert False

    def var_for(self, expr: RLoc) -> SMTExpr:
        return self.evaluate_expr(expr)

    def assign(self, expr: RLoc, new_value: SMTExpr) -> None:
        def datatype_assign(expr: RLoc, index: int, fresh_member: SMTExpr) -> None:
            """This exists as a recursive function for assign to call.
            """
            # First we build the fresh value for the base value we're passed.
            original = self.var_for(expr)
            sort = original.sort()
            if sort == self.seq_sort:
                fresh = z3.Concat(
                        z3.SubSeq(original, 0, index),
                        z3.Unit(fresh_member),
                        # Excess is just discarded; we can safely exceed the
                        # length of the sequence.
                        z3.SubSeq(original, index+1, z3.Length(original)))
            else:
                con = sort.constructor(0)
                size = con.arity()

                def transform(i: int) -> SMTExpr:
                    if i == index:
                        return fresh_member
                    else:
                        project = sort.accessor(0, i)
                        return project(original)
                members = map(transform, range(size))

                fresh = con(*members)

            if isinstance(expr, (RName, RFree)):
                self.smt_vars[expr] = fresh
            elif isinstance(expr, (RIndex, RMember)):
                idx = self.get_member_index(expr)
                datatype_assign(expr.base, idx, fresh)

        if isinstance(expr, (RName, RFree)):
            self.smt_vars[expr] = new_value
        elif isinstance(expr, (RMember, RIndex)):
            base_var = get_base_of(expr)

            if base_var not in self.smt_vars:
                var = self.fresh_const(base_var)
                self.smt_vars[base_var] = var

            idx = self.get_member_index(expr)
            datatype_assign(expr.base, idx, new_value)
        else:
            assert False, f"cannot assign to {expr}"

    def havoc(self, expr: RLoc) -> None:
        if isinstance(expr, (RName, RFree)):
            var = self.fresh_const(expr)
            self.smt_vars[expr] = var
            return var
        elif isinstance(expr, (RMember, RIndex)):
            base_var = get_base_of(expr)

            if base_var not in self.smt_vars:
                var = self.fresh_const(base_var)
                self.smt_vars[base_var] = var
            else:
                fresh = self.fresh_const(expr)
                self.assign(expr, fresh)
        else:
            assert False

    def const_eval(self, expr: SMTExpr) -> Optional[SMTExpr]:
        self.smt_solver.check()
        model = self.smt_solver.model()
        val = model.evaluate(expr)
        if self.smt_solver.check(expr != val) == z3.unsat:
            return val
        else:
            return None

    def constrain_index(
            self,
            seq_expr: SMTExpr,
            index: int,
            ctx: Context
            ) -> None:
        """Throws an error if the new constraint already conflicts.
        """
        assert seq_expr.sort() == self.seq_sort, f"{seq_expr} is not a sequence"

        cond = index < z3.Length(seq_expr)
        if cond in self.seq_len_constraints:
            return
        else:
            self.seq_len_constraints.add(cond)
        result = self.smt_solver.check(cond)
        if result != z3.sat:
            raise ContradictionError(f"index ({index}) out of bounds", ctx)
        else:
            self.smt_solver.add(cond)

    def get_tuple_index(self, expr: SMTExpr, at: int, expr_ctx: Context) -> SMTExpr:
        print("indexing", expr, "at", at)
        sort = expr.sort()
        if sort == self.seq_sort:
            self.constrain_index(expr, at, expr_ctx)
            return expr[at]
        elif sort in self.tuple_sort_sizes:
            project = sort.accessor(0, at)
            return project(expr)
        else:
            assert False, f"{expr} was not a tuple"
            

    def evaluate_fold_expr(self, expr: RFoldExpr) -> SMTExpr:
        in_tuple = self.evaluate_expr(expr.folded_var)

        in_tuple_sort = in_tuple.sort()
        if in_tuple_sort == self.seq_sort:
            in_len_expr = self.const_eval(z3.Length(in_tuple))
            if in_len_expr is None:
                raise RefinementError("size not statically known", expr.folded_var)
            in_len = in_len_expr.as_long()
        elif in_tuple_sort in self.tuple_sort_sizes:
            in_len = self.tuple_sort_sizes[in_tuple_sort]
        else:
            raise RefinementError("not a tuple", expr.folded_var)

        def eval_int(e: RExpr) -> int:
            var = self.evaluate_expr(e)
            if var.sort() != z3.IntSort(ctx=self.smt_context):
                raise RefinementError("not an integer", e)
            smt_val = self.const_eval(var)
            if smt_val is None:
                raise RefinementError("not statically known", e)
            val = smt_val.as_long()
            if val < 0:
                raise RefinementError("fold start cannot be less than 0", e)
            return val

        start = eval_int(expr.start) if expr.start else 0
        end = eval_int(expr.end) if expr.end else in_len

        if end < start:
            raise RefinementError(f"start ({start}) less than end ({end})", expr)

        offset = end - start

        if offset < 2:
            return in_tuple

        acc_var = RName(expr.acc_var)
        cur_var = RName(expr.cur_var)

        fold_body = expr.fold_expr
        for i in range(start, end-1):
            acc_expr = RIndex(expr.folded_var, end-1) if i == end-2 else expr.fold_expr

            fold_body = rexpr_substitute(fold_body,
                    {cur_var: RIndex(expr.folded_var, i), acc_var: acc_expr})

        fold_body_smt = self.evaluate_expr(fold_body)

        if in_tuple_sort == self.seq_sort:
            before_start = z3.SubSeq(in_tuple, 0, start)
            after_end = z3.SubSeq(in_tuple, end, in_len - end)
            out_tuple = z3.Concat(before_start, fold_body_smt, after_end)
        else:
            assert in_tuple_sort in self.tuple_sort_sizes
            before_start = [self.get_tuple_index(in_tuple, i, expr.folded_var) for i in range(start)]
            after_end = [self.get_tuple_index(in_tuple, i, expr.folded_var) for i in range(end, in_len)]

            out_tuple = self.build_tuple(before_start + [fold_body_smt] + after_end)

        return out_tuple

    def evaluate_expr(self, expr: RExpr) -> SMTExpr:
        assert expr is not None
        if isinstance(expr, (RName, RFree)):
            if expr not in self.smt_vars:
                print("smt_vars", self.smt_vars)
            assert expr in self.smt_vars, f"{expr} not declared"
            return self.smt_vars[expr]
        elif isinstance(expr, RIndex):
            base = self.evaluate_expr(expr.base)
            return self.get_tuple_index(base, expr.index, expr)
        elif isinstance(expr, RMember):
            base = self.evaluate_expr(expr.base)
            sort = base.sort()
            assert isinstance(sort, z3.DatatypeSortRef), \
                    f"unknown sort {sort}, type {base_ty}, for {expr.base}"
            idx = self.get_member_index(expr)
            project = sort.accessor(0, idx)
            return project(base)
        elif isinstance(expr, RArith):
            lhs = self.evaluate_expr(expr.lhs)
            rhs = self.evaluate_expr(expr.rhs)

            assert lhs.sort().is_int()
            assert rhs.sort().is_int()

            if expr.op == RArithOp.plus:
                return lhs + rhs
            if expr.op == RArithOp.minus:
                return lhs - rhs
            if expr.op == RArithOp.mult:
                return lhs * rhs
            if expr.op == RArithOp.div:
                return lhs / rhs
        elif isinstance(expr, RLogic):
            members = [self.evaluate_expr(a) for a in expr.args]
            if expr.op == RLogicOp.and_op:
                return z3.And(members)
            if expr.op == RLogicOp.or_op:
                return z3.Or(members)
        elif isinstance(expr, RCmp):
            lhs = self.evaluate_expr(expr.lhs)
            rhs = self.evaluate_expr(expr.rhs)

            tup = self.regularize_sorts(lhs, rhs)
            if tup is None:
                raise RefinementError("type mismatch with {expr.rhs}", expr.lhs)
            lhs, rhs = tup

            if expr.op == RCmpOp.eq:
                return lhs == rhs
            elif expr.op == RCmpOp.not_eq:
                return lhs != rhs
            elif expr.op == RCmpOp.lt:
                return lhs < rhs
            elif expr.op == RCmpOp.lt_eq:
                return lhs <= rhs
            elif expr.op == RCmpOp.gt:
                return lhs > rhs
            elif expr.op == RCmpOp.gt_eq:
                return lhs >= rhs
        elif isinstance(expr, RIntLiteral):
            return z3.IntVal(expr.value, ctx=self.smt_context)
        elif isinstance(expr, RLenExpr):
            smt_tuple = self.evaluate_expr(expr.expr)
            sort = smt_tuple.sort()
            if (size := self.tuple_sort_sizes.get(sort)) is not None:
                return z3.IntVal(size, ctx=self.smt_context)
            elif sort == self.seq_sort:
                return z3.Length(smt_tuple)
            else:
                raise RefinementError("did not evaluate to a tuple", expr.expr)
        elif isinstance(expr, RDupExpr):
            base = self.evaluate_expr(expr.expr)
            sort = base.sort()
            base_ty = self.type_of(expr.expr)
            if isinstance(base_ty, RIntType):
                assert sort.is_int()
                return self.build_tuple([base] * expr.size)
            elif isinstance(base_ty, RTupleType):
                if base_ty.size is None:
                    assert sort == self.seq_sort
                else:
                    assert base_ty.size == expr.size
                    assert self.tuple_sort_sizes.get(sort) == expr.size
                return base
        elif isinstance(expr, RFoldExpr):
            return self.evaluate_fold_expr(expr)
        elif isinstance(expr, RTupleExpr):
            return self.build_tuple([self.evaluate_expr(m) for m in expr.members])
        elif isinstance(expr, RNoneExpr):
            return self.none_sort.singleton
        else:
            assert False, f"unhandled expr type {type(expr)}"

    def assume(self, cond: RExpr) -> None:
        smt_expr = self.evaluate_expr(cond)
        self.smt_solver.add(smt_expr)

    def check_cond(self, cond: RCond) -> None:
        print("checking", cond.expr)
        smt_expr = self.evaluate_expr(cond.expr)
        CONFIG = {
            "should_log": True,
            "show_statistics": False,
            "show_vars": True,
            "show_raw_vars": False,
            "show_priors": True,
            # Check if the existing conditions are satisfiable before checking
            # the smt expressions.
            "check_priors": True,
        }

        if CONFIG["check_priors"]:
            assert self.smt_solver.check() == z3.sat, "priors were unsatisfiable"

        # Basically, in order to prove that the constraints are "valid" --
        # evaluates to true for all possible variable values -- we put a not
        # around the condition and then check that conditions is unsatisfiable
        # -- that we have no way it can be *untrue*.
        # We could even turn on the smt solver's produce proof mode to allow
        # us to extract a proof of this.
        try:
            result = self.smt_solver.check(z3.Not(smt_expr))
        except z3.Z3Exception as exc:
            print("exception:", exc)
            return False
        print("Result:", "valid" if result == z3.unsat else "invalid",
                "raw:", result)
        if result == z3.sat and CONFIG["should_log"]:
            if CONFIG["show_statistics"]:
                print("Statistics:", self.smt_solver.statistics())
            model = self.smt_solver.model()
            if CONFIG["show_vars"]:
                for k, v in self.smt_vars.items():
                    if z3.is_ast(v):
                        print(f"    {k}: {v}  ==>  {model.eval(v)}")
                    elif CONFIG["show_raw_vars"]:
                        print(f"    {k}: {v}")
            if CONFIG["show_priors"]:
                print("Given:", self.smt_solver)
            def eval_inside(expr: z3.ExprRef) -> z3.ExprRef:
                # The kinds to go inside
                kinds = (
                        z3.Z3_OP_AND,
                        z3.Z3_OP_OR,
                        z3.Z3_OP_NOT,
                        z3.Z3_OP_IMPLIES,
                        z3.Z3_OP_DISTINCT,
                        z3.Z3_OP_EQ,
                        z3.Z3_OP_LT,
                        z3.Z3_OP_GT,
                        z3.Z3_OP_LE,
                        z3.Z3_OP_GE
                        )
                if z3.is_app(expr) and expr.decl().kind() in kinds:
                    decl = expr.decl()
                    children = map(eval_inside, expr.children())
                    return decl(*children)
                else:
                    return model.eval(expr)
            print("Goals:")
            if z3.is_app(smt_expr) and smt_expr.decl().kind() == z3.Z3_OP_AND:
                constraints = smt_expr.children()
            else:
                constraints = [smt_expr]
            for constraint in constraints:
                if z3.is_app(constraint):
                    narrowed = eval_inside(constraint)
                    print(f"    {constraint}   ===>   {narrowed}")
                else:
                    print(f"    {constraint}")
            #print("Counter example:", self.smt_solver.model())
            print()
        else:
            print("Given:", self.smt_solver)

        if result != z3.unsat:
            raise RefinementError("refinement type check failed", cond)

    def interpret_stmt(self, stmt: RStmt) -> None:
        print("interpreting", stmt)
        if isinstance(stmt, RHavoc):
            self.havoc(stmt.var)
        elif isinstance(stmt, RDecl):
            self.declare(stmt.var, stmt.type)
        elif isinstance(stmt, RExprAssign):
            # TODO: integrate information from the type field.
            # This will especially tricky considering that type information
            # for self inside __init__ might be incomplete?
            print("assign", stmt.expr, "to", stmt.var)
            try:
                smt_expr = self.evaluate_expr(stmt.expr)
                print("evaluates to", smt_expr)
                self.assign(stmt.var, smt_expr)
            except RefinementError as err:
                self.havoc(stmt.var)
                raise err
        elif isinstance(stmt, RExit):
            # This currently doesn't have any purpose; it's here for the future
            # when we have to deal with control flow.
            pass
        elif isinstance(stmt, RAssert):
            self.check_cond(stmt.cond)
        elif isinstance(stmt, RAssume):
            self.assume(stmt.expr)

    def check(self, stmts: list[RStmt], msg: MessageBuilder) -> None:
        for stmt in stmts:
            try:
                self.interpret_stmt(stmt)
            except RefinementError as err:
                if err.ctx.line != -1 and err.ctx.column != -1:
                    msg.fail(err.msg, err.ctx)
                    if isinstance(err, ContradictionError):
                        break
                else:
                    print("don't understand location")
                    raise err

    # TODO: an analysis to see if stuff is used before it's declared maybe?
    # To catch problems in e.g. function refinements. OTOH, maybe that could
    # be useful for something.
    # NOTE: Actually I think I can kind of catch that during translation --
    # I'll build up the substitution list as I go.
