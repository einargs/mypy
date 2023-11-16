from contextlib import contextmanager
import torch
from z3 import *

x, y, z = Ints('x y z')

I = IntSort()
B = BoolSort()

s = Solver() #SolverFor("HORN")

@contextmanager
def stack():
    s.push()
    yield None
    s.pop()


# p, q = Bools('p q')

#s.push()
#s.add(Not(q))
#s.assert_and_track(q, p)
#print(s.check())
#print(s.unsat_core())
#s.pop()
#
#
#s.push()
#s.assert_and_track((x + 5) == 10, p)
#s.assert_and_track(x == 6, q)
#
#print(s.check())
#print(s.unsat_core())
#s.pop()


any_tuple = Datatype("any tuple")
any_tuple.declare('cons', ('car', I), ('cdr', any_tuple))
any_tuple.declare('nil')
any_tuple = any_tuple.create()

def to_any_tuple(e: ExprRef) -> ExprRef:
    # maybe have a hash table from sorts to sizes
    sort = e.sort()
    con = sort.constructor(0)
    size = con.arity()

    out = any_tuple.nil
    for i in reversed(range(size)):
        mem = sort.accessor(0, i)(e)
        out = any_tuple.cons(mem, out)
    return out

tup = Datatype("tup")
tup.declare("mk", ('fst', I), ('snd', I))
tup = tup.create()

with stack():

    t1 = tup.mk(1, 2)
    t2 = any_tuple.cons(1, any_tuple.cons(2, any_tuple.nil))
    cond = to_any_tuple(t1) == t2
    print(cond)
    s.add(cond)
    print(s.check())

iseq = SeqSort(I)

def to_seq(e: ExprRef) -> ExprRef:
    sort = e.sort()
    size = sort.constructor(0).arity()

    return Concat(*(Unit(sort.accessor(0, i)(e)) for i in range(size)))
    # out = Unit(sort.accessor(0, 0)(e))
    # for i in range(1, i):
    #     out = out + Unit(sort.accessor(0, i)(e))
    # return out

with stack():
    t1 = tup.mk(1, 2)
    t3 = Const('t3', SeqSort(I))
    t2 = Unit(IntVal(1)) + Unit(IntVal(2))
    v = Int('v')
    s.add(t3 == to_seq(tup.mk(v, 2)))
    print(s.check(to_seq(t1)[1] == IntVal(2)))

    long_seq1 = Concat(*(Unit(IntVal(i)) for i in range(8)))

    long_seq2 = Concat(SubSeq(long_seq1,0,5), Unit(IntVal(20)), SubSeq(long_seq1,6,Length(long_seq1)))
    s.check()
    m = s.model()
    print(m.eval(long_seq2))

def local(l, e, rest):
    return substitute(rest, (l, e))

with stack():
    # def f(x, y):
    # havoc x0;
    # assume x0 = x;
    # havoc y0;
    # assume y0 = y;
    # assume x == 4 and y > 0;
    # ret = x + y;
    # assert ret > 4;
    #fpre = Function('fpre', I, I, B)
    
    f = Function('f', I, I, I, B)
    main = Function('main', I, I, B)
    ret, x0, y0 = Ints('ret x0 y0')

    #s.add(fpre == Lambda([x, y], And(x == 4, y > 0)))

    def fpre(x, y):
        return And(x == 4, y > 0)

    s.add(
        ForAll([x0, x, y0, y, ret],
        Implies(And(x0 == x, y0 == y),
        Implies(fpre(x, y),
        local(ret, x + y,
        And(ret > 4, f(x0, y0, ret)
    ))))))

    # def main(x):
    # assume x > 0;
    # y = f(4, x);
    # assert y > 4;
    # z = f(4, 2);
    # assert z > 4
    s.add(
        ForAll([x0, x, ret],
        Implies(x0 == x,
        Implies(x > 0,
        And(fpre(4, x), ForAll(y, Implies(f(4, x, y),
        And(y > 5,
        And(fpre(4, -2), ForAll(z, Implies(f(4, -2, z),
        And(z > 5)
    )))))))))))

    #s.add(ForAll([x,ret], Implies(x > 0, main(x, ret))))
    print(s.check())

