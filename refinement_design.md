# Testing
Use the command below to run the modified mypy on the example file.

```bash
python3 -m mypy example/example.py --no-incremental --show-traceback
```

Use the command below to run the refinement type specific tests. Currently since
I'm assuming that all annotations are refinement types, the annotation tests are
all broken.

```bash
pytest -n0 -k "check-refinement.test"
```

# Refinement Design
Dealing with aliasing is a problem here. Possibly in the future I could do a
rust-style ownership thing that prevents aliasing problems.

Sam does use data structures somewhat (i.e. tensors in lists) so that's
something to keep in mind.

We need to be able to refine:
- tuples of ints, both of any arity and specific arity
- ints
- None
- maybe: bools (so that they can be used for refined predicates)


## VC Generation
For VC generation, the way it will work is that I'll have a binder that has a
list of VC conditions and then dictionary mapping raw in-context variables to an
incrementing ID. VC conditions will use a variable name + ID to indicate the
unique "version" of a variable. A variable being mutated or otherwise used in
such a way that existing VC conditions involving it become invalid will cause
the ID to increment, so that future uses of the variable will be different.

I'm going to need a stage where I convert `RefinementVar`s to either the actual
in-scope variables or "meta" variables that are constant throughout type
checking. UPDATE: Actually, meta variables will just never be invalidated.

Entering a new function scope should put vc conditons on a stack (probably using
`@contextmanager`).

## Loading from refined types
I think that I should only load from refined types if they have a Const
modifier. If they don't, I should erase the extra type information and only rely
on the `verification_var` property so that type info doesn't get loaded from it.

## Refinement variable substitution in return types
In order to get correctly substituted return types, why don't I just modify the
returned type in the expr checker? The question is how to substitute for actual
refinement types...

Okay, so I've developed a trick where `BaseType`s can have a verification
variable added to them. This is used to allow me to directly

## Refinement Tags
This would act as a crude version of term-level function predicates.

```python
IsCat = RefinementTag('IsCat')

def is_cat(
    m: Annotated[Tensor, V, ...]
) -> Annotated[bool, B, after, Implies(B,
IsCat(V))]:
    ...
```

## Design Work
I really need to sit down and do some design work for a redesign from the bottom
up.

# TODO
## Tomorrow
0. Get RSelf working for methods.
1. Figure out final stuff.
2. Write the pytorch stubs.
3. Figure out default variables.
4. Fix the RSelf bound variable hack I'm using right now.
   - Develop refinement variables local to a given type for use in the RSelf
     stuff?

## Programmes
- Generate equality constraints for non-int types by generating constraints for
  all members of the type. This solves the tuple equality problem. This may use
  Z3's data type capabilities; it may not.
- General variable programme
  - Even more important given what I'm doing to pass around information on
    erased self arguments.
  - Overhaul the variable system so that bound variables aren't just strings.
  - Develop diagnostics and pretty printing of variables.
  - Maybe: Consolidate verification vars and refinement vars and such not into
    a cohesive model of a syntax implemented alongside substitution functions
    inside a separate module.
- Develop a restricted python syntax of expressions ala Z3 that allows you to
  define functions for use in refinement types.
- Get type aliases and generics working for refinement types.
- Maybe: automatic inference of equality constriants on RSelf for direct
  assignments of arguments in constructors?
- Consider coming up with some way of marking refinement types as being
  currently loaded from (this would require associating verification vars with
  types?). Currently I use `has_been_touched` to prevent loading the same
  information over and over again, but I suspect that this will cause problems
  with subproperties that have their own refinement information. I think I might
  need to make `has_been_touched` directly mark the "source" of a refinement
  constraint. I'm really not sure.


## Future Notes
- Currently subsumption checking for return statements uses the latest values.
  I'm going to leave this as is for now, and maybe when I introduce post
  conditions I'll go back and change this.

## General
- It currently can't understand accessing the result of a function return value.
- It looks like z3 has tools for inspecting the ast and even substituting within
  it -- see decl and substitute.
- Overhaul the bound variable system and fix the clunky RSelf hack I have right
  now.
- Start using the type variable infrastructure to uniquely identify refinement
  variables.
  - This could help with e.g. one property referring to another property -- the
    refinement variables would be unique'd so that if a second property is
    brought in later, any constraints involving it on the first property snap
    into place.
- figure out when a variable with a refinement type can be invalidated.
  Assignment shouldn't invalidate it bc then it'll be checked.
  - I'm thinking
- I need to prefix the refinement variables with the module they're defined in.
  (Maybe?)
- There's the possiblity for some cool interaction between refinements and
  the type of any length tuples, where refining them by equality or length could
  allow me to narrow the base type.
- arithmetic operations in refinement annotations
- expand the passing of verification variables attached to types inside
  `ExprChecker` to also do the constraint generation for integer literals.
  Possibly even set it up to do it for `RealVar`s. This will make stuff like
  generating constraints for arithmetic operations or comparisons easier.
- Refinement tags
- consider setting stuff up to automatically infer member equalities based on
  the values passed into constructors. So knowing that `a = Container(value=1)`
  means that `a.shape == 1`. I think this can maybe be done with just post
  conditions?
- get destructuring a tuple working
- add predicate/control flow based refinement
- debug tools to, e.g., show all constraints relevant to a refinement variable
  or term variable. Also, tools to check the validity of a condition inline.
- design and implement the after condition stuff
- show an error if you use a variable when refining a value of type `None`. More
  generally this is about actually type checking the constraints on refinement
  variables. Technically not really necessary, but probably a nice to have.
- get it to understand integer operations
- inference of refinement types esp. for literals. This means taking `a = 1` and
  figuring out that the inferred type of `a` should be
  `Annotated[int, A, A == 1]`.
- Can I remove `RefinementValue` from `mypy/types.py`?

## Needed tests
- [ ] Test how to deal with tuples of unknown arity? Does this require tuple
  equality?
- [ ] check invalidation of variables on:
  - [ ] assignment
  - [ ] use in an expression
  - [ ] when a property is invalidated
  - [ ] that a variable with a refined type is not invalidated by assignment
    because that is checked.
- [ ] test refinement variable uniqueness
  - [ ] when inferring a new type from a return value.
- [ ] test how this stuff interacts with overloaded functions

# Luke thoughts
- See how much I can do without needing to add annotations
- How much "why" information can I give?


# Fun syntax
I could take advantage of the recongition of slice syntax to do:

```python
int[V: V==4 and A == B]
Annotation[int, V: V==4 and A == B]
int[V: V == 4 : V == 6] # second `:` is after
int[V: V == 4, after: V == 6]
Annotation[int, V: V == 4, after: V == 6]
```
