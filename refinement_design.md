# Testing
Use the below code to run the modified mypy on the example file.

```bash
python3 -m mypy example/example.py --no-incremental --show-traceback
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

# TODO
- enable assignment to variables
  * invalidate verification vars that are assigned to
  * Add variables with refinement types to things.
  * if the type of a variable is inferred and should be a refined type, make
    sure it is.
  * test multiple assignments
  * currently I don't think this can handle adding refinement type info to
    indexed lvalues.
- enable calling functions with refinement types
- make sure that return values of refined functions that are assigned to
  variables have refined types inferred.
- arithmetic operations in refinement annotations
- expand the passing of verification variables attached to types inside
  `ExprChecker` to also do the constraint generation for integer literals.
  Possibly even set it up to do it for `RealVar`s. This will make stuff like
  generating constraints for arithmetic operations or comparisons easier.
- Refinement tags
- get destructuring a tuple working
- add predicate/control flow based refinement
- write tests? Maybe
- design and implement the after condition stuff
- show an error if you use a variable when refining a value of type `None`. More
  generally this is about actually type checking the constraints on refinement
  variables. Technically not really necessary, but probably a nice to have.
- get it to understand integer operations
- inference of refinement types esp. for literals. This means taking `a = 1` and
  figuring out that the inferred type of `a` should be
  `Annotated[int, A, A == 1]`.

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
