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
- maybe: bools (so that we understand not)

## VC Generation
For VC generation, the way it will work is that I'll have a binder that has a
list of VC conditions and then dictionary mapping raw in-context variables to an
incrementing ID. VC conditions will use a variable name + ID to indicate the
unique "version" of a variable. A variable being mutated or otherwise used in
such a way that existing VC conditions involving it become invalid will cause
the ID to increment, so that future uses of the variable will be different.

I'm going to need a stage where I convert `RefinementVar`s to either the actual
in-scope variables or "meta" variables that are constant throughout type
checking.

Entering a new function scope should put vc conditons on a stack (probably using
`@contextmanager`).

# TODO
- give literals refinement types
- enable tuples (do I use the SMT vector stuff? look into it)
- add predicate/control flow based refinement
- write tests? Maybe
- design and implement the after condition stuff
- show an error if you use a variable when refining a value of type `None`. More
  generally this is about actually type checking the constraints on refinement
  variables. Technically not really necessary, but probably a nice to have.
- get it to understand integer operations

# Fun syntax
I could take advantage of the recongition of slice syntax to do:

```python
int[V: V==4 and A == B]
Annotation[int, V: V==4 and A == B]
int[V: V == 4 : V == 6] # second `:` is after
int[V: V == 4, after: V == 6]
Annotation[int, V: V == 4, after: V == 6]
```