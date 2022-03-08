class RefinementLoc:
    def __getattr__(self, prop):
        return RefinementVarProp(self, prop)

    def __eq__(self, other):
        return RefinementConstraint(self, other)

    def __ne__(self, other):
        return RefinementConstraint(self, other)
    
    def __lt__(self, other):
        return RefinementConstraint(self, other)
    
    def __le__(self, other):
        return RefinementConstraint(self, other)
    
    def __gt__(self, other):
        return RefinementConstraint(self, other)
    
    def __ge__(self, other):
        return RefinementConstraint(self, other)

    def __contains__(self, item):
        return None


class RefinementVar(RefinementLoc):
    name: str

    def __init__(self, name):
        self.name = name


class RefinementVarProp(RefinementLoc):
    prop: str
    var: RefinementLoc

    def __init__(self, var, prop):
        self.var = var
        self.prop = prop


class RefinementInfo:
    def __init__(self, var: RefinementLoc, constraints):
        self.var = var
        self.constraints = constraints


class RefinementConstraint:
    def __init__(self, lhs: RefinementLoc, rhs: RefinementLoc):
        self.lhs = lhs
        self.rhs = rhs

    def __contains__(self, name):
        return RefineInfo(name, self)
