class RefinementVar:
    name: str

    def __init__(self, name):
        self.name = name

enable_refinement_type_checking = object()

T = RefinementVar('T')
