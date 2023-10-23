from sympy import Basic


class Formulas(Basic):

    def __str__(self):
        if len(self.args) > 1:
            return "Formulas(%s)" % ", ".join([str(f) for f in self.args])
        return str(self.args[0])


class FormulasOr(Basic):

    def __str__(self):
        if len(self.args) > 1:
            return " or ".join([str(f) for f in self.args])
        return str(self.args[0])



