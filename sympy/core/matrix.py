from .expr import Expr
from ..core.function import Function


class BasicMatrix(Expr):
    is_Matrix = True
    is_square = True # todo determine thats
    is_commutative = True
    infinite = False
    extended_negative = False
    zero = False
    is_MatrixExpr = False
    rows = 2

    pass


class BasicVector(Expr):

    def __len__(self):
        return len(self.args)
    pass




class Determinant(Function):
    pass

