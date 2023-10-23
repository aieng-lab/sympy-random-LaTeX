# Generated from C:/Git/Masterarbeit/src/sympy/parsing/latex\LaTeX.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .LaTeXParser import LaTeXParser
else:
    from LaTeXParser import LaTeXParser

# This class defines a complete generic visitor for a parse tree produced by LaTeXParser.

class LaTeXVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by LaTeXParser#math.
    def visitMath(self, ctx:LaTeXParser.MathContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#implication.
    def visitImplication(self, ctx:LaTeXParser.ImplicationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#formulas.
    def visitFormulas(self, ctx:LaTeXParser.FormulasContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#l_interval.
    def visitL_interval(self, ctx:LaTeXParser.L_intervalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#r_interval.
    def visitR_interval(self, ctx:LaTeXParser.R_intervalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#variable.
    def visitVariable(self, ctx:LaTeXParser.VariableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#set.
    def visitSet(self, ctx:LaTeXParser.SetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#in_set.
    def visitIn_set(self, ctx:LaTeXParser.In_setContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#relation.
    def visitRelation(self, ctx:LaTeXParser.RelationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#equality.
    def visitEquality(self, ctx:LaTeXParser.EqualityContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#text.
    def visitText(self, ctx:LaTeXParser.TextContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#expr.
    def visitExpr(self, ctx:LaTeXParser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#env.
    def visitEnv(self, ctx:LaTeXParser.EnvContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#cases_row.
    def visitCases_row(self, ctx:LaTeXParser.Cases_rowContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#cases.
    def visitCases(self, ctx:LaTeXParser.CasesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#matrix.
    def visitMatrix(self, ctx:LaTeXParser.MatrixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#matrix_row.
    def visitMatrix_row(self, ctx:LaTeXParser.Matrix_rowContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#additive.
    def visitAdditive(self, ctx:LaTeXParser.AdditiveContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#mp.
    def visitMp(self, ctx:LaTeXParser.MpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#operator.
    def visitOperator(self, ctx:LaTeXParser.OperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#mp_nofunc.
    def visitMp_nofunc(self, ctx:LaTeXParser.Mp_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#unary.
    def visitUnary(self, ctx:LaTeXParser.UnaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#unary_nofunc.
    def visitUnary_nofunc(self, ctx:LaTeXParser.Unary_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#postfix.
    def visitPostfix(self, ctx:LaTeXParser.PostfixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#postfix_nofunc.
    def visitPostfix_nofunc(self, ctx:LaTeXParser.Postfix_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#postfix_op.
    def visitPostfix_op(self, ctx:LaTeXParser.Postfix_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#eval_at.
    def visitEval_at(self, ctx:LaTeXParser.Eval_atContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#eval_at_sub.
    def visitEval_at_sub(self, ctx:LaTeXParser.Eval_at_subContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#eval_at_sup.
    def visitEval_at_sup(self, ctx:LaTeXParser.Eval_at_supContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#exp.
    def visitExp(self, ctx:LaTeXParser.ExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#exp_nofunc.
    def visitExp_nofunc(self, ctx:LaTeXParser.Exp_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#comp.
    def visitComp(self, ctx:LaTeXParser.CompContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#comp_nofunc.
    def visitComp_nofunc(self, ctx:LaTeXParser.Comp_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#group.
    def visitGroup(self, ctx:LaTeXParser.GroupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#abs_group.
    def visitAbs_group(self, ctx:LaTeXParser.Abs_groupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#number.
    def visitNumber(self, ctx:LaTeXParser.NumberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#special_symbols.
    def visitSpecial_symbols(self, ctx:LaTeXParser.Special_symbolsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#modifier.
    def visitModifier(self, ctx:LaTeXParser.ModifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#tuple.
    def visitTuple(self, ctx:LaTeXParser.TupleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#atom.
    def visitAtom(self, ctx:LaTeXParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#probability.
    def visitProbability(self, ctx:LaTeXParser.ProbabilityContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#bars.
    def visitBars(self, ctx:LaTeXParser.BarsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#bra.
    def visitBra(self, ctx:LaTeXParser.BraContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#ket.
    def visitKet(self, ctx:LaTeXParser.KetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#mathit.
    def visitMathit(self, ctx:LaTeXParser.MathitContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#mathit_text.
    def visitMathit_text(self, ctx:LaTeXParser.Mathit_textContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#frac.
    def visitFrac(self, ctx:LaTeXParser.FracContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#binom.
    def visitBinom(self, ctx:LaTeXParser.BinomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#floor.
    def visitFloor(self, ctx:LaTeXParser.FloorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#ceil.
    def visitCeil(self, ctx:LaTeXParser.CeilContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#func_normal.
    def visitFunc_normal(self, ctx:LaTeXParser.Func_normalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#func_bracket.
    def visitFunc_bracket(self, ctx:LaTeXParser.Func_bracketContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#func_binary_bracket.
    def visitFunc_binary_bracket(self, ctx:LaTeXParser.Func_binary_bracketContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#func.
    def visitFunc(self, ctx:LaTeXParser.FuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#args.
    def visitArgs(self, ctx:LaTeXParser.ArgsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#limit_sub.
    def visitLimit_sub(self, ctx:LaTeXParser.Limit_subContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#func_arg.
    def visitFunc_arg(self, ctx:LaTeXParser.Func_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#func_arg_noparens.
    def visitFunc_arg_noparens(self, ctx:LaTeXParser.Func_arg_noparensContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#subexpr.
    def visitSubexpr(self, ctx:LaTeXParser.SubexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#supexpr.
    def visitSupexpr(self, ctx:LaTeXParser.SupexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#subeq.
    def visitSubeq(self, ctx:LaTeXParser.SubeqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LaTeXParser#supeq.
    def visitSupeq(self, ctx:LaTeXParser.SupeqContext):
        return self.visitChildren(ctx)



del LaTeXParser