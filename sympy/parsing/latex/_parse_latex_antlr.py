# Ported from latex2sympy by @augustt198
# https://github.com/augustt198/latex2sympy
# See license in LICENSE.txt
import re
from importlib.metadata import version
import sympy
from sympy import IndexedBase, Integer, Symbol, Function, Pow, Mul, S, evaluate, latex, NotAllowedSymbolException
from sympy.concrete.expr_with_intlimits import ExprWithOtherLimits
from sympy.core.function import MultiDerivative, Derivative, SimpleDerivative
from sympy.external import import_module
from sympy.parsing.latex.logic import is_logic_formula, StringFormula
from sympy.printing.str import StrPrinter
from sympy.parsing.latex.text import LaTeXText

from sympy.parsing.latex.errors import LaTeXParsingError
from sympy.stats import Expectation, Covariance, Variance

LaTeXParser = LaTeXLexer = MathErrorListener = None

try:
    LaTeXParser = import_module('sympy.parsing.latex._antlr.LaTeXParser',
                                import_kwargs={'fromlist': ['LaTeXParser']}).LaTeXParser
    LaTeXLexer = import_module('sympy.parsing.latex._antlr.LaTeXLexer',
                               import_kwargs={'fromlist': ['LaTeXLexer']}).LaTeXLexer
except Exception:
    pass

ErrorListener = import_module('antlr4.error.ErrorListener',
                              warn_not_installed=True,
                              import_kwargs={'fromlist': ['ErrorListener']}
                              )

known_symbols = {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta', 'iota',
                 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma', 'tau',
                 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
                 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi',
                 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega'}

not_allowed_symbols = {'star', 'dagger', 'diamond', 'circledcirc', 'bigtriangleup', 'phantom', 'hphantom', 'angle',
                       'circ', 'lcm', 'forall', 'exists', 'over', 'ast', 'vdash', 'models', 'aleph', 'frak', }

ignored_symbols = {'mbox'}

CHECK = True
def set_check(state):
    global CHECK
    CHECK = state

if ErrorListener:
    class MathErrorListener(ErrorListener.ErrorListener):  # type: ignore
        def __init__(self, src):
            super(ErrorListener.ErrorListener, self).__init__()
            self.src = src

        def syntaxError(self, recog, symbol, line, col, msg, e):
            fmt = "%s\n%s\n%s"
            marker = "~" * col + "^"

            if msg.startswith("missing"):
                err = fmt % (msg, self.src, marker)
            elif msg.startswith("no viable"):
                err = fmt % ("I expected something else here", self.src, marker)
            elif msg.startswith("mismatched"):
                names = LaTeXParser.literalNames
                expected = [
                    names[i] for i in e.getExpectedTokens() if i < len(names)
                ]
                if len(expected) < 10:
                    expected = " ".join(expected)
                    err = (fmt % ("I expected one of these: " + expected, self.src,
                                  marker))
                else:
                    err = (fmt % ("I expected something else here", self.src,
                                  marker))
            else:
                err = fmt % ("I don't understand this", self.src, marker)
            raise LaTeXParsingError(err)

def can_be_parsed(input_string):
    pattern = r'\d+\.\.\.\d+'
    return len(re.findall(pattern, input_string)) == 0

def parse_latex(sympy):
    if not can_be_parsed(sympy):
        raise LaTeXParsingError("Formula contains invalid pattern: <%s>" % sympy)

    if '$' in sympy:
        return LaTeXText(sympy)

    if is_logic_formula(sympy):
        return StringFormula(sympy, recurse=parse_latex)


    #print(sympy)
    antlr4 = import_module('antlr4')

    if None in [antlr4, MathErrorListener] or \
            not version('antlr4-python3-runtime').startswith('4.12'):
        raise ImportError("LaTeX parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime) or"
                          " conda (antlr-python-runtime), version 4.12")

    try:
        matherror = MathErrorListener(sympy)

        stream = antlr4.InputStream(sympy)
        lex = LaTeXLexer(stream)
        lex.removeErrorListeners()
        lex.addErrorListener(matherror)

        tokens = antlr4.CommonTokenStream(lex)
        parser = LaTeXParser(tokens)

        # remove default console error listener
        parser.removeErrorListeners()
        parser.addErrorListener(matherror)

        implication = parser.math().implication()
        expr = convert_implication(implication)


        expr = postprocessing(expr, sympy)
        if isinstance(expr, Exception):
            raise expr
    except (LaTeXParsingError, TypeError, NotAllowedSymbolException) as e:
        # try to use StringFormula as backup
        return StringFormula(sympy, recurse=parse_latex)

    return expr


def postprocessing(expr, text):
    PI_I = S.ImaginaryUnit * S.Pi
    # reasonability check if parsing was complete
    # unfortunately, sympy sometimes only parses the beginning without error
    latex_ = latex(expr)

    shortened_text = text.replace(r'\cdot', '').replace('\mathrm{d}', '').strip()
    shortened_tex = latex_.replace(r'\cdot', '').replace('\mathrm{d}', '').strip()

    if CHECK and len(shortened_tex) < len(shortened_text) / 2:
        raise LaTeXParsingError("Parsing seems to bee invalid due to too short string length <%s>, parsed <%s>, printed <%s>" % (text, expr, latex_))

    # apply safeguards to prevent mathematical constants are treated as variables
    try:
        i = sympy.Symbol('i')
        if i in expr:
            # check if this is the imaginary unit

            # check if i used as usual index
            if not any(s in text for s in ['i=1', 'i=0', 'i = 1', 'i = 0', '_i']):

                if PI_I in expr:
                    with evaluate(False):
                        expr = expr.subs(i, S.I)
                else:
                    # this is a candidate for the imaginary number, but this is still not for sure
                    with evaluate(False):
                        expr = expr.subs(i, sympy.core.numbers.ICandidate())

        e = sympy.Symbol('e')
        if e in expr:
            # _e indicates logarithm to the base e, while e^ indicates the exponential function (e.g., e^x), but e^2 is likely to be the variable e squared
            if '_e' in text or ('e^' in text and not 'e^2' in text):
                with evaluate(False):
                    expr = expr.subs(e, sympy.E)
    except Exception:
        pass


    return expr

def convert_implication(impl):
    if impl.formulas():
        return convert_formulas(impl.formulas())

    lh = convert_implication(impl.implication(0))
    rh = convert_implication(impl.implication(1))
    if impl.RIGHT_ARROW():
        return sympy.Implies(lh, rh, evaluate=False)
    elif impl.LEFT_RIGHT_ARROW():
        return sympy.Equivalent(lh, rh, evaluate=False)
    else:
        raise ValueError

def convert_formulas(formulas):
    relations = formulas.relation()
    rel = []
    for r in relations:
        rel.append(convert_relation(r))

    if len(relations) > 1:
        if formulas.TEXT_OR():
            return sympy.FormulasOr(*rel)
        return sympy.Formulas(*rel)
    return rel[0]

def convert_relation(rel):
    if rel.expr():
        return convert_expr(rel.expr())
    elif rel.set_():
        return convert_set(rel.set_())
    elif rel.variable():
        return convert_variable(rel.variable())

    lh = convert_relation(rel.relation(0))
    rh = convert_relation(rel.relation(1))
    if rel.LT():
        return sympy.StrictLessThan(lh, rh, evaluate=False)
    elif rel.LTE():
        return sympy.LessThan(lh, rh, evaluate=False)
    elif rel.GT():
        return sympy.StrictGreaterThan(lh, rh, evaluate=False)
    elif rel.GTE():
        return sympy.GreaterThan(lh, rh, evaluate=False)
    elif rel.EQUAL():
        if ':=' in rel.getText():
            return sympy.core.relational.Defines(lh, rh, evaluate=False)
        else:
            return sympy.Eq(lh, rh, evaluate=False)
    elif rel.DEFINES():
        return sympy.core.relational.Defines(lh, rh, evaluate=False)
    elif rel.NEQ():
        return sympy.Ne(lh, rh, evaluate=False)
    elif rel.APPROX():
        return sympy.Approx(lh, rh)
    elif rel.SUPSET():
        return sympy.sets.sets.SuperSet(lh, rh)
    elif rel.SUBSET():
        return sympy.sets.sets.SubSet(lh, rh)
    else:
        raise ValueError("Unknown relation %s" % rel)


def convert_set(expr):
    if expr.variable():
        variable = convert_variable(expr.variable())
        if isinstance(variable, sympy.sets.sets.VarIsInSet):
            base_set = variable.args[1]
            variable = variable.args[0]
        else:
            base_set = S.UniversalSet

        conditions = convert_formulas(expr.formulas())
        return sympy.ConditionSet(variable, conditions, base_set)
    elif expr.SPECIAL_SETS():
        special_set = convert_special_set(expr.SPECIAL_SETS())
        if expr.subexpr():
            subexpr = _convert_subexpr(expr.subexpr())
            special_set = sympy.SubExpr(special_set, subexpr)
        if expr.supexpr():
            supexpr = _convert_supexpr(expr.supexpr())
            special_set = sympy.SupExpr(special_set, supexpr)
        return special_set

    elif expr.atom():
        # interval like [a,b)
        l_interval = expr.l_interval()
        r_interval = expr.r_interval()
        if l_interval:
            left_open = (l_interval.L_PAREN() or l_interval.R_BRACKET()) is None
            right_open = (r_interval.R_PAREN() or r_interval.L_BRACKET()) is None
            left = convert_atom(expr.atom(0))
            right = convert_atom(expr.atom(1))
            if expr.SUB():
                # todo this is not quite right in any case? probably its enough due to restriction to atoms
                if len(expr.SUB()) == 1:
                    left = -left
                elif len(expr.SUB()) == 2:
                    left = -left
                    right = -right

            return sympy.Interval(left, right, left_open=left_open, right_open=right_open)
        else:
            # case like {1, 2, 3}
            atoms = [convert_atom(a) for a in expr.atom()]
            return sympy.FiniteSet(*atoms)
    else:
        print('Unexpected set expression')


special_set_map = {
    # reals
    r'\R': S.Reals,
    r'\RR': S.Reals,
    r'\mathbb{R}': S.Reals,
    r'\mathbf{R}': S.Reals,
    # naturals
    r'\N': S.Naturals,
    r'\mathbb{N}': S.Naturals,
    r'\mathbb N': S.Naturals,
    r'\mathbf{N}': S.Naturals,
    r'\N_0': S.Naturals0,
    # integers
    r'\Z': S.Integers,
    r'\mathbb{Z}': S.Integers,
    r'\mathbb Z': S.Integers,
    r'\mathbf{Z}': S.Integers,
    # complex numebrs
    r'\C': S.Complexes,
    r'\mathbb{C}': S.Complexes,
    r'\mathbb C': S.Complexes,
    r'\mathbf{C}': S.Complexes,
    # rational numbers
    r'\Q': S.Rationals,
    r'\mathbb{Q}': S.Rationals,
    r'\mathbb Q': S.Rationals,
    r'\mathbf{Q}': S.Rationals,
    r'\emptyset': sympy.EmptySet,
    r'\varnothing': sympy.EmptySet,
    r'∅': sympy.EmptySet,
    '\{\}': sympy.EmptySet,
    '\{ \}': sympy.EmptySet
}
def convert_special_set(text):
    if hasattr(text, 'EMPTY_SET') and text.EMPTY_SET():
        return sympy.EmptySet

    text = text.getText()

    if text in special_set_map:
        return special_set_map[text]

    return sympy.sets.sets.NamedSet(text)

def convert_variable(expr):
    if expr.LETTER():
        v = create_symbol(expr.LETTER().getText())
    elif expr.SYMBOL():
        v = create_symbol(expr.SYMBOL().getText())
    else:
        raise ValueError

    if expr.in_set():
        if expr.in_set().set_():
            return sympy.sets.sets.VarIsInSet(v, convert_set(expr.in_set().set_()))
        elif expr.in_set().LETTER():
            return sympy.sets.sets.VarIsInSet(v, sympy.Symbol(str(expr.in_set().LETTER())))
        else:
            raise ValueError

    return v


def create_symbol(symbol):
    if isinstance(symbol, str):
        name = symbol
    else:
        name = symbol.getText()

    if name.startswith('\\'):
        name = name.removeprefix('\\')
        if name in not_allowed_symbols:
            raise sympy.NotAllowedSymbolException("Found not allowed symbol %s" % name)

        if name not in known_symbols:
            raise sympy.NotAllowedSymbolException("Found not known symbol %s" % name)

    if name == 'pi':
        return sympy.pi

    if name in not_allowed_symbols:
        raise sympy.NotAllowedSymbolException("Found not allowed symbol %s" % name)

    return sympy.Symbol(name)

def get_symbol_text(expr):
    name = expr.getText()
    if name.startswith('\\'):
        name = name[1:]

    if name in not_allowed_symbols:
        raise sympy.NotAllowedSymbolException("Found not allowed symbol %s" % name)

    return name

def convert_expr(expr):
    if expr.NEGATE():
        return sympy.Not(expr.additive(), evaluate=False)

    return convert_add(expr.additive())


def convert_add(add):
    if add.ADD():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.Add(lh, rh, evaluate=False)
    elif add.SUB():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        if hasattr(rh, "is_Atom") and rh.is_Atom:
            return sympy.Add(lh, -1 * rh, evaluate=False)
        return sympy.Add(lh, sympy.Mul(-1, rh, evaluate=False), evaluate=False)
    elif add.PM():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.PlusMinus(lh, rh, evaluate=False)
    elif add.CMD_CUP():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.Union(lh, rh, evaluate=False)
    elif add.CMD_CAP():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.Intersection(lh, rh, evaluate=False)
    elif add.OR():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.Or(lh, rh, evaluate=False)
    else:
        return convert_mp(add.mp())

def _convert_derivative(wrt, power, f):
    wrt = convert_wrt(wrt)

    if isinstance(power, int):
        power = sympy.Number(power)
    if isinstance(power, Integer):
        if int(power) == 1:
            dd = sympy.Derivative(f, wrt)
        elif power > 0:
            dd = sympy.core.function.MultiDerivative(f, wrt, power)
        else:
            raise ValueError("negative power: %s" % power)
    else:
        dd = sympy.core.function.MultiDerivative(f, wrt, power)
    return dd

def convert_mp(mp, no_func=False):
    if hasattr(mp, 'mp'):
        mp_left = mp.mp(0)
        mp_right = mp.mp(1)
    else:
        mp_left = mp.mp_nofunc(0)
        mp_right = mp.mp_nofunc(1)
        no_func = True

    if mp.MUL() or mp.CMD_TIMES() or mp.CMD_CDOT():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return sympy.Mul(lh, rh, evaluate=False)
    elif mp.DIV() or mp.CMD_DIV() or mp.COLON():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)

        if mp.DIV():
            # check for d/dx or d^2/dx^2
            try:
                if isinstance(rh, Mul):
                    rrh = rh.args[0]
                    f = rh.args[1]
                    wrt, power = _convert_upper_lower(lh, rrh)
                    return _convert_derivative(wrt, power, f)
            except AttributeError:
                pass


        return sympy.Mul(lh, sympy.Pow(rh, -1, evaluate=False), evaluate=False)
    elif hasattr(mp, 'SET_MINUS') and mp.SET_MINUS():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return sympy.sets.sets.SetMinus(lh, rh)
    elif not no_func:
        #if mp.AND():
         #   lh = convert_mp(mp_left)
          #  rh = convert_mp(mp_right)
           # if str(lh) == '1' and str(rh) == '2':
            #    return sympy.core.symbol.Str('1,2')
            #return sympy.And(lh, rh, evaluate=False)
        return convert_operator(mp.operator())
    else:
        return convert_unary(mp.unary_nofunc())

def convert_operator(expr):
    if expr.operator():
        lhs = convert_operator(expr.operator(0))
        rhs = convert_operator(expr.operator(1))
        return sympy.BinaryOperator(sympy.core.Text(sympy.core.String(expr.BINARY_OPERATORS().getText())), lhs, rhs)
    else:
        if hasattr(expr, 'unary'):
            return convert_unary(expr.unary())
        else:
            return convert_unary(expr.unary_nofunc())


def convert_unary(unary):
    if hasattr(unary, 'unary'):
        nested_unary = unary.unary()
    else:
        nested_unary = unary.unary_nofunc()
    if hasattr(unary, 'postfix_nofunc'):
        first = unary.postfix()
        tail = unary.postfix_nofunc()
        postfix = [first] + tail
    else:
        postfix = unary.postfix()

    if unary.ADD():
        return convert_unary(nested_unary)
    elif unary.SUB():
        numabs = convert_unary(nested_unary)
        # Use Integer(-n) instead of Mul(-1, n)
        return -numabs
    elif hasattr(unary, 'CHOOSE') and unary.CHOOSE():
        if unary.LETTER():
            n = sympy.Symbol(unary.LETTER(0).getText())
            k = sympy.Symbol(unary.LETTER(1).getText())
        else:
            n = convert_expr(unary.expr(0))
            k = convert_expr(unary.expr(1))
        return sympy.binomial(n, k)
    elif postfix:
        def convert_start(start):
            if len(start) == 1:
                return start[0]
            else:
                return sympy.Mul(start[0], convert_start(start[1:]))

        start, derivatives = _split_fractions(postfix)
        if len(derivatives) == 0:
            return start
        else:
            if isinstance(start, list):
                return convert_start(derivatives)

            return sympy.Mul(start, convert_start(derivatives), evaluate=False)

def _convert_upper_lower(upper, lower):
    if isinstance(upper, Pow) and isinstance(lower, Pow):
        upper_first_arg = upper.args[0]
        lower_first_arg = lower.args[0]

        powers = [upper.args[1], lower.args[1]]
        if len(set(powers)) == 1:
            power = powers[0]
        else:
            # this is not part of a derivative
            raise AttributeError
    elif isinstance(upper, Symbol) and isinstance(lower, Symbol):
        upper_first_arg = upper
        lower_first_arg = lower

        power = 1
    else:
        raise AttributeError

    if isinstance(upper_first_arg, Symbol) and isinstance(lower_first_arg, Symbol) \
            and str(upper_first_arg) == 'd' and str(lower_first_arg).startswith('d'):
        # something like d/dx
        wrt = str(lower_first_arg).removeprefix('d').removeprefix(r'\text{d}').removeprefix(r'\mathrm{d}')
    else:
        raise AttributeError

    return wrt, power

split_cache = {}
def _split_fractions(postfixs):
    start = []
    derivatives = []
    current = start
    did_start = False
    for i, postfix in enumerate(postfixs):
        power = None
        try:
            exp = postfix.exp()
            if exp.CARET():
                if i == 0:
                    diff = exp.exp().comp().atom().DIFFERENTIAL()
                    wrt = diff.getText().removeprefix('d')
                    power = convert_atom(postfix.exp().atom())
                    current = [(power, wrt)]
                    did_start = True
                else:
                    raise AttributeError
            else:
                atom = exp.comp().atom()
                if atom.frac():

                    frac = atom.frac()
                    upper = convert_expr(frac.upper)
                    lower = convert_expr(frac.lower)
                    wrt, power = _convert_upper_lower(upper, lower)

                    if did_start:
                        derivatives.append(current)
                    else:
                        did_start = True
                    current = [(power, wrt)]
                elif atom.DIFFERENTIAL():
                    diff = atom.DIFFERENTIAL()
                    if i == 0:
                        wrt = diff.getText().removeprefix('d').removeprefix(r'\text{d}').removeprefix(r'\mathrm{d}')
                        power = 1
                        current = [(power, wrt)]
                        did_start = True
                    else:
                        text = atom.DIFFERENTIAL().getText()

                        if text.startswith('d'):
                            split = text.split(' ')
                            if len(split) == 2:
                                # this might be a multiplication like d*x
                                first = split[0]
                                second = split[1]
                                current.append(first)
                                current.append(second)
                            elif len(text) > 1:
                                first = text[0]
                                second = text[1:]
                                if second.startswith('\\') or len(second) == 1:
                                    current.append(first)
                                    current.append(second)
                            else:
                                raise AttributeError("Warning! Got unexpected dx! <%s>" % text)
                        else:
                            # this may be caused by a double integral (e.g. \int f(x, y) dx dy
                            raise AttributeError("Warning! Got unexpected dx! <%s>" % text)
                else:
                    raise AttributeError
        except AttributeError:
            current.append(postfix)

    if len(current) > 1:
        # check if last element was 'd' (as part of a d/dx which gets split up)
        c = convert_postfix_list([current[-1]])
        f = str(c)
        if (f == 'd' or f.startswith('d**')) and not isinstance(c, sympy.Symbol):
        # todo more advanced check
        # postfixs[4].parentCtx.parentCtx.parentCtx.mp(1).unary()
            current = current[:-1]
            if not did_start:
                start = start[:-1]


    if len(current) > 0 and did_start:
        derivatives.append(current)

    converted_derivatives = []
    for d in derivatives:
        # todo improve e.g. d/dx f(x) d/dy h(x, y) -> currently (d/dx f(x)) * (d/dy h(x, y))
        dd = d[1:]
        power = d[0][0]
        if isinstance(power, int):
            power = sympy.Number(power)
        wrt = convert_wrt(d[0][1])

        if len(dd) > 0:
            dd = convert_postfix_list(dd)
            if isinstance(power, Integer) or isinstance(power, int):
                if int(power) == 1:
                    dd = sympy.Derivative(dd, wrt, evaluate=False)
                else:
                    if power < 0:
                        dd = sympy.core.function.MultiDerivative(dd, wrt, -power)
                        dd = sympy.Pow(dd, -1)
                    else:
                        dd = sympy.core.function.MultiDerivative(dd, wrt, power)
            else:
                if str(power).startswith('-'):
                    dd = sympy.core.function.MultiDerivative(dd, wrt, -power)
                    dd = sympy.Pow(dd, -1)
                else:
                    dd = sympy.core.function.MultiDerivative(dd, wrt, power)
            converted_derivatives.append(dd)
        else:
            if power == 1:
                converted_derivatives.append(Symbol("d" + wrt))
            elif power > 0:
                converted_derivatives.append(Pow(Symbol("d" + wrt), power, evaluate=False))
            else:
                raise ValueError("Negative power : %s" % power)
    if len(start) > 0:
        start = convert_postfix_list(start)

    return start, converted_derivatives

def convert_number(number):
    s = number.getText().replace(",", "")
    return sympy.Number(s)


def convert_derivative_power(expr):
    if expr.SYMBOL():
        return create_symbol(expr.SYMBOL().getText())
    elif expr.LETTER():
        return create_symbol(expr.LETTER().getText())
    elif expr.number():
        return convert_number(expr.number())
    else:
        raise ValueError("Unexpected expression in derivative power!")

def convert_postfix_list(arr, i=0):
    if i >= len(arr):
        raise LaTeXParsingError("Index out of bounds")

    res = convert_postfix(arr[i])
    if isinstance(res, sympy.Expr):
        if i == len(arr) - 1:
            return res  # nothing to multiply by
        else:
            if i > 0:
                left = convert_postfix(arr[i - 1])
                right = convert_postfix(arr[i + 1])
                if isinstance(left, sympy.Expr) and isinstance(
                        right, sympy.Expr):
                    left_syms = convert_postfix(arr[i - 1]).atoms(sympy.Symbol)
                    right_syms = convert_postfix(arr[i + 1]).atoms(sympy.Symbol)

                    # if the left and right sides contain no variables and the
                    # symbol in between is 'x', treat as multiplication.
                    if not (left_syms or right_syms) and str(res) == 'x':
                        return convert_postfix_list(arr, i + 1)
            # multiply by next
            try:
                group = arr[i+1].exp().comp().group()
                if str(res) == 'E' and (group.L_BRACKET() or group.L_PAREN()):
                    res = Expectation(convert_expr(group.expr()))
                    i += 1
            except Exception:
                try:
                    exp = arr[i+1].exp()
                    caret = exp.CARET()
                    group = exp.exp().comp().group()
                    if caret and str(res) == 'E' and (group.L_BRACKET() or group.L_PAREN()):
                        pow = convert_atom(exp.atom())
                        exp = convert_exp(exp.exp())
                        res = sympy.Pow(Expectation(exp), pow)
                        i += 1
                except Exception:
                    pass
            if i + 1 < len(arr):
                rhs = convert_postfix_list(arr, i + 1)

                r = sympy.Mul(res, rhs, evaluate=False)

                # check if a lot of multiplications are nested out of single letter symbols -> this indicates text i.e. no valid formula
                if isinstance(res, Symbol) and isinstance(rhs, Mul) and isinstance(rhs.args[1], Mul) and all(isinstance(a, Symbol) for a in [rhs.args[0], rhs.args[1].args[0]]):
                    raise LaTeXParsingError("Got suspiciously many single letter symbols next to each other. This is probably text? <%s>" % r)

                return r
            else:
                return res
    elif isinstance(res, (sympy.Intersection, sympy.Union, sympy.Add, sympy.Or, sympy.Set, sympy.Tuple)):
        if i == len(arr) - 1:
            return res  # nothing to multiply by
        raise ValueError

    else:  # must be derivative
        wrt = convert_wrt(res[0])

        if i == len(arr) - 1:
            raise LaTeXParsingError("Expected expression for derivative")
        else:
            expr = convert_postfix_list(arr, i + 1)
            return sympy.Derivative(expr, wrt)

def convert_wrt(wrt):
    if not isinstance(wrt, sympy.Function) :
        if hasattr(wrt, 'free_symbols') and len(wrt.free_symbols) == 1 and wrt != wrt.free_symbols:
            wrt = list(wrt.free_symbols)[0]
    return wrt

def convert_group(expr):
    if expr.SINGLE_QUOTES():
        l = len(expr.SINGLE_QUOTES)
        exp = convert_expr(expr.expr())
        if l == 1:
            return sympy.Derivative(exp)
        else:
            raise NotImplementedError

    return MultiDerivative(convert_expr(expr.expr()), len(expr.SINGLE_QUOTES()))



def do_subs(expr, at):
    if at.expr():
        at_expr = convert_expr(at.expr())
        syms = at_expr.atoms(sympy.Symbol)
        if len(syms) == 0:
            return expr
        elif len(syms) > 0:
            sym = next(iter(syms))
            return expr.subs(sym, at_expr)
    elif at.equality():
        lh = convert_expr(at.equality().expr(0))
        rh = convert_expr(at.equality().expr(1))
        return expr.subs(lh, rh)


def convert_postfix(postfix):
    if isinstance(postfix, str):
        return sympy.Symbol(postfix)
    if hasattr(postfix, 'exp'):
        exp_nested = postfix.exp()
    else:
        exp_nested = postfix.exp_nofunc()

    exp = convert_exp(exp_nested)
    for op in postfix.postfix_op():
        if op.BANG():
            if isinstance(exp, list):
                raise LaTeXParsingError("Cannot apply postfix to derivative")
            exp = sympy.factorial(exp, evaluate=False)
        elif op.eval_at():
            ev = op.eval_at()
            at_b = None
            at_a = None
            if ev.eval_at_sup():
                at_b = do_subs(exp, ev.eval_at_sup())
            if ev.eval_at_sub():
                at_a = do_subs(exp, ev.eval_at_sub())
            if at_b is not None and at_a is not None:
                exp = sympy.Add(at_b, -1 * at_a, evaluate=False)
            elif at_b is not None:
                exp = at_b
            elif at_a is not None:
                exp = at_a

    return exp


def convert_exp(exp):
    if hasattr(exp, 'exp'):
        exp_nested = exp.exp()
    else:
        exp_nested = exp.exp_nofunc()

    if exp_nested:
        base = convert_exp(exp_nested)
        if isinstance(base, list):
            raise LaTeXParsingError("Cannot raise derivative to power")
        try: # this is very strange, but str(base) raises an exception for input (a+b\mathrm{i})^{-1}
            if str(base) == 'e':
                base = sympy.E
        except Exception as e:
            pass

        if exp.atom():
            exponent = convert_atom(exp.atom())
        elif exp.expr():
            exponent = convert_expr(exp.expr())
        else:
            raise ValueError

        if str(exponent) == 'complement':
            expr = sympy.sets.sets.SetComplement(base)
        if str(exponent) == 'c':
            if str(base).islower():
                expr = sympy.Pow(base, exponent, evaluate=False)
            else:
                expr = sympy.sets.sets.SetComplement(base)
        else:
            expr = sympy.Pow(base, exponent, evaluate=False)
    else:
        if hasattr(exp, 'comp'):
            expr = convert_comp(exp.comp())
        else:
            expr = convert_comp(exp.comp_nofunc())

    return expr

def convert_comp(comp):
    if comp.group():
        expr = convert_expr(comp.group().expr())
    elif comp.abs_group():
        e = convert_expr(comp.abs_group().expr())
        if isinstance(e, sympy.core.BasicMatrix):
            expr = sympy.Determinant(e, check=False)
        else:
            expr = sympy.Abs(e, evaluate=False)
    elif comp.atom():
        expr = convert_atom(comp.atom())
    elif comp.floor():
        expr = convert_floor(comp.floor())
    elif comp.ceil():
        expr = convert_ceil(comp.ceil())
    elif comp.func():
        expr = convert_func(comp.func())
    else:
        raise ValueError("Error")
    return expr

def convert_atom(atom):
    if atom.LETTER():
        sname = atom.LETTER().getText()
        if atom.subexpr():
            subscript = _convert_subexpr(atom.subexpr())
            base = IndexedBase(sname)
            return base[subscript]
        if atom.SINGLE_QUOTES():
            sname += atom.SINGLE_QUOTES().getText()  # put after subscript for easy identify
        return create_symbol(sname)
    elif atom.SYMBOL():
        s = get_symbol_text(atom.SYMBOL())
        if s == "infty":
            return sympy.oo
        else:
            if atom.subexpr():
                subscript = None
                if atom.subexpr().expr():  # subscript is expr
                    if len(atom.subexpr().expr()) == 1:
                        subscript = convert_expr(atom.subexpr().expr(0))
                    else:
                        subscript = sympy.Formulas(*[convert_expr(e) for e in atom.subexpr().expr()])
                elif atom.subexpr().variable():
                    subscript = convert_variable(atom.subexpr().variable())
                else:  # subscript is atom
                    subscript = convert_atom(atom.subexpr().atom())
                #subscriptName = StrPrinter().doprint(subscript)
                #s += '_{' + subscriptName + '}'
                base = IndexedBase(create_symbol(atom.SYMBOL()))
                return base[subscript]
            else:
                return create_symbol(atom.SYMBOL())
    elif atom.number():
        return convert_number(atom.number())
    elif atom.DIFFERENTIAL():
        var = get_differential_var(atom.DIFFERENTIAL())
        return create_symbol('d' + var.name)
    elif atom.DERIVATIVE_SYMBOLS():
        var = get_differential_var(atom.DERIVATIVE_SYMBOLS())
        return create_symbol('d' + var.name)
    elif atom.mathit():
        text = rule2text(atom.mathit().mathit_text())
        return create_symbol(text)
    elif atom.frac():
        return convert_frac(atom.frac())
    elif atom.binom():
        return convert_binom(atom.binom())
    elif atom.bars():
        val = convert_expr(atom.bars().expr())
        if isinstance(val, sympy.core.BasicMatrix):
            return sympy.Determinant(val, check=False)
        else:
            return sympy.functions.Norm(val)
    elif atom.env():
        env = atom.env()
        if env.NORM() or env.BAR():
            expr = convert_expr(env.expr())
            if env.subexpr():
                subexpr = _convert_subexpr(env.subexpr())
                return sympy.functions.Norm(expr, subexpr)
            else:
                return sympy.functions.Norm(expr)
        elif env.cases():
            return convert_cases(env.cases())
        mat = convert_matrix(env)
        return sympy.core.BasicMatrix(*mat)
    elif atom.GREEK():
        sname = atom.GREEK().getText()
        if atom.subexpr():
            subscript = _convert_subexpr(atom.subexpr())
            base = IndexedBase(sname)
            return base[subscript]
        if atom.SINGLE_QUOTES():
            sname += atom.SINGLE_QUOTES().getText()  # put after subscript for easy identify
        return create_symbol(sname)
    elif atom.modifier():
        return sympy.core.Modifier(sympy.core.symbol.Str(str(atom.modifier().MODIFIER())), convert_expr(atom.modifier().expr()))
    elif atom.INFINITY():
        return sympy.oo
    elif atom.DOTS():
        return sympy.core.numbers.Dots()
    elif atom.IMAGINARY_UNIT():
        return sympy.I
    elif atom.tuple_():
        content = [convert_expr(e) for e in atom.tuple_().expr()]
        return sympy.Tuple(*content)
    elif atom.getText():
        text = atom.getText()
        if len(text) > 0:
            return create_symbol(text)
    raise ValueError("Unknown atom %s" % atom) # probably .text()?

def convert_DDX(ddx):
    return ddx.getText()[-1]

def convert_cases(cases):
    cases_list = []
    for row in cases.cases_row():
        cases_list.append(convert_case(row))
    return sympy.Piecewise(*cases_list)

def convert_case(case):
    lhs = convert_expr(case.expr())
    if case.TEXT_OTHERWISE():
        rhs = sympy.Otherwise()
    else:
        rel = case.relation()
        rhs = convert_relation(rel)
    return lhs, rhs

def convert_matrix(matrix):
    mat = []
    if matrix.matrix():
        for row in matrix.matrix().matrix_row():
            mat.append(sympy.core.BasicVector(*[convert_expr(e) for e in row.expr()]))
    elif matrix.matrix_row():
        for row in matrix.matrix_row():
            mat.append(sympy.core.BasicVector(*[convert_expr(e) for e in row.expr()]))
    else:
        raise ValueError("Unknown Matrix type %s" % matrix)

    return mat

def rule2text(ctx):
    stream = ctx.start.getInputStream()
    # starting index of starting token
    startIdx = ctx.start.start
    # stopping index of stopping token
    stopIdx = ctx.stop.stop

    return stream.getText(startIdx, stopIdx)


def convert_frac(frac):
    if frac.FUNC_OVER():
        lhs = convert_expr(frac.expr(0))
        rhs = convert_expr(frac.expr(1))
        inverse_denom = sympy.Pow(rhs, -1, evaluate=False)
        return sympy.Mul(lhs, inverse_denom, evaluate=False)

    diff_op = False
    partial_op = False
    if frac.lower and frac.upper:
        lower_itv = frac.lower.getSourceInterval()
        lower_itv_len = lower_itv[1] - lower_itv[0] + 1
        if (frac.lower.start == frac.lower.stop
                and frac.lower.start.type == LaTeXLexer.DIFFERENTIAL):
            wrt = get_differential_var_str(frac.lower.start.text)
            diff_op = True
        elif (lower_itv_len == 2 and frac.lower.start.type == LaTeXLexer.SYMBOL
              and frac.lower.start.text == '\\partial'
              and (frac.lower.stop.type == LaTeXLexer.LETTER
                   or frac.lower.stop.type == LaTeXLexer.SYMBOL)):
            partial_op = True
            wrt = frac.lower.stop.text
            if frac.lower.stop.type == LaTeXLexer.SYMBOL:
                wrt = wrt[1:]

        if diff_op or partial_op:
            wrt = create_symbol(wrt)
            if (diff_op and frac.upper.start == frac.upper.stop
                    and frac.upper.start.type == LaTeXLexer.LETTER
                    and frac.upper.start.text == 'd'):
                return [wrt]
            elif (partial_op and frac.upper.start == frac.upper.stop
                  and frac.upper.start.type == LaTeXLexer.SYMBOL
                  and frac.upper.start.text == '\\partial'):
                return [wrt]
            upper_text = rule2text(frac.upper)

            expr_top = None
            if diff_op and upper_text.startswith('d'):
                expr_top = parse_latex(upper_text[1:])
            elif partial_op and frac.upper.start.text == '\\partial':
                expr_top = parse_latex(upper_text[len('\\partial'):])
            if expr_top:
                return sympy.Derivative(expr_top, wrt)
    if frac.upper:
        expr_top = convert_expr(frac.upper)
    else:
        expr_top = sympy.Number(frac.upperd.text)
    if frac.lower:
        expr_bot = convert_expr(frac.lower)
    elif frac.lowers:
        expr_bot = create_symbol(frac.lowers.text)
    else:
        expr_bot = sympy.Number(frac.lowerd.text)
    inverse_denom = sympy.Pow(expr_bot, -1, evaluate=False)
    if expr_top == 1:
        return inverse_denom
    else:
        return sympy.Mul(expr_top, inverse_denom, evaluate=False)

def convert_binom(binom):
    expr_n = convert_expr(binom.n)
    expr_k = convert_expr(binom.k)
    return sympy.binomial(expr_n, expr_k, evaluate=False)

def convert_floor(floor):
    val = convert_expr(floor.val)
    return sympy.floor(val, evaluate=False)

def convert_ceil(ceil):
    val = convert_expr(ceil.val)
    return sympy.ceiling(val, evaluate=False)

def convert_func(func):
    derivative = wrt = result = None

    if func.DDX():
        wrt = convert_DDX(func.DDX())
        derivative = 1

    if func.func_normal():
        if func.L_PAREN() or func.L_BRACE():  # function called with parenthesis
            arg = convert_func_arg(func.func_arg()[0])
        else:
            arg = convert_func_arg(func.func_arg_noparens())

        name = func.func_normal().start.text
        if name.startswith("\\"):
            name = name[1:]

        # change arc<trig> -> a<trig>
        if name in [
                "arcsin", "arccos", "arctan", "arccsc", "arcsec", "arccot"
        ]:
            name = "a" + name[3:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        elif name in ["arsinh", "arcosh", "artanh"]:
            name = "a" + name[2:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)
        elif name == "exp":
            expr = sympy.exp(arg, evaluate=False)
        elif name in ("log", "lg", "ln"):
            if func.subexpr():
                if func.subexpr().expr():
                    base = convert_expr(func.subexpr().expr(0))
                    if str(base) == 'e':
                        base = sympy.E
                else:
                    base = convert_atom(func.subexpr().atom())
            elif name == "lg":  # ISO 80000-2:2019
                base = 10
            elif name in ("ln", "log"):  # SymPy's latex printer prints ln as log by default
                base = sympy.E
            expr = sympy.log(arg, base, evaluate=False)
        elif func.func_normal().FUNC_RE():
            expr = sympy.re(arg)
        elif func.func_normal().FUNC_IM():
            expr = sympy.im(arg)

        func_pow = None
        should_pow = True
        if func.supexpr():
            if func.supexpr().expr():
                func_pow = convert_expr(func.supexpr().expr())
            else:
                func_pow = convert_atom(func.supexpr().atom())

        if name in [
                "sin", "cos", "tan", "csc", "sec", "cot", "sinh", "cosh",
                "tanh"
        ]:
            if func_pow == -1:
                name = "a" + name
                should_pow = False
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        if func_pow and should_pow:
            expr = sympy.Pow(expr, func_pow, evaluate=False)

        if "det" in name:
            expr = sympy.Determinant(arg, check=False)


        result = expr
    elif func.func_bracket():
        name = func.func_bracket().start.text.lower()

        if func.L_BRACKET() or func.L_PAREN():  # function called with parenthesis
            arg = convert_func_arg(func.func_arg()[0])
        else:
            arg = convert_func_arg(func.func_arg_noparens())

        if 'var' in name:
            return Variance(arg)
        elif name in ['e', r'\mathbb{e}', r'\operatorname{e}', r'\mathbb e']:
            return Expectation(arg)

        raise ValueError("can not match %s" % name)
    elif func.func_binary_bracket():
        name = func.func_binary_bracket().start.text.lower()
        args = [convert_func_arg(a) for a in func.func_arg()]
        if 'cov' in name:
            return Covariance(args[0], args[1])
        raise ValueError("can not match " + func)
    elif func.LETTER() or func.SYMBOL() or func.GREEK():
        if func.LETTER():
            fname = func.LETTER().getText()
        elif func.SYMBOL():
            fname = get_symbol_text(func.SYMBOL())
        elif func.GREEK():
            fname = func.GREEK().getText()
        fname = str(fname)  # can't be unicode
        lower = fname.lower()

        if func.subexpr():
            if func.subexpr().expr():  # subscript is expr
                subscript = convert_expr(func.subexpr().expr(0))
            else:  # subscript is atom
                subscript = convert_atom(func.subexpr().atom())
            subscriptName = StrPrinter().doprint(subscript)
            if len(subscriptName) == 1:
                fname += '_' + subscriptName
            else:
                fname += '_{' + subscriptName + '}'

        func_pow = None
        pow_paren = False
        if func.supexpr():
            if func.supexpr().expr():
                func_pow = convert_expr(func.supexpr().expr())
                try:
                    pow_paren = func.supexpr().expr().additive().mp().operator().unary().postfix(0).exp().comp().group().L_PAREN()
                except AttributeError:
                    pass
            else:
                func_pow = convert_atom(func.supexpr().atom())

        input_args = func.args()
        output_args = []
        while input_args.args():  # handle multiple arguments to function
            output_args.append(convert_expr(input_args.expr()))
            input_args = input_args.args()

        output_args.append(convert_expr(input_args.expr()))

        f = sympy.Function(fname)
        if func.SINGLE_QUOTES():
            derivative = len(func.SINGLE_QUOTES().getText())
            if len(output_args) == 1:
                wrt = convert_wrt(output_args[0])
                if not isinstance(wrt, sympy.Function) and hasattr(wrt, 'free_symbols') and len(wrt.free_symbols) > 1:
                    wrt = None
            else:
                pass
                #print("Warning: Got multiple output_args for SINGLE_QUOTES function: %s" % output_args)
        elif func.atom():
            a = convert_atom(func.atom())
            if isinstance(a, Integer):
                derivative = int(a)
            elif isinstance(a, Symbol):
                derivative = str(a)
            else:
                raise ValueError("Unknown type of derivative! %s" % a)
        result = f(*output_args)

        if lower in ['var', 'e', 'cov', 'det']:
            if lower == 'var':
                return Variance(*output_args)
            elif lower == 'cov':
                return Covariance(*output_args)
            elif lower == 'det':
                return sympy.Determinant(*output_args)
            elif 'E' in fname:
                return Expectation(*output_args)

        if func_pow == -1:
            result = sympy.core.function.InverseFunction(result)
        elif func_pow:
            if pow_paren:
                if len(output_args) == 1:
                    wrt = convert_wrt(output_args[0])
                    #if hasattr(wrt, 'free_symbols') and len(wrt.free_symbols):
                     #   wrt = list(wrt.free_symbols)[0]
                # case like f^{(n)}
                result = sympy.core.function.MultiDerivative(result, wrt, func_pow)
            else:
                # case like f^n, could be either f(f(...)) or (f(...))^n (2nd case more likely)
                if func_pow != 1:
                    result = sympy.Pow(result, func_pow)
    elif func.FUNC_INT():
        result = handle_integral(func)
    elif func.FUNC_SQRT():
        expr = convert_expr(func.base)
        if func.root:
            r = convert_expr(func.root)
            result = sympy.root(expr, r, evaluate=False)
        else:
            result = sympy.sqrt(expr, evaluate=False)
    elif func.FUNC_OVERLINE():
        expr = convert_expr(func.base)
        if isinstance(expr, sympy.Set) or str(expr).isupper():
            result = sympy.sets.sets.SetComplement(expr)
        else:
            result = sympy.conjugate(expr, evaluate=False)
    elif func.FUNC_SUM():
        result = handle_sum_or_prod(func, "summation")
    elif func.FUNC_PROD():
        result = handle_sum_or_prod(func, "product")
    elif func.FUNC_LIM():
        result = handle_limit(func, sympy.Limit)
    elif func.FUNC_MAX():
        result = handle_setfunction(func, sympy.sets.sets.SetMax)
    elif func.FUNC_MIN():
        result = handle_setfunction(func, sympy.sets.sets.SetMin)
    elif func.FUNC_SUP():
        result = handle_setfunction(func, sympy.sets.sets.SetSup)
    elif func.FUNC_INF():
        result = handle_setfunction(func, sympy.sets.sets.SetInf)
    elif func.probability():
        letters = func.probability().LETTER()
        if len(letters) == 2 and func.probability().EQUAL():
            if func.probability().subexpr():
                subexpr = _convert_subexpr(func.probability().subexpr())
                eq = sympy.Eq(create_symbol(letters[0]), IndexedBase(create_symbol(letters[1]), evaluate=False)[subexpr])
                return sympy.stats.BasicProbability(eq)

            return sympy.stats.BasicProbability(sympy.Eq(create_symbol(letters[0]), create_symbol(letters[1]), evaluate=False))
        elif func.probability().BAR():
            atoms = func.probability().atom()
            return sympy.stats.ConditionalProbability(convert_atom(atoms[0]), convert_atom(atoms[1]))
        elif len(letters) == 1:
            if func.probability().subexpr():
                name = "%s_%s" % (letters[0].getText(), _convert_subexpr(func.probability().subexpr()))
            else:
                name = letters[0].getText()
            return sympy.stats.BasicProbability(create_symbol(name))
        raise ValueError("Unexpected len of letters %d" % len(letters))
    elif func.group():
        result = convert_expr(func.group().expr())
        wrt = create_symbol('x')
        return Derivative(result, wrt, evaluate=False)
    else:
        raise ValueError("Unknown function!")

    if isinstance(result, Function) and derivative is not None:
        if derivative == 1:
            if wrt is None:
                return SimpleDerivative(result)
            else:
                return Derivative(result, wrt, evaluate=False)
        else:
            if isinstance(derivative, int):
                derivative = sympy.Number(derivative)
            return MultiDerivative(result, wrt, derivative)
    return result


def convert_func_arg(arg):
    if hasattr(arg, 'expr'):
        return convert_expr(arg.expr())
    else:
        return convert_mp(arg.mp_nofunc(), no_func=True)


def handle_integral(func):
    if func.additive():
        integrand = convert_add(func.additive())
    elif func.frac():
        integrand = convert_frac(func.frac())
    else:
        integrand = 1

    int_var = None
    if func.DIFFERENTIAL():
        int_var = get_differential_var(func.DIFFERENTIAL())
    else:
        for sym in integrand.atoms(sympy.Symbol):
            s = str(sym)
            if len(s) > 1 and s[0] == 'd':
                if s[1] == '\\':
                    int_var = sympy.Symbol(s[2:])
                else:
                    int_var = sympy.Symbol(s[1:])
                int_sym = sym
        if int_var:
            integrand = integrand.subs(int_sym, 1)
        else:
            # Assume dx by default
            int_var = sympy.Symbol('x')

    if func.subexpr():
        lower = _convert_subexpr(func.subexpr())
        if func.supexpr().atom():
            upper = convert_atom(func.supexpr().atom())
        else:
            upper = convert_expr(func.supexpr().expr())
        return sympy.Integral(integrand, (int_var, lower, upper))
    else:
        return sympy.Integral(integrand, int_var)

def _convert_supexpr(expr):
    if expr.expr():
        func_pow = convert_expr(expr.expr())
    else:
        func_pow = convert_atom(expr.atom())
    return func_pow

def _convert_subexpr(subexpr):
    if subexpr.atom():
        return convert_atom(subexpr.atom())
    else:
        if subexpr.expr():
            l = len(subexpr.expr())
            if l == 2:
                return sympy.Formulas(convert_expr(subexpr.expr(0)), convert_expr(subexpr.expr(1)))
            elif l == 1:
                return convert_expr(subexpr.expr()[0])
        elif subexpr.variable():
            return convert_variable(subexpr.variable())

    raise ValueError("Unexpected subexpression!")


def handle_sum_or_prod(func, name):
    val = convert_mp(func.mp())
    if func.subeq().atom():
        atom = convert_atom(func.subeq().atom())
        text = func.subeq().text()
        if text:
            t = text.getText()
            m = {
                "\\text{prime}": "\\text{ prime}"
            }
            t = m.get(t, t)
            limits = (atom, sympy.core.symbol.Str(t))
            return ExprWithOtherLimits(val, *limits)
        else:
            limits = (atom)
    else:
        iter_var = convert_expr(func.subeq().equality().expr(0))
        start = convert_expr(func.subeq().equality().expr(1))
        if func.supexpr():
            if func.supexpr().expr():  # ^{expr}
                end = convert_expr(func.supexpr().expr())
            else:  # ^atom
                end = convert_atom(func.supexpr().atom())

            limits = (iter_var, start, end)
        else:
            limits = (iter_var, start)
            return ExprWithOtherLimits(val, *limits)

    if len(limits) == 3:
        if name == "summation":
            return sympy.Sum(val, limits)
        elif name == "product":
            return sympy.Product(val, limits)
    else:
        if name == "summation":
            return sympy.Sum(val, *limits)
        elif name == "product":
            return sympy.Product(val, *limits)


def handle_limit(func, type):
    sub = func.limit_sub()
    if sub.LETTER():
        var = create_symbol(sub.LETTER().getText())
    elif sub.SYMBOL():
        var = create_symbol(sub.SYMBOL().getText())
    else:
        var = create_symbol('x')
    if sub.SUB():
        direction = "-"
    elif sub.ADD():
        direction = "+"
    else:
        direction = "+-"
    approaching = convert_expr(sub.expr())
    content = convert_mp(func.mp())

    return type(content, var, approaching, direction)

def handle_setfunction(func, type):
    if func.formulas():
        content = convert_formulas(func.formulas())
    else:
        content = convert_set(func.set())

    if func.subexpr():
        sub = _convert_subexpr(func.subexpr())
        return type(content, sub)

    return type(content)
def get_differential_var(d):
    text = get_differential_var_str(d.getText())
    return create_symbol(text)


def get_differential_var_str(text):
    for i in range(1, len(text)):
        c = text[i]
        if not (c == " " or c == "\r" or c == "\n" or c == "\t"):
            idx = i
            break
    text = text[idx:]
    text = text.replace('mathrm{d}', '').replace('text{d}', '')
    if len(text) > 0 and text[0] == "\\":
        text = text[1:]
    return text
