

"""
A Printer which converts an expression into its LaTeX equivalent.
"""
from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING

import itertools

import sympy
from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr, BasicMatrix
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative, UndefinedFunction, SimpleDerivative
from sympy.core.numbers import One
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse
from sympy.printing.stats import PermutationStats

from sympy.tensor.array import NDimArray

# sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE

from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str

from sympy.util import RandomDecidedChoice, RandomDecidedTruthValue, RandomChoice, RandomTruthValue
from sympy.utilities.iterables import has_variety, sift
from sympy.settings import randomize_settings

import re
import random

if TYPE_CHECKING:
    from sympy.vector.basisdependent import BasisDependent

def random_latex(expr, use_default_randomized_settings=True, **settings):
    if use_default_randomized_settings:
        settings.update(randomize_settings)
    return latex(expr, **settings)


# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at
# https://docs.mathjax.org/en/latest/tex.html#supported-latex-commands
# This variable only contains those functions which SymPy uses.
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                            'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec',
                            'csc', 'cot', 'coth', 're', 'im', 'frac', 'root',
                            'arg', 'lcm'
                            ]

tex_greek_dictionary = {
    'Alpha': r'\mathrm{A}',
    'Beta': r'\mathrm{B}',
    'Gamma': r'\Gamma',
    'Delta': r'\Delta',
    'Epsilon': r'\mathrm{E}',
    'Zeta': r'\mathrm{Z}',
    'Eta': r'\mathrm{H}',
    'Theta': r'\Theta',
    'Iota': r'\mathrm{I}',
    'Kappa': r'\mathrm{K}',
    'Lambda': r'\Lambda',
    'Mu': r'\mathrm{M}',
    'Nu': r'\mathrm{N}',
    'Xi': r'\Xi',
    'omicron': 'o',
    'Omicron': r'\mathrm{O}',
    'Pi': r'\Pi',
    'Rho': r'\mathrm{P}',
    'Sigma': r'\Sigma',
    'Tau': r'\mathrm{T}',
    'Upsilon': r'\Upsilon',
    'Phi': r'\Phi',
    'Chi': r'\mathrm{X}',
    'Psi': r'\Psi',
    'Omega': r'\Omega',
    'lamda': r'\lambda',
    'Lamda': r'\Lambda',
    'khi': r'\chi',
    'Khi': r'\mathrm{X}',
    'varepsilon': r'\varepsilon',
    'varkappa': r'\varkappa',
    'varphi': r'\varphi',
    'varpi': r'\varpi',
    'varrho': r'\varrho',
    'varsigma': r'\varsigma',
    'vartheta': r'\vartheta',
}

other_symbols = {'aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar',
                     'hslash', 'mho', 'wp'}

# Variable name modifiers
modifier_dict: dict[str, Callable[[str], str]] = {
    # Accents
    'mathring': lambda s: r'\mathring{'+s+r'}',
    'ddddot': lambda s: r'\ddddot{'+s+r'}',
    'dddot': lambda s: r'\dddot{'+s+r'}',
    'ddot': lambda s: r'\ddot{'+s+r'}',
    'dot': lambda s: r'\dot{'+s+r'}',
    'check': lambda s: r'\check{'+s+r'}',
    'breve': lambda s: r'\breve{'+s+r'}',
    'acute': lambda s: r'\acute{'+s+r'}',
    'grave': lambda s: r'\grave{'+s+r'}',
    'tilde': lambda s: r'\tilde{'+s+r'}',
    'hat': lambda s: r'\hat{'+s+r'}',
    'bar': lambda s: r'\bar{'+s+r'}',
    'vec': lambda s: r'\vec{'+s+r'}',
    'prime': lambda s: "{"+s+"}'",
    'prm': lambda s: "{"+s+"}'",
    # Faces
    'bold': lambda s: r'\boldsymbol{'+s+r'}',
    'bm': lambda s: r'\boldsymbol{'+s+r'}',
    'cal': lambda s: r'\mathcal{'+s+r'}',
    'scr': lambda s: r'\mathscr{'+s+r'}',
    'frak': lambda s: r'\mathfrak{'+s+r'}',
    # Brackets
    'norm': lambda s: r'\left\|{'+s+r'}\right\|',
    'avg': lambda s: r'\left\langle{'+s+r'}\right\rangle',
    'abs': lambda s: r'\left|{'+s+r'}\right|',
    'mag': lambda s: r'\left|{'+s+r'}\right|',
}

greek_letters_set = frozenset(greeks)

_between_two_numbers_p = (
    re.compile(r'[0-9][} ]*$'),  # search
    re.compile(r'[0-9]'),  # match
)


def latex_escape(s: str) -> str:
    """
    Escape a string such that latex interprets it as plaintext.

    We cannot use verbatim easily with mathjax, so escaping is easier.
    Rules from https://tex.stackexchange.com/a/34586/41112.
    """
    s = s.replace('\\', r'\textbackslash')
    for c in '&%$#_{}':
        s = s.replace(c, '\\' + c)
    s = s.replace('~', r'\textasciitilde')
    s = s.replace('^', r'\textasciicircum')
    return s

class LatexPrinter(Printer):
    printmethod = "_latex"

    _default_settings: dict[str, Any] = {
        "full_prec": False,
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "itex": False,
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_str": None,
        "mode": "plain",
        "mul_symbol": None,
        "mul_symbol_numbers": "dot",
        "pow2_as_mul": False,
        "pow3_as_mul": False,
        "pow3_as_pow2_mul": False,
        "symbol_names": {},
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "ri",
        "gothic_re_im": False,
        "decimal_separator": "period",
        "perm_cyclic": True,
        "parenthesize_super": True,
        "min": None,
        "max": None,
        "diff_operator": "d",
        "diff_force_wrt": False,
        "max_prime": 3,
        "wrt": "\\frac{%s}{%s}",
        "matrix": "pmatrix",
        "array_side": 'c',
        "plus_minus": "\\pm",
        "expected_value": "\\mathbb{E}",
        "variance": "\\mathbb{Var}",
        "covariance": "\\mathbb{Cov}",
        "probability": "\\mathbb{P}",
        "function_brackets": "%s[%s]",
        "exp_function": "e",
        "left_right_brackets": False,
        "order": "original",
        "return_stats": False,
        "optional_brackets": False,
        "blanks": True,
        "determinant": r"\left|%s\right|",
        "frac": r"\frac",
        "frac_short_mode": True,
        "set": "\{%s\}",
        "superset": r"\supset",
        "subset": r"\subset",
        "condition_set": "\left{%s|%s\}",
        "set_base_set_lhs": True,
        "formula_delimiter": ",",
        "math_set": "\mathbb{%s}",
        "cases_if": "for",
        "cases_else": "otherwise",
        "cases_force_else": True,
        "cases_order": "original",
        "dots": r"\cdots",
        "implies": r"\Rightarrow",
        "pi": r"\pi",
        "infinity": r"\infty",
        "approx": r"\approx",
        "emptyset": r"\emptyset",
        "set_complement": r"%s^\complement",
        "set_minus": r"\setminus",
        "interval_open": ('(', ')'),
        "interval": r"\left%s%s\right%s",
        "strip_.0": True,
        "space": "\;",
        "defines": r"\coloneqq",
        "choose": r'\binom{%s}{%s}',
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)
        self._init()


    def _init(self):
        if 'mode' in self._settings:
            valid_modes = ['inline', 'plain', 'equation', 'equation*']
            if self._settings['mode'] not in valid_modes:
                raise ValueError("'mode' must be one of 'inline', 'plain', "
                                 "'equation' or 'equation*'")

        if self._settings['fold_short_frac'] is None and \
                self._settings['mode'] == 'inline':
            self._settings['fold_short_frac'] = True

        mul_symbol_table = {
            None: r"",
            'blank': r" ",
            "ldot": r"\,.\, ",
            "dot": r"%s\cdot " % self._get_optional_blank(),
            "times": r"%s\times " % self._get_optional_blank(),
        }
        try:
            self._settings['mul_symbol_latex'] = \
                mul_symbol_table[self._settings['mul_symbol']]
        except KeyError:
            self._settings['mul_symbol_latex'] = \
                self._settings['mul_symbol']
        try:
            self._settings['mul_symbol_latex_numbers'] = \
                mul_symbol_table[self._settings['mul_symbol'] or 'dot']
        except KeyError:
            if (self._settings['mul_symbol'].strip() in
                    ['', ' ', '\\', '\\,', '\\:', '\\;', '\\quad']):
                self._settings['mul_symbol_latex_numbers'] = \
                    mul_symbol_table['dot']
            else:
                self._settings['mul_symbol_latex_numbers'] = \
                    self._settings['mul_symbol']

        if self._settings['mul_symbol_latex_numbers'] in ['', ' ']:
            try:
                self._settings['mul_symbol_latex_numbers'] = \
                    mul_symbol_table[self._settings['mul_symbol_numbers']]
            except KeyError:
                self._settings['mul_symbol_latex_numbers'] = \
                    self._settings['mul_symbol_numbers']
        self._delim_dict = {'(': ')', '[': ']'}

        imaginary_unit_table = {
            None: r"i",
            "i": r"i",
            "ri": r"\mathrm{i}",
            "ti": r"\text{i}",
            "j": r"j",
            "rj": r"\mathrm{j}",
            "tj": r"\text{j}",
        }
        imag_unit = self._settings['imaginary_unit']
        self._settings['imaginary_unit_latex'] = imaginary_unit_table.get(imag_unit, imag_unit)

        diff_operator_table = {
            None: r"d",
            "d": r"d",
            "rd": r"\mathrm{d}",
            "td": r"\text{d}",
        }
        diff_operator = self._settings['diff_operator']
        self._settings["diff_operator_latex"] = diff_operator_table.get(diff_operator, diff_operator)

        self.stats = PermutationStats()

    def get_stats(self):
        return self.stats

    def _get_left_right_parens(self):
        if self._settings['left_right_brackets']:
            left_brace = r'\left('
            right_brace = r'\right)'
        else:
            left_brace = '('
            right_brace = ')'
        return left_brace, right_brace

    def _add_parens(self, s) -> str:
        left_paren, right_paren = self._get_left_right_parens()
        return (left_paren + "{}" + right_paren).format(s)

    # TODO: merge this with the above, which requires a lot of test changes
    def _add_parens_lspace(self, s) -> str:
        left_paren, right_paren = self._get_left_right_parens()
        return (left_paren + " {}" + right_paren).format(s)

    def parenthesize(self, item, level, is_neg=False, strict=False, is_derivative=False) -> str:
        prec_val = precedence_traditional(item)
        if is_neg and strict:
            return self._add_parens(self._print(item))

        if (prec_val < level) or ((not strict) and prec_val <= level):
            return self._add_parens(self._print(item))
        else:
            if (self._get_setting('optional_brackets') or is_derivative) and (isinstance(item, Mul) or isinstance(item, Add)):
                return self._add_parens(self._print(item))

            return self._print(item)

    def parenthesize_super(self, s):
        """
        Protect superscripts in s

        If the parenthesize_super option is set, protect with parentheses, else
        wrap in braces.
        """
        if "^" in s:
            if self._settings['parenthesize_super']:
                return self._add_parens(s)
            else:
                return "{{{}}}".format(s)
        return s

    def _init_random_params(self):
        for param, entry in self._settings.items():
            if isinstance(entry, RandomDecidedChoice) or isinstance(entry, RandomDecidedTruthValue):
                entry.decide()

        self._init()

    def doprint(self, expr):
        self._init_random_params()

        try:
            if SimpleDerivative in expr:
                self._settings['diff_force_wrt'] = False
        except TypeError:
            pass

        tex = Printer.doprint(self, expr)
        stats = self.get_stats()
        return_stats = self._settings['return_stats']

        if self._settings['mode'] == 'plain':
            s = tex
        elif self._settings['mode'] == 'inline':
            s = r"$%s$" % tex
        elif self._settings['itex']:
            s = r"$$%s$$" % tex
        else:
            env_str = self._settings['mode']
            s = r"\begin{%s}%s\end{%s}" % (env_str, tex, env_str)
        return (s, stats) if return_stats else s

    def _needs_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        return not ((expr.is_Integer and expr.is_nonnegative)
                    or (expr.is_Atom and (expr is not S.NegativeOne
                                          and expr.is_Rational is False)))

    def _needs_function_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        passed as an argument to a function, False otherwise. This is a more
        liberal version of _needs_brackets, in that many expressions which need
        to be wrapped in brackets when added/subtracted/raised to a power do
        not need them when passed to a function. Such an example is a*b.
        """
        if not self._needs_brackets(expr):
            return False
        else:
            # Muls of the form a*b*c... can be folded
            if expr.is_Mul and not self._mul_is_clean(expr):
                return True
            # Pows which don't need brackets can be folded
            elif expr.is_Pow and not self._pow_is_clean(expr):
                return True
            # Add and Function always need brackets
            elif expr.is_Add or expr.is_Function:
                return True
            else:
                return False

    def _needs_mul_brackets(self, expr, first=False, last=False, div=False) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of a Mul, False otherwise. This is True for Add,
        but also for some container objects that would not need brackets
        when appearing last in a Mul, e.g. an Integral. ``last=True``
        specifies that this expr is the last to appear in a Mul.
        ``first=True`` specifies that this expr is the first to appear in
        a Mul.
        """
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral

        if expr.is_Mul:
            if not first and expr.could_extract_minus_sign():
                return True
        elif precedence_traditional(expr) < PRECEDENCE["Mul"]:
            return True
        elif expr.is_Relational:
            return True
        if expr.is_Piecewise:
            return True
        if any(expr.has(x) for x in (Mod,)):
            return True
        if (not last and
                any(expr.has(x) for x in (Integral, Product, Sum))):
            return True
        if div and len(expr.args) > 1:
            return True
        if isinstance(expr, sympy.PlusMinus):
            return True

        return False

    def _needs_add_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of an Add, False otherwise.  This is False for most
        things.
        """
        if expr.is_Relational:
            return True
        if any(expr.has(x) for x in (Mod,)):
            return True
        if expr.is_Add:
            return False # todo???
        return False

    def _mul_is_clean(self, expr) -> bool:
        for arg in expr.args:
            if arg.is_Function:
                return False
        return True

    def _pow_is_clean(self, expr) -> bool:
        return not self._needs_brackets(expr.base)

    def _do_exponent(self, expr: str, exp):
        if exp is not None:
            left_paren, right_paren = self._get_left_right_parens()
            return (left_paren + "%s" + right_paren + "^%s") % (expr, self._get_optional_curly_braces(exp, True))
        else:
            return expr

    def _print_Basic(self, expr):
        name = self._deal_with_super_sub(expr.__class__.__name__)
        if expr.args:
            ls = [self._print(o) for o in expr.args]
            s = r"\operatorname{{{}}}\left({}\right)"
            return s.format(name, ", ".join(ls))
        else:
            return r"\text{{{}}}".format(name)

    def _print_SubExpr(self, expr):
        if self._needs_curly_brackets(expr):
            sub = "{%s}" % self._print(expr.args[1])
        else:
            sub = self._print(expr.args[1])

        return "%s_%s" % (self._print(expr.args[0]), sub)

    def _print_SupExpr(self, expr):
        if self._needs_curly_brackets(expr):
            sup = "{%s}" % self._print(expr.args[1])
        else:
            sup = self._print(expr.args[1])

        return "%s^%s" % (self._print(expr.args[0]), sup)

    def _print_Formulas(self, expr):
        if all(x.is_number for x in expr.args):
            delimiter = ',' # special case like x_{1,2}
        else:
            delimiter = self._get_setting('formula_delimiter')
            if any(isinstance(a, (Symbol, Number)) for a in expr.args):
                delimiter = ','

            if delimiter == ',':
                delimiter += self._get_optional_blank()
            else:
                delimiter += ' '
        return delimiter.join([self._print(f) for f in expr.args])

    def _print_FormulasOr(self, expr):
        return r'\text{ or }'.join([self._print(f) for f in expr.args])

    def _print_LaTeXText(self, expr):
        return expr.getText()

    def _print_Text(self, expr):
        return expr.text

    def _print_BasicMatrix(self, expr):
        lines = []
        matrix_type = self._get_setting('matrix')
        delimiter = ',' if matrix_type == 'simple' else '&'

        for row in expr.args:
            lines.append((" %s " % delimiter).join([self._print(r) for r in row.args]))

        if matrix_type == 'array':
            out_str = r"\begin{%s}%s%s\end{%s}"
            sides = self._get_setting('array_side')
            n = len(expr.args[0])
            if sides == 'random':
                array_side = '{' + ''.join(random.choice('lcr') for _ in range(n)) + '}'
            else:
                array_side = '{' + (sides * n) + '}'
            return (out_str % (matrix_type, array_side, "\\\\".join(lines), matrix_type))
        elif matrix_type == 'pmatrix':
            out_str = r"\begin{%s}%s\end{%s}"
            return (out_str % (matrix_type, "\\\\".join(lines), matrix_type))

        return "[[" + "],[".join(lines) + "]]"

    def _print_PlusMinus(self, expr):
        lhs = expr.args[0]
        rhs = expr.args[1]
        return r"%s %s %s" % (self._print(lhs), self._get_setting('plus_minus'), self._print(rhs))

    def _print_bool(self, e: bool | BooleanTrue | BooleanFalse):
        return r"\text{%s}" % e

    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        return r"\text{%s}" % e

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)

        tex = ""
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif term.could_extract_minus_sign():
                tex += " - "
                term = -term
            else:
                tex += " + "
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = self._add_parens(term_tex)
            tex += term_tex

        return tex

    def _print_Cycle(self, expr):
        from sympy.combinatorics.permutations import Permutation
        if expr.size == 0:
            return r"\left( \right)"
        expr = Permutation(expr)
        expr_perm = expr.cyclic_form
        siz = expr.size
        if expr.array_form[-1] == siz - 1:
            expr_perm = expr_perm + [[siz - 1]]
        term_tex = ''
        for i in expr_perm:
            term_tex += str(i).replace(',', r"\;")
        term_tex = term_tex.replace('[', r"\left( ")
        term_tex = term_tex.replace(']', r"\right)")
        return term_tex

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation
        from sympy.utilities.exceptions import sympy_deprecation_warning

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=8,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            return self._print_Cycle(expr)

        if expr.size == 0:
            return r"\left( \right)"

        lower = [self._print(arg) for arg in expr.array_form]
        upper = [self._print(arg) for arg in range(len(lower))]

        row1 = " & ".join(upper)
        row2 = " & ".join(lower)
        mat = r" \\ ".join((row1, row2))
        return r"\begin{pmatrix} %s \end{pmatrix}" % mat


    def _print_AppliedPermutation(self, expr):
        perm, var = expr.args
        return r"\sigma_{%s}(%s)" % (self._print(perm), self._print(var))

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        strip = False if self._settings['full_prec'] else True
        low = self._settings["min"] if "min" in self._settings else None
        high = self._settings["max"] if "max" in self._settings else None
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)

        if str_real.endswith('.0') and self._get_setting('strip_.0'):
            str_real = str_real.removesuffix('.0')

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_latex_numbers']

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]
            if self._settings['decimal_separator'] == 'comma':
                mant = mant.replace('.','{,}')

            return r"%s%s10^%s" % (mant, separator, self._get_optional_curly_braces(exp, True))
        elif str_real == "+inf":
            return self._print_Infinity(expr)
        elif str_real == "-inf":
            return r"-" + self._print_Infinity(expr)
        else:
            if self._settings['decimal_separator'] == 'comma':
                str_real = str_real.replace('.','{,}')
            return str_real

    def _print_Cross(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \times %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Curl(self, expr):
        vec = expr._expr
        return r"\nabla\times %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Divergence(self, expr):
        vec = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Dot(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \cdot %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                 self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Gradient(self, expr):
        func = expr._expr
        return r"\nabla %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Laplacian(self, expr):
        func = expr._expr
        return r"\Delta %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _get_optional_blank(self):
        return " " if self._get_setting('blanks') else ""

    def _needs_original_order(self, expr):
        return any(c in expr for c in [sympy.core.numbers.Dots])

    def _print_Mul(self, expr: Expr):
        from sympy.simplify import fraction
        separator: str = str(self._settings['mul_symbol_latex'])
        numbersep: str = str(self._settings['mul_symbol_latex_numbers'])

        def convert(expr) -> str:
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                order = self.order
                if self._needs_original_order(expr):
                    order = 'original'

                if order == 'random':
                    args = list(expr.args)
                    random.shuffle(args)
                elif order == 'original':
                    args = list(expr.args)
                else:
                    args = expr.as_ordered_factors()


                return convert_args(args)




        def convert_args(args) -> str:
            _tex = last_term_tex = ""

            for i, term in enumerate(args):
                if term is S.One and not str(args[1]).startswith("2") and (len(_tex) > 0 or i + 1 < len(args)):
                    continue

                term_tex = self._print(term)

                if self._needs_mul_brackets(term, first=(i == 0),
                                            last=(i == len(args) - 1)):
                    term_tex = self._add_parens(term_tex)

                if  _between_two_numbers_p[0].search(last_term_tex) and \
                    _between_two_numbers_p[1].match(str(term)):
                    # between two numbers
                    _tex += numbersep
                elif term_tex[0].isdigit() and len(_tex) > 0:
                    # also use the number separator in this case (since this looks better: x*4 instead of x4)
                    _tex += numbersep
                elif _tex:
                    def ends_with_letter_or_special_pattern(string):
                        # tests if string ends with a case like x, x^k, x^{n+m}, which requires a non empty seperator
                        # if next term starts with brackets (otherwise x^k(t) would be interpreted as function x with (x(t))^k)!!
                        pattern = r"[a-zA-Z]$|[a-zA-Z]\^\{[a-zA-Z0-9+\-*/\\]*\}$|[a-zA-Z]\^[a-zA-Z0-9]$"
                        return bool(re.search(pattern, string))

                    if (term_tex.startswith('(') or term_tex.startswith(r'\left')) and ends_with_letter_or_special_pattern(last_term_tex):
                        if len(separator.strip()) == 0:
                            _tex += numbersep
                        else:
                            _tex += separator
                    elif len(separator) == 0 and len(_tex) > 1:
                        _tex += " "
                    else:
                        _tex += separator


                _tex += term_tex
                last_term_tex = term_tex
            return _tex

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        # XXX: _print_Pow calls this routine with instances of Pow...
        if isinstance(expr, Mul):
            args = expr.args
            if args[0] is S.One or any(isinstance(arg, Number) for arg in args[1:]):
                return convert_args(args)

        include_parens = False
        if expr.could_extract_minus_sign():
            expr = -expr
            tex = "-"
            if expr.is_Add:
                tex += "("
                include_parens = True
        else:
            tex = ""

        numer, denom = fraction(expr, exact=True)


        if denom is S.One and Pow(1, -1, evaluate=False) not in expr.args:
            # use the original expression here, since fraction() may have
            # altered it when producing numer and denom
            tex += convert(expr)
        else:
            try:
                # handle some special cases
                # using str(expr) might lead to an infinite recursion (for some reason), especially when S.I is involved
                snumer = str(numer)
                if len(snumer) > 5 and snumer.startswith('1') and set(snumer[1:]) == {'0'}:
                    numer = Pow(10, len(snumer) - 1, evaluate=False)

                sdenom = str(denom)
                if len(sdenom) > 5 and sdenom.startswith('1') and set(sdenom[1:]) == {'0'}:
                    denom = Pow(10, len(sdenom) - 1, evaluate=False)

                    if numer == 1:
                        return "10^{-%d}" % (len(sdenom) - 1)
            except Exception:
                pass

            if isinstance(denom, BasicMatrix):
                sdenom = convert(denom)
                snumer = convert(-numer)
                return "%s^{%s}" % (sdenom, snumer)
            else:
                snumer = convert(numer)
                sdenom = convert(denom)
                ldenom = len(sdenom.split())
                ratio = self._settings['long_frac_ratio']
                if self._settings['fold_short_frac'] and ldenom <= 2 and \
                        "^" not in sdenom:
                    # handle short fractions
                    if self._needs_mul_brackets(denom, div=True):
                        sdenom = self._add_parens(sdenom)

                    if self._needs_mul_brackets(numer, last=False):
                        tex += self._add_parens(snumer) + r"/%s" % (sdenom)
                    else:
                        tex += r"%s/%s" % (snumer, sdenom)
                elif ratio is not None and \
                        len(snumer.split()) > ratio*ldenom:
                    # handle long fractions
                    if self._needs_mul_brackets(numer, last=True):
                        tex += self._create_frac('1', sdenom) + separator + self._add_parens(snumer)
                    elif numer.is_Mul:
                        # split a long numerator
                        a = S.One
                        b = S.One
                        for x in numer.args:
                            if self._needs_mul_brackets(x, last=False) or \
                                    len(convert(a*x).split()) > ratio*ldenom or \
                                    (hasattr(b, 'is_commutative') and hasattr(x, 'is_commutative') and b.is_commutative is x.is_commutative is False):
                                b *= x
                            else:
                                a *= x
                        if self._needs_mul_brackets(b, last=True):
                            tex += self._create_frac(convert(a), sdenom) + separator + self._add_parens(convert(b))
                        else:
                            tex += self._create_frac(convert(a), sdenom) + (r"%s%s" % (separator, convert(b)))
                    elif snumer == '1':
                        tex += self._create_frac('1', sdenom)
                    else:
                        tex += self._create_frac('1', sdenom) + ("%s%s" % (separator, snumer))
                else:
                    tex += self._create_frac(snumer, sdenom)

        if include_parens:
            tex += ")"
        return tex

    def _create_frac(self, sdenom: str, snumer: str):
        if self._get_setting('frac_short_mode') and len(sdenom) == 1 and sdenom.isnumeric() and len(snumer) == 1:
            return "%s%s%s" % (self._get_frac(), sdenom, snumer)
        else:
            return "%s{%s}{%s}" % (self._get_frac(), sdenom, snumer)

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_PrimeIdeal(self, expr):
        p = self._print(expr.p)
        if expr.is_inert:
            return rf'\left({p}\right)'
        alpha = self._print(expr.alpha.as_expr())
        return rf'\left({p}, {alpha}\right)'

    def _print_Pow(self, expr: Pow):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational:
            p: int = expr.exp.p  # type: ignore
            q: int = expr.exp.q  # type: ignore
            if abs(p) == 1 and q != 1 and self._settings['root_notation']:
                base = self._print(expr.base)
                if q == 2:
                    tex = r"\sqrt{%s}" % base
                elif self._settings['itex']:
                    tex = r"\root{%d}{%s}" % (q, base)
                else:
                    tex = r"\sqrt[%d]{%s}" % (q, base)
                if expr.exp.is_negative:
                    return r"%s{1}{%s}" % (self._get_frac(), tex)
                else:
                    return tex
            elif self._settings['fold_frac_powers'] and q != 1:
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                # issue #12886: add parentheses for superscripts raised to powers
                if expr.base.is_Symbol:
                    base = self.parenthesize_super(base)
                if expr.base.is_Function:
                    return self._print(expr.base, exp="%s/%s" % (p, q))
                return r"%s^{%s/%s}" % (base, p, q)
            elif expr.exp.is_negative and hasattr(expr.base, 'is_commutative') and expr.base.is_commutative:
                # special case for 1^(-x), issue 9216
                if expr.base == 1 or expr.base == 10:
                    return r"%s^%s" % (expr.base, self._get_optional_curly_braces(expr.exp, True))
                # special case for (1/x)^(-y) and (-1/-x)^(-y), issue 20252
                if expr.base.is_Rational:
                    base_p: int = expr.base.p  # type: ignore
                    base_q: int = expr.base.q  # type: ignore
                    if base_p * base_q == abs(base_q):
                        if expr.exp == -1:
                            return r"%s{1}{%s{%s}{%s}}" % (self._get_frac(), self._get_frac(), base_p, base_q)
                        else:
                            return r"%s{1}{(%s{%s}{%s})^%s}" % (self._get_frac(), self._get_frac(), base_p, base_q, self._get_optional_curly_braces(abs(expr.exp)))
                # things like 1/x
                return self._print_Mul(expr)
        if expr.base.is_Function:
            return self._print(expr.base, exp=self._print(expr.exp))

        seperator = self._get_setting('mul_symbol_latex')
        if len(seperator) == 0:
            seperator = self._get_setting('mul_symbol_latex_numbers')

        if expr.exp == 2 and self._get_setting('pow2_as_mul'):
            base = self.parenthesize(expr.base, PRECEDENCE['Mul'])
            return "%s %s %s" % (base, seperator, base)
        elif expr.exp == 3 and self._get_setting('pow3_as_pow2_mul'):
            base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
            lhs = "%s^2" % base
            rhs = self.parenthesize(expr.base, PRECEDENCE['Mul'])
            if random.random() < 0.5:
                lhs, rhs = rhs, lhs
            return "%s %s %s" % (lhs, seperator, rhs)
        elif expr.exp == 3 and self._get_setting('pow3_as_mul'):
            base = self.parenthesize(expr.base, PRECEDENCE['Mul'])
            return "%s %s %s %s %s" % (base, seperator, base, seperator, base)

        tex = r"%s^" + self._get_optional_curly_braces(expr.exp)
        return self._helper_print_standard_power(expr, tex)

    def _helper_print_standard_power(self, expr, template: str) -> str:
        exp = self._print(expr.exp)
        # issue #12886: add parentheses around superscripts raised
        # to powers
        base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
        if expr.base.is_Symbol:
            base = self.parenthesize_super(base)
        elif (isinstance(expr.base, Derivative)
            and base.startswith(r'\left(')
            and re.match(r'\\left\(\\d?d?dot', base)
            and base.endswith(r'\right)')):
            # don't use parentheses around dotted derivative
            base = base[6: -7]  # remove outermost added parens
        return template % (base, exp)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_ExprWithOtherLimits(self, expr):
        if len(expr.args) == 2:
            tex = r'\sum_{%s}' % self._print(expr.args[1])
        elif len(expr.args) == 3:
            if isinstance(expr.args[1], sympy.core.symbol.Str):
                tex = r'\sum_{%s %s}' % (self._print(expr.args[1]), self._print(expr.args[2]))
            else:
                tex = r'\sum_{%s=%s}' % (self._print(expr.args[1]), self._print(expr.args[2]))
        else:
            raise ValueError

        if isinstance(expr.args[0], Add):
            tex += self._add_parens(self._print(expr.args[0]))
        else:
            tex += self._print(expr.args[0])
        return tex

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            t = tuple([self._print(i) for i in expr.limits[0]])
            suffix = self._get_optional_curly_braces(t[-1])
            tex = (r"\sum_{%s=%s}^" + suffix + " ") % t

        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        if isinstance(expr.function, Add):
            tex += self._add_parens(self._print(expr.function))
        else:
            tex += self._print(expr.function)

        return tex

    def _print_Product(self, expr):
        if isinstance(expr.limits[0], sympy.Symbol):
            if len(expr.limits) == 1:
                tex = r"\prod_{%s}" % self._print(expr.limits[0])
            else:
                tex = r"\prod_{%s %s}" % (self._print(expr.limits[0]), self._print(expr.limits[1]))
        elif len(expr.limits) == 1:
                t = tuple([self._print(i) for i in expr.limits[0]])
                suffix = self._get_optional_curly_braces(t[-1])
                tex = (r"\prod_{%s=%s}^" + suffix + " ") % t
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\prod_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        if isinstance(expr.function, Add):
            tex += self._add_parens(self._print(expr.function))
        else:
            tex += self._print(expr.function)

        return tex

    def _print_BasisDependent(self, expr: 'BasisDependent'):
        from sympy.vector import Vector

        o1: list[str] = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key=lambda x: x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    arg_str = r'\left(' + self._print(v) + r'\right)'
                    o1.append(' + ' + arg_str + k._latex_form)

        outstr = (''.join(o1))
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr

    def _needs_curly_brackets(self, tex):
        if not isinstance(tex, str):
            tex = str(tex)

        if len(tex) > 1:
            if re.match(r'^\\[a-zA-Z]+$', tex):
                return False # e.g. \\pi

            if re.match(r'^\\[a-zA-Z]+\{([\w+\-*/])+\}$', tex):
                return False # e.g. \\hat{x}

            if re.match(r'^\\[a-zA-Z]+\{([\w+\-*/])+\}+\{([\w+\-*/])+\}$', tex):
                return False

            return True

        return False

    def _get_optional_curly_braces(self, tex: str, fill=False):
        s = '{%s}' if self._needs_curly_brackets(tex) else '%s'
        if fill:
            return s % tex
        return s

    def _print_oo(self, expr):
        return self._print_Infinity(expr)

    def _print_Infinity(self, expr):
        return self._get_setting('infinity')

    def _print_NegativeInfinity(self, expr):
        return '-' + self._print_Infinity(expr)

    def _print_Indexed(self, expr):
        tex_base = self._print(expr.base)

        suffix = '_{%s}'
        if len(expr.indices) == 1:
            first = self._print(expr.indices[0])
            if not self._needs_curly_brackets(first):
                suffix = '_%s'

        if self._needs_curly_brackets(tex_base):
            tex = '{'+tex_base+'}'+suffix % ','.join(map(self._print, expr.indices))
        else:
            tex = tex_base+suffix % ','.join(map(self._print, expr.indices))
        return tex

    def _print_IndexedBase(self, expr):
        return self._print(expr.label)

    def _print_Idx(self, expr):
        label = self._print(expr.label)
        if expr.upper is not None:
            upper = self._print(expr.upper)
            if expr.lower is not None:
                lower = self._print(expr.lower)
            else:
                lower = self._print(S.Zero)
            interval = '{lower}\\mathrel{{..}}\\nobreak {upper}'.format(
                    lower = lower, upper = upper)
            return '{{{label}}}_{{{interval}}}'.format(
                label = label, interval = interval)
        #if no bounds are defined this just prints the label
        return label

    def _print_Derivative(self, expr):
        # test if ' notation can be used
        function = expr.args[0]
        func_name = function.func.__name__
        func_name_usable = isinstance(function.func, sympy.Function)

        if len(function.args) == 0:
            if func_name_usable:
                # no argument is given, so use shortcut notation
                return func_name + "'"
            else:
                pow = expr.args[1].args[1]
                diff = self._get_setting('diff_operator_latex')
                if pow == 1:
                    return r"\frac{%s%s}{%s%s}" % (diff, self._print(expr.args[0]), diff, self._print(expr.args[1].args[0]))
                else:
                    return r"\frac{%s%s^%s}{%s%s^%s}" % (diff, self._print(expr.args[0]), self._print(pow), diff, self._print(expr.args[1].args[0]), self._print(pow))

        function_argument = function.args[0]
        wrt = expr.args[1][0]
        args = ",".join([self._print(a) for a in function.args])
        if func_name_usable and isinstance(function, Function) and (str(function_argument) == str(wrt)):
            return "%s'(%s)" % (func_name, args)
        elif func_name_usable and not self._get_setting('diff_force_wrt'):
            # if the function is nested (e.g. f(g(x)) we might use [f(g(x))]'
            if isinstance(expr.args[0].args[0], UndefinedFunction) and len(expr.free_symbols) <= 1:
                return "[%s(%s)]'" % (func_name, args)

        if requires_partial(expr.expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = self._settings["diff_operator_latex"]

        tex = ""
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += int(num)
            if num == 1:
                if diff_symbol == 'd' or diff_symbol.endswith('}'):
                    tex += r"%s%s" % (diff_symbol, self._print(x))
                else:
                    tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                n = self._print(num)
                if diff_symbol == 'd':
                    tex += (r"%s%s^" + self._get_optional_curly_braces(n)) % (diff_symbol,
                                        self.parenthesize_super(self._print(x)), n)
                else:
                    tex += (r"%s %s^" + self._get_optional_curly_braces(n)) % (diff_symbol,
                                                                              self.parenthesize_super(self._print(x)),n)

        wrt = self._get_setting('wrt')
        if wrt == '%s/%s' and (diff_symbol != 'd' or tex[-1] not in ['x', 'y', 'z']):
            wrt = r"\frac{%s}{%s}"
        if dim == 1:
            tex = wrt % (diff_symbol, tex)
        else:
            d = self._print(dim)
            tex = wrt % ("%s^%s" % (diff_symbol, d), tex)

        if any(i.could_extract_minus_sign() for i in expr.args):
            return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=True,
                                                  strict=True,
                                                  is_derivative=True))

        return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=False,
                                                  strict=True,
                                                  is_derivative=True))

    def _print_SimpleDerivative(self, expr):
        fnc_name = expr.args[0].name
        params = ", ".join([self._print(a) for a in expr.args[0].args])
        return "%s'(%s)" % (fnc_name, params)

    def _print_MultiDerivative(self, expr):
        fnc = self._print(expr.args[0])
        fnc_name = expr.args[0].name
        wrt = expr.args[1]
        pow = expr.args[2]
        wrt_pattern = self._get_setting('wrt')
        if not self._get_setting('diff_force_wrt') or wrt_pattern == '%s/%s':
            # try to write it like f''(x) or f^{(n)}(x)
            tex = None
            if isinstance(pow, sympy.Symbol):
                tex = '%s^{(%s)}%s' % (fnc_name, self._print(pow), fnc.removeprefix(fnc_name))
            elif isinstance(pow, sympy.Integer):
                pow_int = int(pow)
                if pow_int <= self._get_setting('max_prime'):
                    tex = fnc_name + (pow_int * "'") + fnc.removeprefix(fnc_name)
                else:
                    tex = '%s^{(%s)}%s' % (fnc_name, self._print(pow), fnc.removeprefix(fnc_name))

            if tex:
                return tex

        d = self._get_setting('diff_operator_latex')
        wrt = self._print(wrt)
        if wrt_pattern == '%s/%s' and (d != 'd' or wrt not in ['x', 'y', 'z']):
            wrt_pattern = r"\frac{%s}{%s}"
        if (isinstance(pow, sympy.Integer) or isinstance(pow, int)) and int(pow) == 1:
            return (wrt_pattern % (d, d + wrt)) + fnc
        else:
            suffix = "^" + self._print(pow)
            return (wrt_pattern % (d + suffix, d + wrt + suffix)) + fnc

    def _print_Subs(self, subs):
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr,
                                                         latex_subs)

    def _print_Integral(self, expr):
        tex, symbols = "", []
        diff_symbol = self._settings["diff_operator_latex"]

        # Only up to \iiiint exists
        if len(expr.limits) <= 4 and all(len(lim) == 1 for lim in expr.limits):
            # Use len(expr.limits)-1 so that syntax highlighters don't think
            # \" is an escaped quote
            tex = r"\i" + "i"*(len(expr.limits) - 1) + "nt"
            symbols = [r"\,%s%s" % (diff_symbol, self._print(symbol[0]))
                       for symbol in expr.limits]

        else:
            for lim in reversed(expr.limits):
                symbol = lim[0]
                tex += r"\int"

                if len(lim) > 1:
                    if self._settings['mode'] != 'inline' \
                            and not self._settings['itex']:
                        tex += r"\limits"

                    if len(lim) == 3:
                        lower = self._print(lim[1])
                        upper = self._print(lim[2])

                        lower_template = self._get_optional_curly_braces(lower)
                        upper_template = self._get_optional_curly_braces(upper)

                        tex += ("_" + lower_template + "^" + upper_template) % (lower, upper)
                    if len(lim) == 2:
                        upper = self._print(lim[1])
                        upper_template = self._get_optional_curly_braces(upper)
                        tex += "^" + upper_template % (upper)

                symbols.insert(0, r"\,%s%s" % (diff_symbol, self._print(symbol)))

        is_neg = any(i.could_extract_minus_sign() for i in expr.args)
        content = self.parenthesize(expr.function, PRECEDENCE["Mul"], is_neg=is_neg, strict=True)

        if tex[-1].isdigit() and content[0].isdigit():
            return r"%s {%s}%s" % (tex, content, "".join(symbols))
        return r"%s %s%s" % (tex, content, "".join(symbols))

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args

        tex = r"\lim_{%s \to " % self._print(z)
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), self._print(dir))

        if isinstance(e, AssocOp) and not isinstance(e, Mul): # todo actually this should depend on the context
            return tex + self._add_parens(self._print(e))
        else:
            return r"%s %s" % (tex, self._print(e))

    def _hprint_Function(self, func: str) -> str:
        r'''
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, excluding sub- and superscripts, just use that letter
          - if it is a longer name, then put \operatorname{} around it and be
            mindful of undercores in the name
        '''
        func = self._deal_with_super_sub(func)
        superscriptidx = func.find("^")
        subscriptidx = func.find("_")
        if func in accepted_latex_functions:
            name = r"\%s" % func
        elif len(func) == 1 or func.startswith('\\') or subscriptidx == 1 or superscriptidx == 1:
            name = func
        else:
            if superscriptidx > 0 and subscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:min(subscriptidx,superscriptidx)],
                    func[min(subscriptidx,superscriptidx):])
            elif superscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:superscriptidx],
                    func[superscriptidx:])
            elif subscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:subscriptidx],
                    func[subscriptidx:])
            else:
                if len(func.replace("'", "")) == 1:
                    name = func
                else:
                    name = r"\operatorname{%s}" % func
        return name

    def _print_Modifier(self, expr: sympy.core.Modifier):
        return "%s{%s}" % (self._print(expr.args[0]), self._print(expr.args[1]))

    def _update_stats(self, key, value):
        if key in self.stats['tex']:
            if value in self.stats['tex'][key]:
                self.stats['tex'][key][value] += 1
            else:
                self.stats['tex'][key][value] = 1
        else:
            self.stats['tex'][key] = {value: 1}

    def _get_setting(self, setting, update_stats=False):
        if isinstance(self._settings[setting], (str, bool, int, tuple)) or self._settings[setting] is None:
            s = self._settings[setting]
            if update_stats:
                self._update_stats(setting, s)
            return s
        elif isinstance(self._settings[setting], RandomChoice) or isinstance(self._settings[setting], RandomTruthValue):
            s = self._settings[setting]._random_state()
            if update_stats:
                self._update_stats(setting, s)
            return s

        raise ValueError("Unknown setting %s of type %s" % (setting, type(self._settings[setting])))

    def _get_frac(self, update_stats=False):
        return self._get_setting('frac', update_stats=update_stats)

    def _get_derivative_function(self, expr):
        derivative = expr.derivative
        wrt = str(expr.wrt)
        name = expr.func.__name__
        if wrt in [None, 'None'] or not self._get_setting('diff_force_wrt'):
            if isinstance(derivative, int) and 0 < derivative < 4:
                return name + ("'" * derivative)
            elif derivative == 0:
                return name
            return name + "^{(%s)}" % expr.derivative

        # todo

        operator = self._get_setting("diff_operator_latex", update_stats=True)
        suffix = ("^%s" % derivative if derivative > 0 else "")
        a = operator + suffix
        b = operator + wrt + suffix
        return (self._get_setting('wrt', update_stats=True) % (a, b)) + name

    def _print_InverseFunction(self, expr, exp=None) -> str:
        return self._print_Function_(expr.args[0], exp='-1')


    def _print_Function(self, expr: Function, exp=None) -> str:
        r'''
        Render functions to LaTeX, handling functions that LaTeX knows about
        e.g., sin, cos, ... by using the proper LaTeX command (\sin, \cos, ...).
        For single-letter function names, render them as regular LaTeX math
        symbols. For multi-letter function names that LaTeX does not know
        about, (e.g., Li, sech) use \operatorname{} so that the function name
        is rendered in Roman font and LaTeX handles spacing properly.

        expr is the expression involving the function
        exp is an exponent
        '''
        func = expr.func.__name__ #todo self._get_derivative_function(expr)
        if hasattr(self, '_print_' + func) and \
                not isinstance(expr, AppliedUndef):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            return self._print_Function_(expr, exp)
    def _print_Function_(self, expr, exp) -> str:
        func = expr.func.__name__ #todo self._get_derivative_function(expr)
        args = [str(self._print(arg)) for arg in expr.args]
        # How inverse trig functions should be displayed, formats are:
        # abbreviated: asin, full: arcsin, power: sin^-1
        inv_trig_style = self._get_setting('inv_trig_style')
        # If we are dealing with a power-style inverse trig function
        inv_trig_power_case = False
        # If it is applicable to fold the argument brackets
        can_fold_brackets = self._get_setting('fold_func_brackets') and \
            len(args) == 1 and \
            not self._needs_function_brackets(expr.args[0])

        inv_trig_table = [
            "asin", "acos", "atan",
            "acsc", "asec", "acot",
            "asinh", "acosh", "atanh",
            "acsch", "asech", "acoth",
        ]

        # If the function is an inverse trig function, handle the style
        if func in inv_trig_table:
            if inv_trig_style == "abbreviated":
                pass
            elif inv_trig_style == "full":
                func = ("ar" if func[-1] == "h" else "arc") + func[1:]
            elif inv_trig_style == "power":
                func = func[1:]
                inv_trig_power_case = True

                # Can never fold brackets if we're raised to a power
                if exp is not None:
                    can_fold_brackets = False

        if inv_trig_power_case:
            if func in accepted_latex_functions:
                name = r"\%s^{-1}" % func
            else:
                if len(func.replace("'", '')) == 1:
                    name = '%s^{-1}' % func
                else:
                    name = r"\operatorname{%s}^{-1}" % func
        elif exp is not None:
            func_tex = self._hprint_Function(func)
            func_tex = self.parenthesize_super(func_tex)
            suffix = self._get_optional_curly_braces(exp, fill=True)
            name = (r'%s^' + suffix) % (func_tex)
        else:
            name = self._hprint_Function(func)

        if can_fold_brackets and func in accepted_latex_functions:
            # Wrap argument safely to avoid parse-time conflicts
            # with the function name itself
            name += r"{%s}"
        else:
            if self._get_setting('left_right_brackets', update_stats=True):
                name += r"\left(%s\right)"
            else:
                name += r"(%s)"

        if inv_trig_power_case and exp is not None:
            suffix = self._get_optional_curly_braces(exp, True)
            name += r"^" + suffix

        return name % ",".join(args)

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    def _print_ElementwiseApplyFunction(self, expr):
        return r"{%s}_{\circ}\left({%s}\right)" % (
            self._print(expr.function),
            self._print(expr.expr),
        )

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: r'\delta',
                gamma:  r'\Gamma',
                lowergamma: r'\gamma',
                beta: r'\operatorname{B}',
                DiracDelta: r'\delta',
                Chi: r'\operatorname{Chi}'}

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                return self._special_function_classes[cls]
        return self._hprint_Function(str(expr))

    def _print_Lambda(self, expr):
        symbols, expr = expr.args

        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(tuple(symbols))

        tex = r"\left( %s \mapsto %s \right)" % (symbols, self._print(expr))

        return tex

    def _print_IdentityFunction(self, expr):
        return r"\left( x \mapsto x \right)"

    def _hprint_variadic_function(self, expr, exp=None) -> str:
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\%s\left(%s\right)" % (str(expr.func).lower(),
                                       ", ".join(texargs))
        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return r"%s^" + suffix % (tex)
        else:
            return tex

    _print_Min = _print_Max = _hprint_variadic_function

    def _print_floor(self, expr, exp=None):
        tex = r"\left\lfloor{%s}\right\rfloor" % self._print(expr.args[0])

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix) % (tex)
        else:
            return tex

    def _print_ceiling(self, expr, exp=None):
        tex = r"\left\lceil{%s}\right\rceil" % self._print(expr.args[0])

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix) % (tex)
        else:
            return tex

    def _print_log(self, expr, exp=None):
        if self._get_setting('left_right_brackets', update_stats=True):
            left_brace = r'\left('
            right_brace = r'\right)'
        else:
            left_brace = '('
            right_brace = ')'

        sub_expr = ""
        is_e = False
        if len(expr.args) > 1:
            sub = expr.args[1]
            sub_expr = "_" + self._print(sub)
            is_e = sub == sympy.E or str(sub) == 'e'

        if is_e and self._get_setting("ln_notation", update_stats=True):
            tex = r"\ln" + left_brace + self._print(expr.args[0]) + right_brace
        else:
            tex = r"\log" + sub_expr + left_brace + self._print(expr.args[0]) + right_brace

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix) % (tex)
        else:
            return tex

    def _print_Abs(self, expr, exp=None):
        tex = r"|%s|" % self._print(expr.args[0])

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix) % (tex)
        else:
            return tex

    def _print_re(self, expr, exp=None):
        if self._get_setting('gothic_re_im', update_stats=True):
            tex = r"\Re{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{re}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        return self._do_exponent(tex, exp)

    def _print_im(self, expr, exp=None):
        if self._get_setting('gothic_re_im', update_stats=True):
            tex = r"\Im{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{im}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        return self._do_exponent(tex, exp)

    def _print_ICandidate(self, expr):
        return 'i' # since it is only a candidate, always print raw 'i' instead of other versions like \text{i}

    def _print_Not(self, e):
        from sympy.logic.boolalg import (Equivalent, Implies)
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg \left(%s\right)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])

    def _print_Norm(self, expr):
        if len(expr.args) == 1:
            return r"\|%s\|" % self._print(expr.args[0])
        return "\|%s\|_%s" % (self._print(expr.args[0]), self._get_optional_curly_braces(self._print(expr.args[1]), fill=True))

    def _print_LogOp(self, args, char):
        arg = args[0]
        if arg.is_Boolean and not arg.is_Not:
            tex = r"\left(%s\right)" % self._print(arg)
        else:
            tex = r"%s" % self._print(arg)

        for arg in args[1:]:
            if arg.is_Boolean and not arg.is_Not:
                tex += r" %s \left(%s\right)" % (char, self._print(arg))
            else:
                tex += r" %s %s" % (char, self._print(arg))

        return tex

    def _print_And(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\wedge")

    def _print_Or(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\vee")

    def _print_Xor(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\veebar")

    def _print_Implies(self, e, altchar=None):
        return self._print_LogOp(e.args, altchar or self._get_setting('implies', update_stats=True))

    def _print_Equivalent(self, e, altchar=None):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, altchar or r"\Leftrightarrow")

    def _print_conjugate(self, expr, exp=None):
        tex = r"\overline{%s}" % self._print(expr.args[0])

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix) % (tex)
        else:
            return tex

    def _print_polar_lift(self, expr, exp=None):
        func = r"\operatorname{polar\_lift}"
        arg = r"{\left(%s \right)}" % self._print(expr.args[0])

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix + "%s") % (func, arg)
        else:
            return r"%s%s" % (func, arg)

    def _print_ExpBase(self, expr, exp=None):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        if "e" == self._get_setting('exp_function', update_stats=True) or expr.args[0] in [S.One, None]:
            tex = r"e^" + self._get_optional_curly_braces(self._print(expr.args[0]), True)
            return self._do_exponent(tex, exp)

        # exp case
        return "exp(%s)" % self._print(expr.args[0])


    def _print_Exp1(self, expr, exp=None):
        return "e"

    def _print_elliptic_k(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"K^{%s}%s" % (exp, tex)
        else:
            return r"K%s" % tex

    def _print_elliptic_f(self, expr, exp=None):
        tex = r"\left(%s\middle| %s\right)" % \
            (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"F^{%s}%s" % (exp, tex)
        else:
            return r"F%s" % tex

    def _print_elliptic_e(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"E^{%s}%s" % (exp, tex)
        else:
            return r"E%s" % tex

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]),
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex

    def _print_beta(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    def _print_betainc(self, expr, exp=None, operator='B'):
        largs = [self._print(arg) for arg in expr.args]
        tex = r"\left(%s, %s\right)" % (largs[0], largs[1])

        if exp is not None:
            return r"\operatorname{%s}_{(%s, %s)}^{%s}%s" % (operator, largs[2], largs[3], exp, tex)
        else:
            return r"\operatorname{%s}_{(%s, %s)}%s" % (operator, largs[2], largs[3], tex)

    def _print_betainc_regularized(self, expr, exp=None):
        return self._print_betainc(expr, exp, operator='I')

    def _print_uppergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

    def _print_lowergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\gamma^{%s}%s" % (exp, tex)
        else:
            return r"\gamma%s" % tex

    def _hprint_one_arg_func(self, expr, exp=None) -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (self._print(expr.func), exp, tex)
        else:
            return r"%s%s" % (self._print(expr.func), tex)

    _print_gamma = _hprint_one_arg_func

    def _print_Chi(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{Chi}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{Chi}%s" % tex

    def _print_expint(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[1])
        nu = self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{E}_{%s}^{%s}%s" % (nu, exp, tex)
        else:
            return r"\operatorname{E}_{%s}%s" % (nu, tex)

    def _print_fresnels(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"S^{%s}%s" % (exp, tex)
        else:
            return r"S%s" % tex

    def _print_fresnelc(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"C^{%s}%s" % (exp, tex)
        else:
            return r"C%s" % tex

    def _print_subfactorial(self, expr, exp=None):
        tex = r"!%s" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"\left(%s\right)^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix) % (tex)
        else:
            return tex

    def _print_factorial2(self, expr, exp=None):
        tex = r"%s!!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix) % (tex)
        else:
            return tex

    def _print_BinaryOperator(self, expr):
        return "%s %s %s" % (self._print(expr.args[1]), expr.args[0], self._print(expr.args[2]))

    def _print_binomial(self, expr, exp=None):
        tex = self._get_setting('choose') % (self._print(expr.args[0]),
                                     self._print(expr.args[1]))

        if exp is not None:
            suffix = self._get_optional_curly_braces(exp, fill=True)
            return (r"%s^" + suffix) % (tex)
        else:
            return tex

    def _print_RisingFactorial(self, expr, exp=None):
        n, k = expr.args
        base = r"%s" % self.parenthesize(n, PRECEDENCE['Func'])

        tex = r"{%s}^{\left(%s\right)}" % (base, self._print(k))

        return self._do_exponent(tex, exp)

    def _print_FallingFactorial(self, expr, exp=None):
        n, k = expr.args
        sub = r"%s" % self.parenthesize(k, PRECEDENCE['Func'])

        tex = r"{\left(%s\right)}_{%s}" % (self._print(n), sub)

        return self._do_exponent(tex, exp)

    def _hprint_BesselBase(self, expr, exp, sym: str) -> str:
        tex = r"%s" % (sym)

        need_exp = False
        if exp is not None:
            if tex.find('^') == -1:
                tex = r"%s^{%s}" % (tex, exp)
            else:
                need_exp = True

        tex = r"%s_{%s}\left(%s\right)" % (tex, self._print(expr.order),
                                           self._print(expr.argument))

        if need_exp:
            tex = self._do_exponent(tex, exp)
        return tex

    def _hprint_vec(self, vec) -> str:
        if not vec:
            return ""
        s = ""
        for i in vec[:-1]:
            s += "%s, " % self._print(i)
        s += self._print(vec[-1])
        return s

    def _print_besselj(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'J')

    def _print_besseli(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'I')

    def _print_besselk(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'K')

    def _print_bessely(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'Y')

    def _print_yn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'y')

    def _print_jn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'j')

    def _print_hankel1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(1)}')

    def _print_hankel2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(2)}')

    def _print_hn1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(1)}')

    def _print_hn2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(2)}')

    def _hprint_airy(self, expr, exp=None, notation="") -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s%s" % (notation, tex)

    def _hprint_airy_prime(self, expr, exp=None, notation="") -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"{%s^\prime}^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s^\prime%s" % (notation, tex)

    def _print_airyai(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Ai')

    def _print_airybi(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Bi')

    def _print_airyaiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Ai')

    def _print_airybiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Bi')

    def _print_hyper(self, expr, exp=None):
        tex = r"{{}_{%s}F_{%s}\left(\begin{matrix} %s \\ %s \end{matrix}" \
              r"\middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._hprint_vec(expr.ap), self._hprint_vec(expr.bq),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, exp)
        return tex

    def _print_meijerg(self, expr, exp=None):
        tex = r"{G_{%s, %s}^{%s, %s}\left(\begin{matrix} %s & %s \\" \
              r"%s & %s \end{matrix} \middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._print(len(expr.bm)), self._print(len(expr.an)),
              self._hprint_vec(expr.an), self._hprint_vec(expr.aother),
              self._hprint_vec(expr.bm), self._hprint_vec(expr.bother),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, exp)
        return tex

    def _print_dirichlet_eta(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\eta^{%s}%s" % (exp, tex)
        return r"\eta%s" % tex

    def _print_zeta(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s, %s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\zeta^{%s}%s" % (exp, tex)
        return r"\zeta%s" % tex

    def _print_stieltjes(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_{%s}\left(%s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"_{%s}" % self._print(expr.args[0])
        if exp is not None:
            return r"\gamma%s^{%s}" % (tex, exp)
        return r"\gamma%s" % tex

    def _print_lerchphi(self, expr, exp=None):
        tex = r"\left(%s, %s, %s\right)" % tuple(map(self._print, expr.args))
        if exp is None:
            return r"\Phi%s" % tex
        return r"\Phi^{%s}%s" % (exp, tex)

    def _print_polylog(self, expr, exp=None):
        s, z = map(self._print, expr.args)
        tex = r"\left(%s\right)" % z
        if exp is None:
            return r"\operatorname{Li}_{%s}%s" % (s, tex)
        return r"\operatorname{Li}_{%s}^{%s}%s" % (s, exp, tex)

    def _print_jacobi(self, expr, exp=None):
        n, a, b, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s,%s\right)}\left(%s\right)" % (n, a, b, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_gegenbauer(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"C_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_chebyshevt(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"T_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_chebyshevu(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"U_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_legendre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"P_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_assoc_legendre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_hermite(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"H_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_laguerre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"L_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_assoc_laguerre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"L_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_Ynm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Y_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_Znm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Z_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def __print_mathieu_functions(self, character, args, prime=False, exp=None):
        a, q, z = map(self._print, args)
        sup = r"^{\prime}" if prime else ""
        exp = "" if not exp else "^{%s}" % exp
        return r"%s%s\left(%s, %s, %s\right)%s" % (character, sup, a, q, z, exp)

    def _print_mathieuc(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, exp=exp)

    def _print_mathieus(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, exp=exp)

    def _print_mathieucprime(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, prime=True, exp=exp)

    def _print_mathieusprime(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, prime=True, exp=exp)

    def _print_Rational(self, expr):
        if expr.q != 1:
            sign = ""
            p = expr.p
            if expr.p < 0:
                sign = "- "
                p = -p
            if self._get_setting('fold_short_frac', update_stats=True):
                return r"%s%d / %d" % (sign, p, expr.q)
            return r"%s%s{%d}{%d}" % (sign, self._get_frac() ,p, expr.q)
        else:
            return self._print(expr.p)

    def _print_Order(self, expr):
        s = self._print(expr.expr)
        if expr.point and any(p != S.Zero for p in expr.point) or \
           len(expr.variables) > 1:
            s += '; '
            if len(expr.variables) > 1:
                s += self._print(expr.variables)
            elif expr.variables:
                s += self._print(expr.variables[0])
            s += r'\rightarrow '
            if len(expr.point) > 1:
                s += self._print(expr.point)
            else:
                s += self._print(expr.point[0])
        return r"O\left(%s\right)" % s

    def _print_Symbol(self, expr: Symbol, style='plain'):
        name: str = self._settings['symbol_names'].get(expr) or self._settings['symbol_names'].get(str(expr))
        if name is not None:
            if isinstance(name, RandomChoice):
                name = name._random_state()
            return name

        return self._deal_with_super_sub(expr.name, style=style)

    _print_RandomSymbol = _print_Symbol

    def _deal_with_super_sub(self, string: str, style='plain') -> str:
        if '{' in string:
            name, supers, subs = string, [], []
        else:
            name, supers, subs = split_super_sub(string)

            name = translate(name)
            supers = [translate(sup) for sup in supers]
            subs = [translate(sub) for sub in subs]

        # apply the style only to the name
        if style == 'bold':
            name = "\\mathbf{{{}}}".format(name)

        # glue all items together:
        if supers:
            name += "^" + self._get_optional_curly_braces(" ".join(supers), True)
        if subs:
            name += "_" + self._get_optional_curly_braces(" ".join(subs), True)

        return name

    def _print_Defines(self, expr):
        return "%s %s %s" % (self._print(expr.args[0]), self._get_setting('defines'), self._print(expr.args[1]))

    def _print_Relational(self, expr):
        if self._get_setting('itex', update_stats=True):
            gt = r"\gt"
            lt = r"\lt"
        else:
            gt = ">"
            lt = "<"

        charmap = {
            "==": "=",
            ">": gt,
            "<": lt,
            ">=": r"\geq",
            "<=": r"\leq",
            "!=": r"\neq",
            "~": r"\approx",
        }

        inverse_operator = {
            '=': '=',
            r'\approx': r'\approx',
            r'\neq': r'\neq',
            gt: lt,
            lt: gt,
            r'\geq': r'\leq',
            r'\leq': r'\geq',
        }


        allow_random = not isinstance(expr.rhs, (sympy.Piecewise, sympy.core.relational.Relational)) and not isinstance(expr.lhs, sympy.core.relational.Relational)
        old_order = self._settings['order']
        if not allow_random:
            self._settings['order'] = 'original'
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        symbol = charmap[expr.rel_op]
        self._settings['order'] = old_order

        if random.random() > 0.5 and allow_random and self._get_setting('order', update_stats=True) == 'random':
            lhs, rhs = rhs, lhs
            # symmetric operator = and ~ do not need any changes, other operators needs to be inverted
            symbol = inverse_operator[symbol]

        return "%s %s %s" % (lhs, symbol, rhs)

    def _invert_relational(self, rel):
        if isinstance(rel, sympy.Equality):
            op = sympy.Ne
        elif isinstance(rel, sympy.Ne):
            op = sympy.Equality
        elif isinstance(rel, sympy.LessThan):
            op = sympy.StrictGreaterThan
        elif isinstance(rel, sympy.GreaterThan):
            op = sympy.StrictLessThan
        elif isinstance(rel, sympy.StrictLessThan):
            op = sympy.GreaterThan
        elif isinstance(rel, sympy.StrictGreaterThan):
            op = sympy.LessThan
        else:
            raise ValueError("Unexpected type %s" % type(rel))


        return op(*rel.args)

    def _print_Piecewise(self, expr):
        cases_if = self._get_optional_blank() + self._get_setting('cases_if') + self._get_optional_blank()
        space = self._get_setting('space')
        ecpairs = [r"%s & \text{%s}%s %s" % (self._print(e), cases_if, space, self._print(c))
                   for e, c in expr.args[:-1]]
        if expr.args[-1].cond == true:
            if not self._get_setting('cases_force_else') and len(expr.args) == 2 and expr.args[0].args[1].is_Relational:
                # todo
                ecpairs.append(r"%s & %s" % (self._print(expr.args[-1].expr), self._print(self._invert_relational(expr.args[0].args[1]))))
                if self._get_setting('cases_order') == 'random':
                    random.shuffle(ecpairs)
            else:
                ecpairs.append(r"%s & \text{%s}" %
                               (self._print(expr.args[-1].expr), self._get_setting('cases_else')))
        else:
            ecpairs.append(r"%s & \text{%s}%s %s" %
                           (self._print(expr.args[-1].expr),
                            cases_if,
                            space,
                            self._print(expr.args[-1].cond)))
            if self._get_setting('cases_order') == 'random':
                random.shuffle(ecpairs)
        tex = r"\begin{cases} %s \end{cases}"
        return tex % r" \\".join(ecpairs)

    def _print_matrix_contents(self, expr):
        lines = []

        if isinstance(expr, sympy.core.matrix.BasicMatrix):
            for row in expr.args:
                lines.append(" & ".join([self._print(i) for i in row.args]))
        else:
            for line in range(expr.rows):  # horrible, should be 'rows'
                lines.append(" & ".join([self._print(i) for i in expr[line, :]]))

        mat_str = self._get_setting('mat_str', update_stats=True)
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if isinstance(expr, sympy.core.matrix.BasicMatrix) or expr.cols <= 10:
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'

        out_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        out_str = out_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            out_str = out_str.replace('%s', '{' + 'c'*expr.cols + '}%s')
        return out_str % r"\\".join(lines)

    def _print_MatrixBase(self, expr):
        out_str = self._print_matrix_contents(expr)
        if self._settings['mat_delim']:
            left_delim: str = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            out_str = r'\left' + left_delim + out_str + \
                      r'\right' + right_delim
        return out_str

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True)\
            + '_{%s, %s}' % (self._print(expr.i), self._print(expr.j))

    def _print_MatrixSlice(self, expr):
        def latexslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = None
            if x[1] == dim:
                x[1] = None
            return ':'.join(self._print(xi) if xi is not None else '' for xi in x)
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + r'\left[' +
                latexslice(expr.rowslice, expr.parent.rows) + ', ' +
                latexslice(expr.colslice, expr.parent.cols) + r'\right]')

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and hasattr(mat, 'is_MatrixExpr') and  mat.is_MatrixExpr):
            return r"\left(%s\right)^{T}" % self._print(mat)
        else:
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return r"\left(%s\right)^{T}" % s
            else:
                return "%s^{T}" % s

    def _print_Trace(self, expr):
        mat = expr.arg
        return r"\operatorname{tr}\left(%s \right)" % self._print(mat)

    def _print_Adjoint(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and hasattr(mat, 'is_MatrixExpr') and mat.is_MatrixExpr):
            return r"\left(%s\right)^{\dagger}" % self._print(mat)
        else:
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return r"\left(%s\right)^{\dagger}" % s
            else:
                return r"%s^{\dagger}" % s

    def _print_MatMul(self, expr):
        from sympy import MatMul

        # Parenthesize nested MatMul but not other types of Mul objects:
        parens = lambda x: self._print(x) if isinstance(x, Mul) and not isinstance(x, MatMul) else \
            self.parenthesize(x, precedence_traditional(expr), False)

        args = list(expr.args)
        if expr.could_extract_minus_sign():
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            return '- ' + ' '.join(map(parens, args))
        else:
            return ' '.join(map(parens, args))

    def _print_Pi(self, expr):
        return self._get_setting('pi', update_stats=True)

    def _print_Determinant(self, expr):
        pattern = self._get_setting('determinant', update_stats=True)
        mat = expr.arg
        if hasattr(mat, 'is_MatrixExpr') and mat.is_MatrixExpr:
            from sympy.matrices.expressions.blockmatrix import BlockMatrix
            if isinstance(mat, BlockMatrix):
                return pattern % self._print_matrix_contents(mat.blocks)
            return pattern % self._print(mat)
        return pattern % self._print(mat)


    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return r'\left(%s \bmod %s\right)^{%s}' % \
                (self.parenthesize(expr.args[0], PRECEDENCE['Mul'],
                                   strict=True),
                 self.parenthesize(expr.args[1], PRECEDENCE['Mul'],
                                   strict=True),
                 exp)
        return r'%s \bmod %s' % (self.parenthesize(expr.args[0],
                                                   PRECEDENCE['Mul'],
                                                   strict=True),
                                 self.parenthesize(expr.args[1],
                                                   PRECEDENCE['Mul'],
                                                   strict=True))

    def _print_HadamardProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize

        return r' \circ '.join(
            map(lambda arg: parens(arg, prec, strict=True), args))

    def _print_HadamardPower(self, expr):
        if precedence_traditional(expr.exp) < PRECEDENCE["Mul"]:
            template = r"%s^{\circ \left({%s}\right)}"
        else:
            template = r"%s^{\circ {%s}}"
        return self._helper_print_standard_power(expr, template)

    def _print_KroneckerProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize

        return r' \otimes '.join(
            map(lambda arg: parens(arg, prec, strict=True), args))

    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol) and hasattr(base, 'is_MatrixExpr') and  base.is_MatrixExpr:
            return "\\left(%s\\right)^{%s}" % (self._print(base),
                                              self._print(exp))
        else:
            base_str = self._print(base)
            if '^' in base_str:
                return r"\left(%s\right)^{%s}" % (base_str, self._print(exp))
            else:
                return "%s^{%s}" % (base_str, self._print(exp))

    def _print_MatrixSymbol(self, expr):
        return self._print_Symbol(expr, style=self._settings[
            'mat_symbol_style'])

    def _print_ZeroMatrix(self, Z):
        return "0" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{0}"

    def _print_OneMatrix(self, O):
        return "1" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{1}"

    def _print_Identity(self, I):
        return r"\mathbb{I}" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{I}"

    def _print_PermutationMatrix(self, P):
        perm_str = self._print(P.args[0])
        return "P_{%s}" % perm_str

    def _print_NDimArray(self, expr: NDimArray):

        if expr.rank() == 0:
            return self._print(expr[()])

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.rank() == 0) or (expr.shape[-1] <= 10):
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'
        block_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        block_str = block_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            block_str= block_str.replace('%s','{}%s')
        if self._settings['mat_delim']:
            left_delim: str = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            block_str = r'\left' + left_delim + block_str + \
                        r'\right' + right_delim

        if expr.rank() == 0:
            return block_str % ""

        level_str: list[list[str]] = [[] for i in range(expr.rank() + 1)]
        shape_ranges = [list(range(i)) for i in expr.shape]
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(self._print(expr[outer_i]))
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(
                        r" & ".join(level_str[back_outer_i+1]))
                else:
                    level_str[back_outer_i].append(
                        block_str % (r"\\".join(level_str[back_outer_i+1])))
                    if len(level_str[back_outer_i+1]) == 1:
                        level_str[back_outer_i][-1] = r"\left[" + \
                            level_str[back_outer_i][-1] + r"\right]"
                even = not even
                level_str[back_outer_i+1] = []

        out_str = level_str[0][0]

        if expr.rank() % 2 == 1:
            out_str = block_str % out_str

        return out_str

    def _printer_tensor_indices(self, name, indices, index_map: dict):
        out_str = self._print(name)
        last_valence = None
        prev_map = None
        for index in indices:
            new_valence = index.is_up
            if ((index in index_map) or prev_map) and \
                    last_valence == new_valence:
                out_str += ","
            if last_valence != new_valence:
                if last_valence is not None:
                    out_str += "}"
                if index.is_up:
                    out_str += "{}^{"
                else:
                    out_str += "{}_{"
            out_str += self._print(index.args[0])
            if index in index_map:
                out_str += "="
                out_str += self._print(index_map[index])
                prev_map = True
            else:
                prev_map = False
            last_valence = new_valence
        if last_valence is not None:
            out_str += "}"
        return out_str

    def _print_Tensor(self, expr):
        name = expr.args[0].args[0]
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices, {})

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].args[0]
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        a = []
        args = expr.args
        for x in args:
            a.append(self.parenthesize(x, precedence(expr)))
        a.sort()
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _print_TensorIndex(self, expr):
        return "{}%s{%s}" % (
            "^" if expr.is_up else "_",
            self._print(expr.args[0])
        )

    def _print_PartialDerivative(self, expr):
        if len(expr.variables) == 1:
            return r"\frac{\partial}{\partial {%s}}{%s}" % (
                self._print(expr.variables[0]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )
        else:
            return r"\frac{\partial^%s}{%s}{%s}" % (
                self._get_optional_curly_braces(len(expr.variables), True),
                " ".join([r"\partial {%s}" % self._print(i) for i in expr.variables]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )

    def _print_ArraySymbol(self, expr):
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        return "{{%s}_{%s}}" % (
            self.parenthesize(expr.name, PRECEDENCE["Func"], True),
            ", ".join([f"{self._print(i)}" for i in expr.indices]))

    def _print_Dots(self, expr):
        return self._get_setting('dots', update_stats=True)

    def _print_UniversalSet(self, expr):
        return r"\mathbb{U}"

    def _print_frac(self, expr, exp=None):
        if exp is None:
            return r"\operatorname{frac}{\left(%s\right)}" % self._print(expr.args[0])
        else:
            return r"\operatorname{frac}{\left(%s\right)}^%s" % (
                    self._print(expr.args[0]), self._get_optional_curly_braces(exp, True))

    def _print_tuple(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            sep = "; "
        elif self._settings['decimal_separator'] == 'period':
            sep = ", "
        else:
            raise ValueError('Unknown Decimal Separator')

        if len(expr) == 1:
            # 1-tuple needs a trailing separator
            return self._add_parens_lspace(self._print(expr[0]) + sep)
        else:
            return self._add_parens_lspace(
                sep.join([self._print(i) for i in expr]))

    def _print_TensorProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \wedge '.join(elements)

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_list(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            return r"\left[ %s\right]" % \
                r"; \  ".join([self._print(i) for i in expr])
        elif self._settings['decimal_separator'] == 'period':
            return r"\left[ %s\right]" % \
                r", \  ".join([self._print(i) for i in expr])
        else:
            raise ValueError('Unknown Decimal Separator')


    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left\{ %s\right\}" % r", \  ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_DiracDelta(self, expr, exp=None):
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = r"\delta\left(%s\right)" % self._print(expr.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left( %s \right)" % (
                self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    def _print_SingularityFunction(self, expr, exp=None):
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = r"{\left\langle %s \right\rangle}^{%s}" % (shift, power)
        if exp is not None:
            tex = r"{\left({\langle %s \rangle}^{%s}\right)}^{%s}" % (shift, power, exp)
        return tex

    def _print_Heaviside(self, expr, exp=None):
        pargs = ', '.join(self._print(arg) for arg in expr.pargs)
        tex = r"\theta\left(%s\right)" % pargs
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    def _print_KroneckerDelta(self, expr, exp=None):
        i = self._print(expr.args[0])
        j = self._print(expr.args[1])
        if expr.args[0].is_Atom and expr.args[1].is_Atom:
            tex = r'\delta_{%s %s}' % (i, j)
        else:
            tex = r'\delta_{%s, %s}' % (i, j)
        if exp is not None:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_LeviCivita(self, expr, exp=None):
        indices = map(self._print, expr.args)
        if all(x.is_Atom for x in expr.args):
            tex = r'\varepsilon_{%s}' % " ".join(indices)
        else:
            tex = r'\varepsilon_{%s}' % ", ".join(indices)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return '\\text{Domain: }' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('\\text{Domain: }' + self._print(d.symbols) + ' \\in ' +
                    self._print(d.set))
        elif hasattr(d, 'symbols'):
            return '\\text{Domain on }' + self._print(d.symbols)
        else:
            return self._print(None)

    def _print_FiniteSet(self, s):
        return self._print_set(s.args)

    def _print_set(self, s):
        if self._get_setting('order') == 'random':
            items = list(s)
            random.shuffle(items)
        else:
            items = sorted(s, key=default_sort_key)
        if self._settings['decimal_separator'] == 'comma':
            items = "; ".join(map(self._print, items))
        elif self._settings['decimal_separator'] == 'period':
            items = ", ".join(map(self._print, items))
        else:
            raise ValueError('Unknown Decimal Separator')
        return r"\left\{%s\right\}" % items

    def _print_SetComplement(self, expr):
        return self._get_setting('set_complement') % self._print(expr.args[0])

    def _print_VarIsInSet(self, s):
        return '%s \\in %s' % (self._print(s.args[0]), self._print(s.args[1]))

    def _print_Subset(self, expr):
        return '%s %s %s' % (self._print(expr.args[0]), self._get_setting('subset'), self._print(expr.args[1]))

    def _print_Superset(self, expr):
        return '%s %s %s' % (self._print(expr.args[0]), self._get_setting('superset'), self._print(expr.args[1]))


    _print_frozenset = _print_set

    def _print_Range(self, s):
        def _print_symbolic_range():
            # Symbolic Range that cannot be resolved
            if s.args[0] == 0:
                if s.args[2] == 1:
                    cont = self._print(s.args[1])
                else:
                    cont = ", ".join(self._print(arg) for arg in s.args)
            else:
                if s.args[2] == 1:
                    cont = ", ".join(self._print(arg) for arg in s.args[:2])
                else:
                    cont = ", ".join(self._print(arg) for arg in s.args)

            return(f"\\text{{Range}}\\left({cont}\\right)")

        dots = object()

        if s.start.is_infinite and s.stop.is_infinite:
            if s.step.is_positive:
                printset = dots, -1, 0, 1, dots
            else:
                printset = dots, 1, 0, -1, dots
        elif s.start.is_infinite:
            printset = dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite:
            it = iter(s)
            printset = next(it), next(it), dots
        elif s.is_empty is not None:
            if (s.size < 4) == True:
                printset = tuple(s)
            elif s.is_iterable:
                it = iter(s)
                printset = next(it), next(it), dots, s[-1]
            else:
                return _print_symbolic_range()
        else:
            return _print_symbolic_range()
        return (r"\left\{" +
                r", ".join(self._print(el) if el is not dots else r'\ldots' for el in printset) +
                r"\right\}")

    def __print_number_polynomial(self, expr, letter, exp=None):
        if len(expr.args) == 2:
            if exp is not None:
                return r"%s_{%s}^{%s}\left(%s\right)" % (letter,
                            self._print(expr.args[0]), exp,
                            self._print(expr.args[1]))
            return r"%s_{%s}\left(%s\right)" % (letter,
                        self._print(expr.args[0]), self._print(expr.args[1]))

        tex = r"%s_{%s}" % (letter, self._print(expr.args[0]))
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_bernoulli(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "B", exp)

    def _print_bell(self, expr, exp=None):
        if len(expr.args) == 3:
            tex1 = r"B_{%s, %s}" % (self._print(expr.args[0]),
                                self._print(expr.args[1]))
            tex2 = r"\left(%s\right)" % r", ".join(self._print(el) for
                                               el in expr.args[2])
            if exp is not None:
                tex = r"%s^{%s}%s" % (tex1, exp, tex2)
            else:
                tex = tex1 + tex2
            return tex
        return self.__print_number_polynomial(expr, "B", exp)


    def _print_fibonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "F", exp)

    def _print_lucas(self, expr, exp=None):
        tex = r"L_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_StringFormula(self, expr):
        return expr.getText()

    def _print_tribonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "T", exp)

    def _print_SeqFormula(self, s):
        dots = object()
        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            return r"\left\{%s\right\}_{%s=%s}^{%s}" % (
                self._print(s.formula),
                self._print(s.variables[0]),
                self._print(s.start),
                self._print(s.stop)
            )
        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2),
                        s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(dots)
        else:
            printset = tuple(s)

        return (r"\left[" +
                r", ".join(self._print(el) if el is not dots else r'\ldots' for el in printset) +
                r"\right]")

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_Interval(self, i):
        if i.start == i.end:
            return r"\left\{%s\right\}" % self._print(i.start)

        else:
            open = self._get_setting('interval_open', update_stats=True)
            if i.left_open:
                left = open[0]
            else:
                left = '['

            if i.right_open:
                right = open[1]
            else:
                right = ']'

            arg = ("," + self._get_optional_blank()).join([self._print(i.start), self._print(i.end)])
            return self._get_setting('interval') % (left, arg, right)

    def _print_AccumulationBounds(self, i):
        return r"\left\langle %s, %s\right\rangle" % \
                (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cup ".join(args_str)

    def _print_Complement(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \setminus ".join(args_str)

    def _print_Intersection(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cap ".join(args_str)

    def _print_SymmetricDifference(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \triangle ".join(args_str)

    def _print_ProductSet(self, p):
        prec = precedence_traditional(p)
        if len(p.sets) >= 1 and not has_variety(p.sets):
            return self.parenthesize(p.sets[0], prec) + "^{%d}" % len(p.sets)
        return r" \times ".join(
            self.parenthesize(set, prec) for set in p.sets)

    def _print_EmptySet(self, e):
        return self._get_setting('emptyset', update_stats=True)

    def _print__math_set(self, symbol: str):
        return self._get_setting('math_set') % symbol

    def _print_Naturals(self, n):
        return self._print__math_set('N')

    def _print_Naturals0(self, n):
        return self._print__math_set('N') + '_0'

    def _print_Integers(self, i):
        return self._print__math_set('Z')

    def _print_Rationals(self, i):
        return self._print__math_set('Q')

    def _print_Reals(self, i):
        return self._print__math_set('R')

    def _print_Complexes(self, i):
        return self._print__math_set('C')

    def _print_ImaginaryUnit(self, expr):
        return self._get_setting('imaginary_unit_latex')

    def _print_ImageSet(self, s):
        expr = s.lamda.expr
        sig = s.lamda.signature
        xys = ((self._print(x), self._print(y)) for x, y in zip(sig, s.base_sets))
        xinys = r", ".join(r"%s \in %s" % xy for xy in xys)
        return r"\left\{%s\; \middle|\; %s\right\}" % (self._print(expr), xinys)

    def _print_ConditionSet(self, s):
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        pattern = self._get_setting('condition_set', update_stats=True)
        condition = self._print(s.condition)
        base_set = self._print(s.base_set)

        if s.base_set is S.UniversalSet:
            lhs = vars_print
            rhs = condition
        elif self._get_setting('set_base_set_lhs'):
            lhs = r"%s \in %s" % (vars_print, base_set)
            rhs = condition
        else:
            lhs = vars_print
            rhs = self._get_setting('formula_delimiter').join(["%s \in %s" % (vars_print, base_set), condition])

        return pattern % (lhs, rhs)

    def _print_Set(self, s):
        pattern = self._get_setting('set')
        args = [self._print(a) for a in s.args]
        if any(',' in a for a in args):
            delimiter = ';'
        else:
            delimiter = ','
        return pattern % (delimiter + self._get_optional_blank()).join(args)

    def _print_SetMinus(self, s):
        return "%s %s %s" % (self._print(s.args[0]), self._get_setting('set_minus'), self._print(s.args[1]))

    def _print_SetX(self, s, type):
        sub = ''
        if len(s.args) == 2:
            sub = "_{%s}" % self._print(s.args[1])

        return r'\%s%s{%s}' % (type, sub, self._print(s.args[0]))


    def _print_SetMin(self, s):
        return self._print_SetX(s, 'min')

    def _print_SetMax(self, s):
        return self._print_SetX(s, 'max')

    def _print_SetSup(self, s):
        return self._print_SetX(s, 'sup')

    def _print_SetInf(self, s):
        return self._print_SetX(s, 'inf')

    def _print_NamedSet(self, s):
        name = s.args[0]
        return name

    def _print_Percentage(self, expr):
        return "%s\\%%" % self._print(expr.args[0])

    def _print_PowerSet(self, expr):
        arg_print = self._print(expr.args[0])
        return r"\mathcal{{P}}\left({}\right)".format(arg_print)

    def _print_ComplexRegion(self, s):
        vars_print = ', '.join([self._print(var) for var in s.variables])
        return r"\left\{%s\; \middle|\; %s \in %s \right\}" % (
            self._print(s.expr),
            vars_print,
            self._print(s.sets))

    def _print_Contains(self, e):
        return r"%s \in %s" % tuple(self._print(a) for a in e.args)

    def _print_FourierSeries(self, s):
        if s.an.formula is S.Zero and s.bn.formula is S.Zero:
            return self._print(s.a0)
        return self._print_Add(s.truncate()) + r' + \ldots'

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_FiniteField(self, expr):
        return r"\mathbb{F}_{%s}" % expr.mod

    def _print_IntegerRing(self, expr):
        return r"\mathbb{Z}"

    def _print_RationalField(self, expr):
        return r"\mathbb{Q}"

    def _print_RealField(self, expr):
        return r"\mathbb{R}"

    def _print_ComplexField(self, expr):
        return r"\mathbb{C}"

    def _print_PolynomialRing(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left[%s\right]" % (domain, symbols)

    def _print_FractionField(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left(%s\right)" % (domain, symbols)

    def _print_PolynomialRingBase(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        inv = ""
        if not expr.is_Poly:
            inv = r"S_<^{-1}"
        return r"%s%s\left[%s\right]" % (inv, domain, symbols)

    def _print_Poly(self, poly):
        cls = poly.__class__.__name__
        terms = []
        if self._get_setting('left_right_brackets', update_stats=True):
            left_brace = r'\left('
            right_brace = r'\right)'
        else:
            left_brace = '('
            right_brace = ')'
        for monom, coeff in poly.terms():
            s_monom = ''
            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))

            if coeff.is_Add:
                if s_monom:
                    s_coeff = (left_brace + "%s" + right_brace) % self._print(coeff)
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + " " + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        expr = ' '.join(terms)
        gens = list(map(self._print, poly.gens))
        domain = "domain=%s" % self._print(poly.get_domain())

        args = ", ".join([expr] + gens + [domain])
        if cls in accepted_latex_functions:
            tex = (r"\%s" + left_brace + "%s" + right_brace) % (cls, args)
        else:
            tex = (r"\operatorname{%s}" + left_brace + "%s" + right_brace) % (cls, args)

        return tex

    def _print_ComplexRootOf(self, root):
        cls = root.__class__.__name__
        if cls == "ComplexRootOf":
            cls = "CRootOf"
        expr = self._print(root.expr)
        index = root.index
        if cls in accepted_latex_functions:
            return r"\%s {\left(%s, %d\right)}" % (cls, expr, index)
        else:
            return r"\operatorname{%s} {\left(%s, %d\right)}" % (cls, expr,
                                                                 index)

    def _print_RootSum(self, expr):
        cls = expr.__class__.__name__
        args = [self._print(expr.expr)]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        if cls in accepted_latex_functions:
            return r"\%s {\left(%s\right)}" % (cls, ", ".join(args))
        else:
            return r"\operatorname{%s} {\left(%s\right)}" % (cls,
                                                             ", ".join(args))

    def _print_OrdinalOmega(self, expr):
        return r"\omega"

    def _print_OmegaPower(self, expr):
        exp, mul = expr.args
        if mul != 1:
            if exp != 1:
                return r"{} \omega^{{{}}}".format(mul, exp)
            else:
                return r"{} \omega".format(mul)
        else:
            if exp != 1:
                return r"\omega^{{{}}}".format(exp)
            else:
                return r"\omega"

    def _print_Ordinal(self, expr):
        return " + ".join([self._print(arg) for arg in expr.args])

    def _print_PolyElement(self, poly):
        mul_symbol = self._settings['mul_symbol_latex']
        return poly.str(self, PRECEDENCE, "{%s}^{%d}", mul_symbol)

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return r"\frac{%s}{%s}" % (numer, denom)

    def _print_euler(self, expr, exp=None):
        m, x = (expr.args[0], None) if len(expr.args) == 1 else expr.args
        tex = r"E_{%s}" % self._print(m)
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        if x is not None:
            tex = r"%s\left(%s\right)" % (tex, self._print(x))
        return tex

    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_UnifiedTransform(self, expr, s, inverse=False):
        return r"\mathcal{{{}}}{}_{{{}}}\left[{}\right]\left({}\right)".format(s, '^{-1}' if inverse else '', self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_MellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M')

    def _print_InverseMellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M', True)

    def _print_LaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L')

    def _print_InverseLaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L', True)

    def _print_FourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F')

    def _print_InverseFourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F', True)

    def _print_SineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN')

    def _print_InverseSineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN', True)

    def _print_CosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS')

    def _print_InverseCosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS', True)

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(Symbol(object.name))

    def _print_LambertW(self, expr, exp=None):
        arg0 = self._print(expr.args[0])
        exp = r"^{%s}" % (exp,) if exp is not None else ""
        if len(expr.args) == 1:
            result = r"W%s\left(%s\right)" % (exp, arg0)
        else:
            arg1 = self._print(expr.args[1])
            result = "W{0}_{{{1}}}\\left({2}\\right)".format(exp, arg1, arg0)
        return result

    def _print_Probability_Functions(self, expr, type):
        name = self._get_setting(type)
        arg = self._print(expr.args[0])
        function_brackets = self._get_setting('function_brackets')
        return function_brackets % (name, arg)

    def _print_Expectation(self, expr):
        return self._print_Probability_Functions(expr, 'expected_value')

    def _print_Variance(self, expr):
        return self._print_Probability_Functions(expr, 'variance')

    def _print_Covariance(self, expr):
        name = self._settings['covariance']
        args = [self._print(a) for a in expr.args]
        function_brackets = self._settings['function_brackets']
        if isinstance(function_brackets, str):
            bracket_mode = function_brackets
        else:
            bracket_mode = function_brackets._random_state()
        return bracket_mode % (name, ("," + self._get_optional_blank()).join(args))

    def _print_Probability(self, expr):
        return self._print_Probability_Functions(expr, 'probability')

    def _print_with_original_order(self, print_function):
        old_order = self._settings['order']
        self._settings['order'] = 'original'
        s = print_function()
        self._settings['order'] = old_order
        return s

    def _print_BasicProbability(self, expr):
        p = self._get_setting('probability')
        arg = expr.args[0]
        return self._print_with_original_order(lambda : p + self._add_parens(self._print(arg)))

    def _print_ConditionalProbability(self, expr):
        p = self._get_setting('probability')
        return "%s(%s|%s)" % (p, self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Morphism(self, morphism):
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        return "%s\\rightarrow %s" % (domain, codomain)

    def _print_TransferFunction(self, expr):
        num, den = self._print(expr.num), self._print(expr.den)
        return r"\frac{%s}{%s}" % (num, den)

    def _print_Series(self, expr):
        args = list(expr.args)
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                            False)
        return ' '.join(map(parens, args))

    def _print_MIMOSeries(self, expr):
        from sympy.physics.control.lti import MIMOParallel
        args = list(expr.args)[::-1]
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                             False) if isinstance(x, MIMOParallel) else self._print(x)
        return r"\cdot".join(map(parens, args))

    def _print_Parallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    def _print_MIMOParallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    def _print_Feedback(self, expr):
        from sympy.physics.control import TransferFunction, Series

        num, tf = expr.sys1, TransferFunction(1, 1, expr.var)
        num_arg_list = list(num.args) if isinstance(num, Series) else [num]
        den_arg_list = list(expr.sys2.args) if \
            isinstance(expr.sys2, Series) else [expr.sys2]
        den_term_1 = tf

        if isinstance(num, Series) and isinstance(expr.sys2, Series):
            den_term_2 = Series(*num_arg_list, *den_arg_list)
        elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
            if expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = tf, Series(*num_arg_list, expr.sys2)
        elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            else:
                den_term_2 = Series(num, *den_arg_list)
        else:
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            elif expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = Series(*num_arg_list, *den_arg_list)

        numer = self._print(num)
        denom_1 = self._print(den_term_1)
        denom_2 = self._print(den_term_2)
        _sign = "+" if expr.sign == -1 else "-"

        return r"\frac{%s}{%s %s %s}" % (numer, denom_1, _sign, denom_2)

    def _print_MIMOFeedback(self, expr):
        from sympy.physics.control import MIMOSeries
        inv_mat = self._print(MIMOSeries(expr.sys2, expr.sys1))
        sys1 = self._print(expr.sys1)
        _sign = "+" if expr.sign == -1 else "-"
        return r"\left(I_{\tau} %s %s\right)^{-1} \cdot %s" % (_sign, inv_mat, sys1)

    def _print_TransferFunctionMatrix(self, expr):
        mat = self._print(expr._expr_mat)
        return r"%s_\tau" % mat

    def _print_DFT(self, expr):
        return r"\text{{{}}}_{{{}}}".format(expr.__class__.__name__, expr.n)
    _print_IDFT = _print_DFT

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(Symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return "%s:%s" % (pretty_name, pretty_morphism)

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(NamedMorphism(
            morphism.domain, morphism.codomain, "id"))

    def _print_CompositeMorphism(self, morphism):
        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [self._print(Symbol(component.name)) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = "\\circ ".join(component_names_list) + ":"

        pretty_morphism = self._print_Morphism(morphism)
        return component_names + pretty_morphism

    def _print_Category(self, morphism):
        return r"\mathbf{{{}}}".format(self._print(Symbol(morphism.name)))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        latex_result = self._print(diagram.premises)
        if diagram.conclusions:
            latex_result += "\\Longrightarrow %s" % \
                            self._print(diagram.conclusions)

        return latex_result

    def _print_DiagramGrid(self, grid):
        latex_result = "\\begin{array}{%s}\n" % ("c" * grid.width)

        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    latex_result += latex(grid[i, j])
                latex_result += " "
                if j != grid.width - 1:
                    latex_result += "& "

            if i != grid.height - 1:
                latex_result += "\\\\"
            latex_result += "\n"

        latex_result += "\\end{array}\n"
        return latex_result

    def _print_FreeModule(self, M):
        return '{{{}}}^{{{}}}'.format(self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in m))

    def _print_SubModule(self, m):
        return r"\left\langle {} \right\rangle".format(",".join(
            '{' + self._print(x) + '}' for x in m.gens))

    def _print_ModuleImplementedIdeal(self, m):
        return r"\left\langle {} \right\rangle".format(",".join(
            '{' + self._print(x) + '}' for [x] in m._module.gens))

    def _print_Quaternion(self, expr):
        # TODO: This expression is potentially confusing,
        # shall we print it as `Quaternion( ... )`?
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True)
             for i in expr.args]
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_QuotientRing(self, R):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(R.ring),
                 self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        return r"{{{}}} + {{{}}}".format(self._print(x.data),
                 self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        return r"{{{}}} + {{{}}}".format(self._print(m.data),
                 self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(M.base),
                 self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return r"{{{}}} : {{{}}} \to {{{}}}".format(self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    def _print_Manifold(self, manifold):
        string = manifold.name.name
        if '{' in string:
            name, supers, subs = string, [], []
        else:
            name, supers, subs = split_super_sub(string)

            name = translate(name)
            supers = [translate(sup) for sup in supers]
            subs = [translate(sub) for sub in subs]

        name = r'\text{%s}' % name
        if supers:
            name += "^{%s}" % " ".join(supers)
        if subs:
            name += "_{%s}" % " ".join(subs)

        return name

    def _print_Patch(self, patch):
        return r'\text{%s}_{%s}' % (self._print(patch.name), self._print(patch.manifold))

    def _print_CoordSystem(self, coordsys):
        return r'\text{%s}^{\text{%s}}_{%s}' % (
            self._print(coordsys.name), self._print(coordsys.patch.name), self._print(coordsys.manifold)
        )

    def _print_CovarDerivativeOp(self, cvd):
        return r'\mathbb{\nabla}_{%s}' % self._print(cvd._wrt)

    def _print_BaseScalarField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return r'\mathbf{{{}}}'.format(self._print(Symbol(string)))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return r'\partial_{{{}}}'.format(self._print(Symbol(string)))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys.symbols[field._index].name
            return r'\operatorname{{d}}{}'.format(self._print(Symbol(string)))
        else:
            string = self._print(field)
            return r'\operatorname{{d}}\left({}\right)'.format(string)

    def _print_Tr(self, p):
        # TODO: Handle indices
        contents = self._print(p.args[0])
        return r'\operatorname{{tr}}\left({}\right)'.format(contents)

    def _print_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\phi\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\phi\left(%s\right)' % self._print(expr.args[0])

    def _print_reduced_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\lambda\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\lambda\left(%s\right)' % self._print(expr.args[0])

    def _print_divisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^{%s}%s" % (exp, tex)
        return r"\sigma%s" % tex

    def _print_udivisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^*^{%s}%s" % (exp, tex)
        return r"\sigma^*%s" % tex

    def _print_primenu(self, expr, exp=None):
        if exp is not None:
            return r'\left(\nu\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\nu\left(%s\right)' % self._print(expr.args[0])

    def _print_primeomega(self, expr, exp=None):
        if exp is not None:
            return r'\left(\Omega\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\Omega\left(%s\right)' % self._print(expr.args[0])

    def _print_Str(self, s):
        return str(s.name)

    def _print_String(self, s):
        return s.text

    def _print_float(self, expr):
        return self._print(Float(expr))

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_mpq(self, expr):
        return str(expr)

    def _print_Predicate(self, expr):
        return r"\operatorname{{Q}}_{{\text{{{}}}}}".format(latex_escape(str(expr.name)))

    def _print_AppliedPredicate(self, expr):
        pred = expr.function
        args = expr.arguments
        pred_latex = self._print(pred)
        args_latex = ', '.join([self._print(a) for a in args])
        return '%s(%s)' % (pred_latex, args_latex)

    def emptyPrinter(self, expr):
        # default to just printing as monospace, like would normally be shown
        s = super().emptyPrinter(expr)

        return r"\mathtt{\text{%s}}" % latex_escape(s)

def translate(s: str) -> str:
    r'''
    Check for a modifier ending the string.  If present, convert the
    modifier to latex and translate the rest recursively.

    Given a description of a Greek letter or other special character,
    return the appropriate latex.

    Let everything else pass as given.

    >>> from sympy.printing.latex import translate
    >>> translate('alphahatdotprime')
    "{\\dot{\\hat{\\alpha}}}'"
    '''
    # Process the rest
    tex = tex_greek_dictionary.get(s)
    if tex:
        return tex
    elif s.lower() in greek_letters_set:
        return "\\" + s.lower()
    elif s in other_symbols:
        return "\\" + s
    else:
        # Process modifiers, if any, and recurse
        for key in sorted(modifier_dict.keys(), key=len, reverse=True):
            if s.lower().endswith(key) and len(s) > len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        return s

@print_function(LatexPrinter)
def latex(expr, **settings):
    r"""Convert the given expression to LaTeX string representation.

    Parameters
    ==========
    full_prec: boolean, optional
        If set to True, a floating point number is printed with full precision.
    fold_frac_powers : boolean, optional
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
    fold_func_brackets : boolean, optional
        Fold function brackets where applicable.
    fold_short_frac : boolean, optional
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.
    inv_trig_style : string, optional
        How inverse trig functions should be displayed. Can be one of
        ``'abbreviated'``, ``'full'``, or ``'power'``. Defaults to
        ``'abbreviated'``.
    itex : boolean, optional
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.
    ln_notation : boolean, optional
        If set to ``True``, ``\ln`` is used instead of default ``\log``.
    long_frac_ratio : float or None, optional
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.
    mat_delim : string, optional
        The delimiter to wrap around matrices. Can be one of ``'['``, ``'('``,
        or the empty string ``''``. Defaults to ``'['``.
    mat_str : string, optional
        Which matrix environment string to emit. ``'smallmatrix'``,
        ``'matrix'``, ``'array'``, etc. Defaults to ``'smallmatrix'`` for
        inline mode, ``'matrix'`` for matrices of no more than 10 columns, and
        ``'array'`` otherwise.
    mode: string, optional
        Specifies how the generated code will be delimited. ``mode`` can be one
        of ``'plain'``, ``'inline'``, ``'equation'`` or ``'equation*'``.  If
        ``mode`` is set to ``'plain'``, then the resulting code will not be
        delimited at all (this is the default). If ``mode`` is set to
        ``'inline'`` then inline LaTeX ``$...$`` will be used. If ``mode`` is
        set to ``'equation'`` or ``'equation*'``, the resulting code will be
        enclosed in the ``equation`` or ``equation*`` environment (remember to
        import ``amsmath`` for ``equation*``), unless the ``itex`` option is
        set. In the latter case, the ``$$...$$`` syntax is used.
    mul_symbol : string or None, optional
        The symbol to use for multiplication. Can be one of ``None``,
        ``'ldot'``, ``'dot'``, or ``'times'``.
    order: string, optional
        Any of the supported monomial orderings (currently ``'lex'``,
        ``'grlex'``, or ``'grevlex'``), ``'old'``, and ``'none'``. This
        parameter does nothing for `~.Mul` objects. Setting order to ``'old'``
        uses the compatibility ordering for ``~.Add`` defined in Printer. For
        very large expressions, set the ``order`` keyword to ``'none'`` if
        speed is a concern.
    symbol_names : dictionary of strings mapped to symbols, optional
        Dictionary of symbols and the custom strings they should be emitted as.
    root_notation : boolean, optional
        If set to ``False``, exponents of the form 1/n are printed in fractonal
        form. Default is ``True``, to print exponent in root form.
    mat_symbol_style : string, optional
        Can be either ``'plain'`` (default) or ``'bold'``. If set to
        ``'bold'``, a `~.MatrixSymbol` A will be printed as ``\mathbf{A}``,
        otherwise as ``A``.
    imaginary_unit : string, optional
        String to use for the imaginary unit. Defined options are ``'i'``
        (default) and ``'j'``. Adding ``r`` or ``t`` in front gives ``\mathrm``
        or ``\text``, so ``'ri'`` leads to ``\mathrm{i}`` which gives
        `\mathrm{i}`.
    gothic_re_im : boolean, optional
        If set to ``True``, `\Re` and `\Im` is used for ``re`` and ``im``, respectively.
        The default is ``False`` leading to `\operatorname{re}` and `\operatorname{im}`.
    decimal_separator : string, optional
        Specifies what separator to use to separate the whole and fractional parts of a
        floating point number as in `2.5` for the default, ``period`` or `2{,}5`
        when ``comma`` is specified. Lists, sets, and tuple are printed with semicolon
        separating the elements when ``comma`` is chosen. For example, [1; 2; 3] when
        ``comma`` is chosen and [1,2,3] for when ``period`` is chosen.
    parenthesize_super : boolean, optional
        If set to ``False``, superscripted expressions will not be parenthesized when
        powered. Default is ``True``, which parenthesizes the expression when powered.
    min: Integer or None, optional
        Sets the lower bound for the exponent to print floating point numbers in
        fixed-point format.
    max: Integer or None, optional
        Sets the upper bound for the exponent to print floating point numbers in
        fixed-point format.
    diff_operator: string, optional
        String to use for differential operator. Default is ``'d'``, to print in italic
        form. ``'rd'``, ``'td'`` are shortcuts for ``\mathrm{d}`` and ``\text{d}``.

    Notes
    =====

    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    >>> from sympy import latex, Rational
    >>> from sympy.abc import tau
    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    Examples
    ========

    >>> from sympy import latex, pi, sin, asin, Integral, Matrix, Rational, log
    >>> from sympy.abc import x, y, mu, r, tau

    Basic usage:

    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    ``mode`` and ``itex`` options:

    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$
    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    Fraction options:

    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}
    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left(\frac{7}{2} \right)}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets=True))
    \left(2 \tau\right)^{\sin {\frac{7}{2}}}
    >>> print(latex(3*x**2/y))
    \frac{3 x^{2}}{y}
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    3 x^{2} / y
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    \frac{\int r\, dr}{2 \pi}
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=0))
    \frac{1}{2 \pi} \int r\, dr

    Multiplication options:

    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))
    \left(2 \times \tau\right)^{\sin{\left(\frac{7}{2} \right)}}

    Trig options:

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left(\frac{7}{2} \right)}

    Matrix options:

    >>> print(latex(Matrix(2, 1, [x, y])))
    \left[\begin{matrix}x\\y\end{matrix}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    \left[\begin{array}{c}x\\y\end{array}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))
    \left(\begin{matrix}x\\y\end{matrix}\right)

    Custom printing of symbols:

    >>> print(latex(x**2, symbol_names={x: 'x_i'}))
    x_i^{2}

    Logarithms:

    >>> print(latex(log(10)))
    \log{\left(10 \right)}
    >>> print(latex(log(10), ln_notation=True))
    \ln{\left(10 \right)}

    ``latex()`` also supports the builtin container types :class:`list`,
    :class:`tuple`, and :class:`dict`:

    >>> print(latex([2/x, y], mode='inline'))
    $\left[ 2 / x, \  y\right]$

    Unsupported types are rendered as monospaced plaintext:

    >>> print(latex(int))
    \mathtt{\text{<class 'int'>}}
    >>> print(latex("plain % text"))
    \mathtt{\text{plain \% text}}

    See :ref:`printer_method_example` for an example of how to override
    this behavior for your own types by implementing ``_latex``.

    .. versionchanged:: 1.7.0
        Unsupported types no longer have their ``str`` representation treated as valid latex.

    """
    return LatexPrinter(settings).doprint(expr)


def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression. Takes the same
    settings as ``latex()``."""

    print(latex(expr, **settings))


def multiline_latex(lhs, rhs, terms_per_line=1, environment="align*", use_dots=False, **settings):
    r"""
    This function generates a LaTeX equation with a multiline right-hand side
    in an ``align*``, ``eqnarray`` or ``IEEEeqnarray`` environment.

    Parameters
    ==========

    lhs : Expr
        Left-hand side of equation

    rhs : Expr
        Right-hand side of equation

    terms_per_line : integer, optional
        Number of terms per line to print. Default is 1.

    environment : "string", optional
        Which LaTeX wnvironment to use for the output. Options are "align*"
        (default), "eqnarray", and "IEEEeqnarray".

    use_dots : boolean, optional
        If ``True``, ``\\dots`` is added to the end of each line. Default is ``False``.

    Examples
    ========

    >>> from sympy import multiline_latex, symbols, sin, cos, exp, log, I
    >>> x, y, alpha = symbols('x y alpha')
    >>> expr = sin(alpha*y) + exp(I*alpha) - cos(log(y))
    >>> print(multiline_latex(x, expr))
    \begin{align*}
    x = & e^{i \alpha} \\
    & + \sin{\left(\alpha y \right)} \\
    & - \cos{\left(\log{\left(y \right)} \right)}
    \end{align*}

    Using at most two terms per line:
    >>> print(multiline_latex(x, expr, 2))
    \begin{align*}
    x = & e^{i \alpha} + \sin{\left(\alpha y \right)} \\
    & - \cos{\left(\log{\left(y \right)} \right)}
    \end{align*}

    Using ``eqnarray`` and dots:
    >>> print(multiline_latex(x, expr, terms_per_line=2, environment="eqnarray", use_dots=True))
    \begin{eqnarray}
    x & = & e^{i \alpha} + \sin{\left(\alpha y \right)} \dots\nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)}
    \end{eqnarray}

    Using ``IEEEeqnarray``:
    >>> print(multiline_latex(x, expr, environment="IEEEeqnarray"))
    \begin{IEEEeqnarray}{rCl}
    x & = & e^{i \alpha} \nonumber\\
    & & + \sin{\left(\alpha y \right)} \nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)}
    \end{IEEEeqnarray}

    Notes
    =====

    All optional parameters from ``latex`` can also be used.

    """

    # Based on code from https://github.com/sympy/sympy/issues/3001
    l = LatexPrinter(**settings)
    if environment == "eqnarray":
        result = r'\begin{eqnarray}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{eqnarray}'
        doubleet = True
    elif environment == "IEEEeqnarray":
        result = r'\begin{IEEEeqnarray}{rCl}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{IEEEeqnarray}'
        doubleet = True
    elif environment == "align*":
        result = r'\begin{align*}' + '\n'
        first_term = '= &'
        nonumber = ''
        end_term =  '\n\\end{align*}'
        doubleet = False
    else:
        raise ValueError("Unknown environment: {}".format(environment))
    dots = ''
    if use_dots:
        dots=r'\dots'
    terms = rhs.as_ordered_terms()
    n_terms = len(terms)
    term_count = 1
    for i in range(n_terms):
        term = terms[i]
        term_start = ''
        term_end = ''
        sign = '+'
        if term_count > terms_per_line:
            if doubleet:
                term_start = '& & '
            else:
                term_start = '& '
            term_count = 1
        if term_count == terms_per_line:
            # End of line
            if i < n_terms-1:
                # There are terms remaining
                term_end = dots + nonumber + r'\\' + '\n'
            else:
                term_end = ''

        if term.as_ordered_factors()[0] == -1:
            term = -1*term
            sign = r'-'
        if i == 0: # beginning
            if sign == '+':
                sign = ''
            result += r'{:s} {:s}{:s} {:s} {:s}'.format(l.doprint(lhs),
                        first_term, sign, l.doprint(term), term_end)
        else:
            result += r'{:s}{:s} {:s} {:s}'.format(term_start, sign,
                        l.doprint(term), term_end)
        term_count += 1
    result += end_term
    return result
