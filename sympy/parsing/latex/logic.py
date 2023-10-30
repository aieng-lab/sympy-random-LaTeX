import re
from copy import deepcopy

import sympy
from sympy.settings import randomize_settings
from sympy import latex, Dummy


known_variables = frozenset({
    'a', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\varepsilon', r'\zeta', r'\eta', r'\theta',
    r'\vartheta', r'\lambda', r'\mu', r'\nu', r'\xi', r'\rho', r'\sigma', r'\tau', r'\phi', r'\varphi',
    r'\chi', r'\psi', r'\omega',
})


logic_formula_recognizer = [r'\land', r'\lor', r'\neg', r'\exists', r'\forall', r'\lnot', r'\ln', r'\log', r'\not\in', '∧', '∨', r'\colon ', r'\wedge', r'\vee'] # space after \colon is important because of \coloneqq
latex_commands = logic_formula_recognizer + [r'\mathbb{R', r'\mathbb{N', r'\mathbb{Z', r'\mathbb{C', r'\mathbb{Q', r'\wedge', r'\vee', r'\cdot',
                                             r'\Rightarrow', r'\rightarrow', r'\Leftarrow', r'\leftarrow', r'\int', r'\Leftrightarrow', r'\Leftarrow', r'\sqrt', r'\pmod', r'\pm', r'\neq', r'\frac', r'\mathbb',  r'\Re', r'\R',
                                             r'\mathbb{C}', r'\mathrm{mod}', r'\infty', r'\zeta', r'\in', r'\ge', r'\left', r'\le', r'\;', r'\:', r'\,', r'\geq', r'\leq', r'\\{', r'\{',
                                             r'\ldots', r'\}', r'\colon', r'\sum_', r'\sum', r'\prod_', r'\prod', r'\bigsqcup', r'\bigsqcap',
                                             r'\epsilon', r'\pi', r'\not', r'\leqslant', r'\right', r'\!', r'\equiv', r'\mid', r'Rank(',
                                             r'\ldots', r' \mathcal{O}', r'\mathcal{C}', r'\mathcal{R}', r'\mathcal', r'\choose', r'\qquad', r'\quad', r'\dots',
                                              r'\widehat', r'\square', r'\to', r'\tag', 'dx', 's.t.', r'\|', r'\text{d}', r'\triangle', r'\rm', 'aligned', 'align', r'\sec', r'\gray',
                                             r'\iff', '&', r'\models',  r'\mod', 'mod', r'\gcd', r'\dim', r'\times', r'\mathrm{d}', r'\mathrm{Gal}', r'\mathrm',
                                             r'\operatorname', r'\bigr', r'\max', r'\mathfrak', r'\implies', r'\bigcup', r'\star', r'\operatorname{Sym}',
                                              r'\mathcal', r'\cap', r'\subseteq', r'\subset', r'\bigcap', r'\.', r'\cup', r'\cap',
                                             r'\begin', r'\end', r'\vert', r'\hom', r'\bot', r'\supset', r'\subset', r'\sup', r'\not', r'\min', 'trace', r'\bigtriangleup',
                                             r'\setminus', r'\nless', r'\mapsto', r'\overset', r'\overline', r'\overbrace', r'\over', 'P(', r'\biggl', r'\biggr', r'\bigg', r'\cos', r'\sin', r'\varepsilon',
                                             r'\displaystyle', r'\mathscr{P}', r'\langle', r'\propto', r'\limits', r'\lim', r'\Large', r'arc', r'\cos', r'\tan', 'cosh', 'sinh', 'tanh' 'cos', 'sin', 'tan',
                                             r'\vdots', r'\begin', r'\end', r'\iint', r'\iiint', r'\Ker', r'\Hom', 'Ker', 'ker', 'Hom', 'Gal', r'\circ', r'\bigoplus', r'Aut',
                                             r'\curvearrowright', r'\dfrac', r'\longrightarrow', r'\widetilde', r'\ni', r'\rangle', r'\exp', r'bmatrix', r'matrix', r'pmatrix', r'\binom', r'big', r'\mbox', r'\cong',
                                             'even', 'odd', r'\[', r'\]', r'\rtimes',
                                             # must be last!!!
                                             r'\\']
ESCAPE_CHAR_VARIABLES = '#'
ESCAPE_CHAR_FUNCTIONS = '§'
symbols = ['(', ')', '{', '}', '=', '<', '>', '+', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ',', ';',
           '^', '-', '/', '|', '[', ']', '°', '.', "'", '~', '∈', '∩', '!', '¬', '≡', '\n', ESCAPE_CHAR_VARIABLES, ESCAPE_CHAR_FUNCTIONS]


recursion_randomize_settings = randomize_settings.copy()
recursion_randomize_settings['order'] = 'original'

all_variables = set()

CHECK_VARIABLES = True

def set_check_variables(state):
    global CHECK_VARIABLES
    CHECK_VARIABLES = state

def is_logic_formula(text):
    return any(c in text for c in logic_formula_recognizer)

def remove_text_placeholders(input_string):
    def replace(match):
        return ' ' * len(match.group(0))

    for t in ['text', 'mathrm']:
        # Regular expression pattern to match "\\text{...}"
        pattern = r"\\%s\{.*?\}" % t

        # Use re.sub() with a custom replacement function
        input_string = re.sub(pattern, replace, input_string)

    return input_string

class StringFormula:

    def __init__(self, text: str, recurse=None):
        if ESCAPE_CHAR_VARIABLES in text or ESCAPE_CHAR_FUNCTIONS in text:
            raise ValueError("Text contains ESCAPE CHAR <%s>" % text)

        if len(text.strip()) > 60 and not any(c in text for c in logic_formula_recognizer + [')']):
            # large formulas should be a logic formula (otherwise the probability for substitution errors are too large)
            raise ValueError("Too long non-logic formula <%s> (len=%d)" % (text.strip(), len(text.strip())))


        self.raw = text
        text = text.replace(r'\x ', 'x ').replace(r'\y ', 'y ').replace(r'\x^', 'x^').replace(r'\y^', 'y^')
        self.relevant = text
        self.id = None

        self.recursion_command = None
        self.recursion = None
        if recurse:
            recurse_commands = [':', r'\colon', r'\Rightarrow']
            for c in recurse_commands:
                s = self.relevant.split(c)
                if len(s) > 1:
                    self.relevant = s[0]
                    self.recursion_command = c
                    self.recursion = recurse(c.join(s[1:]))
                    text = self.relevant
                    break

        text = remove_text_placeholders(text)
        for c in latex_commands + symbols:
            text = text.replace(c, len(c) * " ")

        text = text.removesuffix('\\')

        variables = set([t for t in text.split(' ') if len(t) > 0])


        mul_ctr = 0
        for v in variables.copy():
            if not v.startswith('\\') and len(v) > 1 and '_' not in v:
                # treat it as multiplication of multiple symbols
                variables.remove(v)
                for i, vv in enumerate(v):
                    mul_ctr += 1
                    if vv == '\\':
                        # now a special symbol like a greek letter has been found
                        variables.add(v[i:])
                        if '\\' in v[i+1:]:
                            raise ValueError("Multiple greek letters in multiplication")
                        break
                    variables.add(vv)

        if '_' in variables:
            variables.remove('_')

        if 'e' in variables:
            # could be eulers number
            variables.remove('e')

        if 'd' in variables:
            # could be derivative
            variables.remove('d')

        # remove Frobenius Norm "_F"
        variables = [v for v in variables if '_F'not in v]

        if "\\" in variables:
            #raise ValueError("WARNING: Some latex command has been missing: \n<%s>\n<%s>\n<%s>"% (self.raw, text, variables))
            variables.remove('\\')

        variables_ = variables
        variables = []
        for v in variables_:
            if '_' in v:
                variables.append(v)
            elif v in known_variables or (v == 'i' and any(z in self.raw for z in [r'\sum_i', r'\sum_{i'])):
                variables.append(v)
            else:
                text = text.replace(v, ' ' * len(v))

        # reduce variables to only known ones
        # sorting is important, so '\\alpha' comes before 'a', otherwise replacing a would cause
        #variables = list(sorted((variables_)))
        variables = list(sorted(variables_))

        if CHECK_VARIABLES and (mul_ctr > 2 or len(variables) > 7):
            raise ValueError("Found suspiciously many variables: <%s> in <%s>" % (variables, self.raw))

        occurences = {}
        for v in variables:
            vv = v.replace('\\', '\\\\') # escape re \\
            try:
                for m in re.finditer(vv, text):
                    occurences[m.start()] = v
            except Exception:
                pass


        # filter those out that are in range of others
        end = 0
        filtered_occurrences = {}
        for k, v in sorted(occurences.items(), key=lambda x: x[0]):
            if end <= k:
                filtered_occurrences[k] = v
                end = k + len(v)

        self.var_fnc_mapping = {'var': [], 'fnc': []}
        self.text = self.relevant
        for k, v in sorted(filtered_occurrences.items(), key=lambda x: -x[0]):
            self.text = self.text[:k] + self.wrap_escape_char(v, add=True) + self.text[k+len(v):]


    def _variables(self):
        s = set()
        for i, x in enumerate(self.text.split(ESCAPE_CHAR_VARIABLES)):
            if i % 2 == 1 and len(x) > 0:
                if x.startswith('\\'):
                    x = x.removeprefix('\\')
                s.add(self.create_symbol(x))
        return s

    @property
    def free_symbols(self):
        s = self._variables()
        if self.recursion is not None:
            return s.union(self.recursion.free_symbols)
        return s

    def wrap_escape_char(self, text: str, add=False, escape=None):
        if escape is None:
            if text in self.var_fnc_mapping['fnc']:
                ESCAPE_CHAR = ESCAPE_CHAR_FUNCTIONS
            elif text in self.var_fnc_mapping['var']:
                ESCAPE_CHAR = ESCAPE_CHAR_VARIABLES
            elif text in ['f', 'g', 'h', 'u', 'v', 'F', 'G', 'H', 'U', 'V']:
                if add:
                    self.var_fnc_mapping['fnc'].append(text)
                ESCAPE_CHAR = ESCAPE_CHAR_FUNCTIONS
            else:
                if add:
                    self.var_fnc_mapping['var'].append(text)
                ESCAPE_CHAR = ESCAPE_CHAR_VARIABLES
        else:
            ESCAPE_CHAR = escape

        return ESCAPE_CHAR + text + ESCAPE_CHAR

    def create_symbol(self, text: str):
        if '_' in text:
            s = text.split('_')
            base = sympy.IndexedBase(s[0])
            if len(s) > 1:
                if s[1] in ['{1,2}', '{1, 2}']:
                    return base[sympy.core.text.String('{1,2}')]
                else:
                    return base[sympy.Symbol(s[1])]
            else:
                return base
        else:
            return sympy.Symbol(text)

    def subs(self, *args, evaluate=False, simultaneous=False, recurse=True):
        new_ = deepcopy(self)
        unordered = False
        if len(args) == 1:
            sequence = args[0]
            if isinstance(sequence, set):
                unordered = True
            elif isinstance(sequence, dict):
                unordered = True
                sequence = sequence.items()
            else:
                raise ValueError(("""
                   When a single argument is passed to subs
                   it should be a dictionary of old: new pairs or an iterable
                   of (old, new) tuples."""))
        elif len(args) == 2:
            sequence = [args]
        else:
            raise ValueError("subs accepts either 1 or 2 arguments")

        if recurse and new_.recursion is not None:
            new_.recursion = new_.recursion.subs(*args, evaluate=evaluate, simultaneous=simultaneous)

        for old, new in sequence:
            for key in ['fnc', 'var']:
                l_old = latex(old)
                if l_old in new_.var_fnc_mapping[key]:
                    new_.var_fnc_mapping[key].remove(l_old)
                    l_new = latex(new)
                    if l_new not in new_.var_fnc_mapping[key]:
                        new_.var_fnc_mapping[key].append(l_new)

        if simultaneous:
            s1 = {}
            s2 = {}
            for i, s in enumerate(sequence):
                intermediate = Dummy('subsitute_%d' % i, commutative=False)
                s1[s[0]] = intermediate
                s2[intermediate] = s[1]
            return new_.subs(s1, simultaneous=False, recurse=False).subs(s2, simultaneous=False, recurse=False)
        else:
            for s in sequence:
                s0 = latex(s[0])
                s1 = latex(s[1])

                for escape in [ESCAPE_CHAR_FUNCTIONS, ESCAPE_CHAR_VARIABLES]:
                    new_t = new_.text.replace(new_.wrap_escape_char(s0, escape=escape), new_.wrap_escape_char(s1, escape=escape))
                    if new_t != new_.text:
                        new_.text = new_t
                        break


        return new_

    def getText(self):
        text = self.text
        for v in self.var_fnc_mapping['fnc'] + self.var_fnc_mapping['var']:
            for escape in [ESCAPE_CHAR_FUNCTIONS, ESCAPE_CHAR_VARIABLES]:
                text = text.replace(self.wrap_escape_char(str(v), escape=escape), str(v))
        if self.recursion is not None:
            return text + self.recursion_command + ' ' + latex(self.recursion, **recursion_randomize_settings)
        return text

    @property
    def args(self):
        if self.recursion is not None:
            return self.recursion.args
        return tuple()

    def get_arg(self, indices):
        return self.recursion.get_arg(indices)

    def set_arg(self, *args):
        return self.recursion.set_arg(*args)

    def __str__(self):
        return self.getText()

    def __contains__(self, item):
        if isinstance(item, sympy.Basic):
            if self.recursion is not None:
                return item in self.recursion
            return False
        elif isinstance(item, str):
            return item in self.__str__()
        else:
            return False
