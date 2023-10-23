import re
from copy import deepcopy

from data.formula.settings import randomize_settings
from sympy import latex, SympifyError, Basic
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.logic import LogicFormula
from tools import timeout

DOLLAR_ESCAPE = '####'

TIMEOUT = 20

def set_timeout(timeout):
    global TIMEOUT
    TIMEOUT = timeout

class LaTeXText:
    def __init__(self, text:str):
        self.raw = text
        self.text = []
        self.id = None
        self.substitutable = True
        if any(x in text for x in ['begin{equation', 'begin{align']):
            self.substitutable = False

        self.initialized = False

    def check_init(self):
        if not self.initialized:
            if __debug__:
                r = self.init()
            else:
                r = timeout(TIMEOUT)(self.init)()
            if r:
                raise r
            self.initialized = True

    def init(self):
        #text = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', self.raw) # replace old displaymode style $$ ...$$ with \[...\]
        text = re.sub(r'\$\$(.*?)\$\$', r'$\1$', self.raw) # replace old displaymode style $$ ...$$ with \[...\]
        if r'\[' in text:
            self.substitutable = False

        if r'\$' in text:
            text = text.replace(r'\$', DOLLAR_ESCAPE)

        for i, t in enumerate(text.split('$')):
            if i % 2 == 0:
                self.text.append(t)
            else:
                try:
                    self.text.append(parse_latex(t))
                except NotImplementedError as e:
                    self.text.append(t)
                    print(e)
                    print(t)
                    self.substitutable = False

    def has_formula_count(self, n):
        return self.raw.count('$') >= 2 * n

    def __contains__(self, item):
        if isinstance(item, (Basic, LaTeXText, LogicFormula)):
            try:
                return any(item in f for f in self.formula_iterator())
            except Exception:
                return False
        elif isinstance(item, type):
            for arg in self.formula_iterator():
                if isinstance(arg, item):
                    return True
                if hasattr(arg, 'func') and isinstance(arg.func, item):
                    return True
        elif isinstance(item, str):
            return item in self.__str__()

        try:
            for a in self.args:
                if hasattr(a, '__contains__') and a is not self:
                    try:
                        if item in a:
                            return True
                    except TypeError:
                        pass
        except SympifyError:
            pass

        return False

    def subs(self, *args, **kwargs):
        if not self.substitutable:
            old = kwargs.get('old', None)
            if old and any(str(o) in self.getText() for o in old):
                raise ValueError("This text is not substitutable: %s" % self.__str__())
            return self

        new_text = []
        for t in self.text:
            if not isinstance(t, str):
                new_text.append(t.subs(*args, **kwargs))
            else:
                new_text.append(t)
        new_ = deepcopy(self)
        new_.text = new_text
        return new_

    def getText(self):
        self.check_init()
        return '$'.join(t if isinstance(t, str) else latex(t, **randomize_settings) for t in self.text).replace(DOLLAR_ESCAPE, r'\$')

    def formula_iterator(self):
        self.check_init()
        for t in self.text:
            if not isinstance(t, str):
                yield t

    def get_formulas(self, as_string=False):
        # dummy parameter needed for compatibility with tools.Text
        return list(self.formula_iterator())

    @property
    def free_symbols(self):
        s = set()
        for t in self.formula_iterator():
            s.update(t.free_symbols)
        return s

    @property
    def args(self):
        return tuple(self.formula_iterator())

    def get_arg(self, indices):
        if len(indices) == 1:
            return self.args[indices[0]]

        return self.args[indices[0]].get_arg(indices[1:])

    def set_arg(self, indices, new_element):
        if not isinstance(indices, (list, tuple)):
            indices = [indices]

        ctr = -1
        for i, t in enumerate(self.text):
            if not isinstance(t, str):
                ctr += 1

                if ctr == indices[0]:
                    if len(indices) == 1:
                        self.text[i] = new_element
                    else:
                        t.set_arg(indices[1:], new_element)
                    return

        raise ValueError("Got indices %s for texts %s" % (indices, self.text))

    def __str__(self):
        return self.getText()

    def __repr__(self):
        return str(self)

    def ___eq__(self, other):
        if not isinstance(other, LaTeXText):
            return False

        if len(self.text) != len(other.text):
            return False

        self.check_init()

        return all(text == o for text, o in zip(self.text, other.text))