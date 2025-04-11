

import functools
import time
import itertools
import json
import math
import random
import re
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from pickle import PicklingError
from threading import Thread

from frozendict import frozendict

import sympy
from sympy import Equality, evaluate, S, latex, Basic, Symbol, Number
from sympy.core.function import DerivativeError
from sympy.parsing.latex import parse_latex

from sympy.printing.latex import random_latex
import uuid
import multiprocessing

from multiprocessing import Pool

from enum import Enum

from sympy.parsing.latex.logic import StringFormula
from sympy.parsing.latex.text import LaTeXText
from sympy.printing.stats import PermutationStats
from sympy.utilities.timeutils import timeout


#https://stackoverflow.com/questions/1151658/python-hashable-dicts
class hashabledict(dict):
  def __key(self):
    return tuple((k,self[k]) for k in sorted(self))
  def __hash__(self):
    return hash(self.__key())
  def __eq__(self, other):
    return self.__key() == other.__key()

def create_sympy(x):
    if isinstance(x, int):
        return sympy.Integer(x)
    return sympy.Symbol(x)

def substitute(expr, permutation: dict):
    from sympy.core.function import UndefinedFunction
    function_permutation = {k:v for k,v in permutation.items() if isinstance(k, UndefinedFunction)}
    other_permutation = {k:v for k,v in permutation.items() if not isinstance(k, UndefinedFunction)}
    if not function_permutation.keys().isdisjoint(function_permutation.values()):
        raise ValueError("Not supported Substituteion: %s" % function_permutation)


    with evaluate(False):
        expr = expr.subs(other_permutation, evaluate=False, simultaneous=True)
        expr = expr.subs(function_permutation, evaluate=False, simultaneous=False)
    return expr

def sympy_copy(expr, allow_none=False):
    import sympy
    if isinstance(expr, sympy.Expr):
        try:
            from sympy import evaluate
            with evaluate(False):
               return expr.func(*[sympy_copy(arg) for arg in expr.args], evaluate=False)
        except Exception:
            if allow_none:
                return None
    return deepcopy(expr)


max_fnc = max

class Strategy(Enum):
    EQUALITY = 'equality'
    INEQUALITY = 'inequality'
    RANDOM_FORMULA = 'random_formula'
    SWAP = 'swap'
    VARIABLES = 'variables'
    CONSTANTS = 'constants'
    DISTRIBUTE = 'distribute'
    TEXT = 'text'

ALL_VARIABLES = frozenset({
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\varepsilon', r'\zeta', r'\eta', r'\theta',
    r'\vartheta', r'\lambda', r'\mu', r'\nu', r'\xi', r'\rho', r'\sigma', r'\tau', r'\phi', r'\varphi',
    r'\chi', r'\psi', r'\omega',
})

class FormulaGenerator:

    # defines cluster of variables that are typically used within a similar context
    variable_cluster = [frozenset({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}),  # parameter
                        frozenset({'i', 'j', 'k', 'l'}),  # indices
                        frozenset({'m', 'n', 'k', 'l'}),  # number of elements
                        frozenset({'p', 'q', 'r', 's', 't'}),  # parameter
                        frozenset({'u', 'v', 'w'}),  # parameter/ vectors
                        frozenset({'x', 'y', 'z'}),  # unknowns
                        frozenset({'\\alpha', '\\beta', '\\gamma', '\\delta', '\\theta', '\\vartheta', '\\psi', '\\phi', '\\rho', '\\varphi'}),  # angles
                        frozenset({'\\tau', '\\sigma', '\\lambda', '\\mu', '\\nu'}), # scalars, permutations
                        frozenset({'X', 'Y', 'Z', 'A', 'B', 'C'}), # random variables
                        frozenset({'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}), # Matrices, Sets
                        frozenset({'Q', 'R', 'T', 'U', 'X', 'Y', 'Z', 'W', 'V'}) # random variables
                        ]
    # defines cluster of functions that are typically used within a similar context
    function_cluster = [frozenset({'f', 'g', 'h', 'u', 'v'}),
                        frozenset({'F', 'G', 'H', 'U', 'V'}),
                        frozenset({'\\tau', '\\sigma', '\\lambda', '\\mu', '\\nu'}),
                        frozenset({'\\psi', '\\phi', '\\rho', '\\varphi'})]

    additional_cluster_entries = {
        r'\alpha': ['y', 'z'],
        r'\beta': ['y', 'z'],
    }

    # defines clusters of variables describing the same letter in different alphabets and cases
    style_cluster = [
        {'lower': 'a', 'upper': 'A', 'greek_lower': '\\alpha'},
        {'lower': 'b', 'upper': 'B', 'greek_lower': '\\beta'},
        {'lower': 'c', 'upper': 'C', 'greek_lower': '\\gamma'},
        {'lower': 'd', 'upper': 'D', 'greek_lower': '\\delta'},
        {'lower': 'f', 'upper': 'F'},
        {'lower': 'g', 'upper': 'G'},
        {'lower': 'h', 'upper': 'H'},
        {'lower': 'u', 'upper': 'U'},
        {'lower': 'v', 'upper': 'V'},
        {'lower': 'x', 'upper': 'X'}
    ]


    """
    Creates a FormulaGenerator with the provided data object.

    data can be either a str representing a formula, a sympy.parsing.latex.text.LaTeXText or python
    dict (requires key "formula" with string formula, more properties can be set explicitly, including "variables" for
    substitutable variables (e.g. "x"), and "functions" for substitutable functions (e.g. "f").
    """

    def __init__(self, data, additional_names=None, id=None, factor_false=1, min_substitution=0, force_substitution=False, max_strategies=None):
        if data is None:
            return

        if additional_names is None:
            additional_names = []

        self.data = data
        self.id = id
        self.max_strategies = max_strategies
        self.indexed = {}
        self.special_variables_candidates = {}
        self.text_substitutions = {}
        self.ids = []

        self.names = additional_names if len(additional_names) > 0 else [None]
        if isinstance(data, str):
            self.formula = data
            with evaluate(False):
                self.current_tex = parse_latex(data)
            self.variables = set()
            for v in self.current_tex.free_symbols:
                if isinstance(v, sympy.Indexed):
                    if isinstance(v.base, sympy.IndexedBase) and len(str(v.base)) == 0:
                        self.variables.update([s for s in v.free_symbols if isinstance(s, sympy.Symbol)])
                    else:
                        self.indexed[v.base] = v
                else:
                    if not 'text' in str(v):
                        self.variables.add(v)
            self.variables = list(self.variables)
            self.functions = set()
            it = sympy.postorder_traversal(self.current_tex)

            for sub_expr in it:
                if isinstance(sub_expr, sympy.core.function.UndefinedFunction):
                    self.functions.add(sub_expr)
                elif hasattr(sub_expr, 'func') and isinstance(sub_expr.func, sympy.core.function.UndefinedFunction):
                    self.functions.add(sub_expr.func)
                elif isinstance(sub_expr, StringFormula):
                    for f in sub_expr.var_fnc_mapping['fnc']:
                        self.functions.add(sympy.Function(f))
            self.functions = list(self.functions)

            self.texs = {self.current_tex}
            self.ids.append(uuid.uuid4())
        elif self._is_text_expr(data): # e.g., LaTeXText
            self.current_tex = data
            self.texs = {self.current_tex}
            self.ids.append(uuid.uuid4())
            self.formula = latex(data)

            self.functions = set()
            functions_str = set()
            for formula in self.current_tex.get_formulas(as_string=False):
                for sub_expr in sympy.postorder_traversal(formula):
                    if isinstance(sub_expr, sympy.core.function.UndefinedFunction):
                        self.functions.add(sub_expr)
                        functions_str.add(str(sub_expr))
                    elif hasattr(sub_expr, 'func') and isinstance(sub_expr.func, sympy.core.function.UndefinedFunction):
                        self.functions.add(sub_expr.func)
                        functions_str.add(str(sub_expr.func))
            self.functions = list(self.functions)

            self.variables = set()
            for v in self.current_tex.free_symbols:
                if isinstance(v, sympy.Indexed):
                    self.indexed[v.base] = v
                else:
                    self.variables.add(v)
            self.variables = list(self.variables)
        else:
            if 'formula' not in data:
                print("Warning: no formula given for %s" % data)
                raise KeyError

            self.formula = data['formula']
            self.variables = data.get('variables', [])
            self.variables = [sympy.Symbol(v) for v in self.variables]
            self.functions = data.get('functions', [])
            self.functions = [sympy.core.function.UndefinedFunction(f) for f in self.functions]
            self.special_variables_candidates = data.get('special_variable_candidates', {})
            if isinstance(self.special_variables_candidates, list):
                c = set(self.special_variables_candidates)
                self.special_variables_candidates = {}
                for v in self.variables:
                    self.special_variables_candidates[v] = c
            elif isinstance(self.special_variables_candidates, dict):
                for v in self.variables:
                    if v in self.special_variables_candidates:
                        self.special_variables_candidates[v] = set(self.special_variables_candidates[v])
                    else:
                        self.special_variables_candidates[v] = set()

            self.names = data.get('names', []) + additional_names
            with evaluate(False):
                try:
                    self.current_tex = parse_latex(self.formula)
                except Exception as e:
                    print("An error occurred parsing <%s>" % self.formula)
                    raise e
                self.texs = [self.current_tex]
                self.ids.append(uuid.uuid4())
                self.text_substitutions = data.get('text_substitution', {})

                formulas = {self.formula}
                for subset in self.__get_all_subsets(self.text_substitutions):
                    if len(subset) > 0:
                        formula = self.formula
                        for k, v in subset.items():
                            formula = formula.replace(k, v)
                        if formula not in formulas:
                            tex = parse_latex(formula)
                            self.texs.append(tex)
                            self.ids.append(uuid.uuid4())
                            formulas.add(formula)

        self.texs = list(self.texs)
        self.original_id = self.ids[0]
        iv = self.initial_version()
        self.versions = {iv[0]: iv[1]}
        self.current_stats = self._create_permutation_stats()
        self.similar_formulas = {}
        self.strategies = [
            Strategy.EQUALITY,
            Strategy.INEQUALITY,
            Strategy.RANDOM_FORMULA,
            Strategy.SWAP,
            Strategy.VARIABLES,
            Strategy.CONSTANTS,
            Strategy.DISTRIBUTE
        ]

        self.all_known_variables = ALL_VARIABLES
        self.create_style_cluster_mapping()
        self.random_formula = None
        self.no_versions = []
        self.returned_initial_version = False
        self.max_none = 100
        self.none_counter = 0
        self.not_allowed_names = []
        self.factor_false = factor_false
        self.min_substitution = min_substitution
        self.force_substitution = force_substitution
        self.random_small_formula = None
        self.random_large_formula = None

    def __get_all_subsets(self, dictionary):
        if len(dictionary) == 0:
            yield frozendict({})
        else:
            subsets = set()
            for key in dictionary.keys():
                dictionary_copy = dictionary.copy()
                value = dictionary_copy[key]
                del dictionary_copy[key]
                if not isinstance(value, list):
                    value = [value]

                for subset in self.__get_all_subsets(dictionary_copy):
                    if len(subset) > 0 and subset not in subsets:
                        subsets.add(subset)
                        yield subset
                    subset = dict(subset)
                    for v in value:
                        subset[key] = v
                        s = frozendict(subset)
                        if s not in subsets:
                            subsets.add(s)
                            yield s

    def copy_tex(self):
        tex = self._random_current_tex()
        with evaluate(False):
            c = sympy_copy(tex)
            self.current_tex = c
            return self.current_tex

    def amounts(self, positive_version: bool = None):
        if positive_version is None:
            return len(self.versions)
        return len([v for v in self.versions if v[1] == positive_version])

    def update_similar_formulas(self, formulas: dict):
        self.similar_formulas = formulas.get(self.formula)

    def update_no_versions(self, no_versions):
        self.no_versions = no_versions

    def update_random_formula(self, random_formula):
        self.random_formula = random_formula

    def create_style_cluster_mapping(self):
        mapping = {}
        for c in self.style_cluster:
            for k in c.values():
                mapping[k] = hashabledict(c)
        self.style_cluster_mapping = mapping

    def _is_text_expr(self, expr):
        return hasattr(expr, 'get_formulas')

    def get_small_formulas(self):
        if self._is_text_expr(self.current_tex):
            formulas = set(self.current_tex.args)
        elif isinstance(self.current_tex, Basic):
            formulas = {self.current_tex}
        else:
            formulas = {}

        result = set()
        for f in formulas:
            if len(latex(f)) < 20 and not isinstance(f, (Symbol, Number)):
                try:
                    s = sympy_copy(f, allow_none=True)
                    if s is not None:
                        result.add(s)
                except Exception:
                    pass
        return result

    def get_large_formulas(self):
        if self._is_text_expr(self.current_tex):
            formulas = set(self.current_tex.args)
        elif isinstance(self.current_tex, Basic):
            formulas = {self.current_tex}
        else:
            formulas = {}

        result = set()
        for f in formulas:
            if len(latex(f)) < 20:
                continue
            try:
                s = deepcopy(f)
                result.add(s)
            except Exception:
                pass

        return result

    def generate_versions(self, max=100, reset=False, only_true_version=False, strategies=None):
        for _ in self.generate_versions_iterator(max=max, reset=reset, only_true_version=only_true_version,
                                                 strategies=strategies):
            pass

    def initial_version(self):
        return (self.formula, True), self._create_permutation_stats()

    def generate_versions_iterator(self, max=100, max_none=None, reset=False, only_true_version=False,
                                   return_stats=False, strategies=None, initial_is_candidate=True, max_time=10):
        if reset:
            iv = self.initial_version()
            self.versions = {iv[0]: iv[1]}
            self.returned_initial_version = False
            self.max_none = 100
            self.none_counter = 0
        max_none_ = max_none or self.max_none

        if not self.returned_initial_version and initial_is_candidate:
            if return_stats:
                yield self.initial_version()
            else:
                yield self.initial_version()[0]

            self.returned_initial_version = True
            max = max - 1

        for i in range(max):
            try:
                use_timeout = False
                if use_timeout:
                    res = timeout(max_time)(lambda: self.generate_random_version(only_true_version=only_true_version,
                                                                           strategies=strategies))()
                    if isinstance(res, Exception):
                        raise res
                    elif len(res) == 2:
                        version, stats = res
                    else:
                        print("Warning: An error occurred during generating a random version for formula <%s>: %s" % (
                        self.formula, res))
                        raise ValueError
                else:
                    version, stats = self.generate_random_version(only_true_version=only_true_version,
                                                                  strategies=strategies)

                if version not in self.versions and version[0] is not None:
                    self.versions[version] = stats
                    # stop early if no new version has been found for a long time
                    self.max_none = max_fnc(100, 2 * len(self.versions))
                    if max_none is None:
                        max_none_ = self.max_none
                    self.none_counter = 0
                    if return_stats:
                        yield version, stats
                    else:
                        yield version
                else:
                    self.none_counter += 1
                    if self.none_counter >= max_none_:
                        return
            except Exception as e:
                print("Warning: An error occurred generating versions for formula <%s>: %s" % (self.formula, e))

    def generate_random_version(self, only_true_version=False, strategies=None):
        try:
            amount_true = self.amounts(True)
            amount_false = self.amounts(False)

            if amount_true + amount_false == 0:
                # both are 0
                amount_true = 1
                amount_false = 1

            # prefer to generate a version that is underrepresented (yet)
            if only_true_version or (
                amount_true == 1 and amount_false == 1) or amount_false >= self.factor_false * amount_true:  # or (amount_false >= amount_true and self._random(0.1)
                tex, stats = self._generate_random_true_version()
                b = True
            else:
                tex, stats = self._generate_random_false_version(strategies=strategies)
                b = False
            return ((tex, b), stats)
        except Exception as e:
            print("Error generating a version: %s" % e)
            return ((None, None), None)

    def _get_random_subset(self, dictionary):
        keys = list(dictionary.keys())
        k = random.randint(0, len(dictionary) + 1)
        random_keys = random.sample(keys, k)
        subset = {key: dictionary[key] for key in random_keys}
        return subset

    def _create_permutation_stats(self, set_current=False):
        stats = PermutationStats()
        stats['formula_name_id'] = self.id
        stats['original_id'] = self.original_id
        stats['is_text'] = isinstance(self.current_tex, LaTeXText)
        if set_current:
            self.current_stats = stats
        return stats

    def _generate_random_true_version(self, expr=None, is_false=False):
        if expr is None:
            expr = self._random_current_tex(set_id=True)
        self.current_tex = expr
        self._create_permutation_stats(set_current=True)
        try:
            self.current_stats['original'] = self._print_expr(self.current_tex)
            self.current_stats['original_id'] = self.original_id
        except ValueError as e:
            print("Could not print current tex")
            print(e)
            return False

        n = len(self.variables)
        k = random.randint(max(0, min(self.min_substitution, n)), n)
        permuting_variables = random.sample(self.variables, k)
        fixed_variables = [v for v in self.variables if v not in permuting_variables]

        n = len(self.functions)
        k = random.randint(max(0, min(self.min_substitution, n)), n)
        permuting_functions = random.sample(self.functions, k)
        fixed_functions = [v for v in self.functions if v not in permuting_functions]

        free_symbols = []
        all_var = permuting_variables + fixed_variables
        try:
            if hasattr(expr, 'free_symbols'):
                for x in list(expr.free_symbols):
                    try:
                        if x not in all_var:
                            x = str(x)
                            free_symbols.append(x)
                    except Exception:
                        pass
        except Exception:
            pass

        permutation = self._generate_permutation(permuting_variables, fixed_variables, permuting_functions,
                                                 fixed_functions, free_symbols)

        if len(permutation) > 0:
            substitution_var = {}
            substitution_fnc = {}
            for k, v in permutation.items():
                if k in permuting_functions:
                    if hasattr(k, 'func') and hasattr(k.func, '__name__'):
                        k_key = k.func.__name__
                    else:
                        k_key = k.__name__
                    if hasattr(v, 'func') and hasattr(v.func, '__name__'):
                        v_key = v.func.__name__
                    else:
                        v_key = v.__name__

                    substitution_fnc[k_key] = v_key
                else:
                    substitution_var[k.name] = v.name

            self.current_stats['substitution_var'] = substitution_var
            self.current_stats['substitution_fnc'] = substitution_fnc
            try:
                version = substitute(expr, permutation)
            except DerivativeError as e:
                if not is_false:
                    print(
                        "A DerivativeError occurred during the substitution <%s> when generating a true version!!!" % e)
                    raise e
                version = expr
            except Exception as e:
                if 'Limits approaching a variable ' not in str(e):
                    print("An error occurred during the substitution <%s> when generating a %s version" % (
                    e, not is_false))
                    try:
                        print(substitution_fnc)
                        print(substitution_var)
                        print(expr)
                        print(self.current_stats)
                    except Exception:
                        pass
                    raise e
                return None, None
        elif self.force_substitution:
            raise ValueError("Substitution empty, but force permutation")
        else:
            version = expr

        try:
            tex, tex_stats = random_latex(version, return_stats=True)
            return tex, self.current_stats.union(tex_stats)
        except Exception as e:
            print("Error when creating random latex for %s" % version)
            print(e)
            return None, None

    def _generate_permutation(self, permuting_variables, fixed_variables, permuting_functions, fixed_functions,
                              exclude):
        fixed_symbols = fixed_variables + fixed_functions

        permuting_variables_candidates = {v: self._get_variable_permutation_candidates(v, False, exclude=exclude) for v
                                          in
                                          permuting_variables}
        permuting_functions_candidates = {f: self._get_variable_permutation_candidates(f, True, exclude=exclude) for f
                                          in
                                          permuting_functions}

        permuting_candidates = {**permuting_variables_candidates, **permuting_functions_candidates}
        permuting_symbols = set(permuting_variables + permuting_functions)
        # todo shuffle?

        all_symbols = set(permuting_variables + fixed_variables + permuting_functions + fixed_functions + exclude)

        # check if there exist variables dependencies, like f + F -> should only be changed consistently e.g. to h + H
        dependencies = {}
        clusters = []

        for v in all_symbols:
            cluster = self.style_cluster_mapping.get(v)
            if cluster is not None:
                if cluster in dependencies.values():
                    # this is a potential conflict
                    dependencies[v] = cluster

                    conflicted_variables = set([key for key, value in dependencies.items() if value == cluster])
                    if all([v in permuting_symbols for v in conflicted_variables]):
                        clusters.append(conflicted_variables)
                    elif all([v not in permuting_symbols for v in conflicted_variables]):
                        pass
                    else:
                        if self._random():
                            # remove the values from permuting symbols
                            for c in cluster.values():
                                if c in permuting_symbols:
                                    permuting_symbols.remove(c)
                        else:
                            # add the missing element to the permuting symbols
                            clusters.append(conflicted_variables)
                            for c in cluster.values():
                                if c not in permuting_symbols:
                                    if c in fixed_functions:
                                        permuting_candidates[c] = self._get_variable_permutation_candidates(c, True)
                                    elif c in fixed_variables:
                                        permuting_candidates[c] = self._get_variable_permutation_candidates(c, False)
                else:
                    dependencies[v] = cluster

        # create permutation
        permutation = {}

        # 1. create permutation for conflicted variables
        # context: used-variables, all_variables-> determine f, F,
        for conflicted_cluster in reversed(clusters):
            cluster_candidates = []

            cluster_entry = dependencies[list(conflicted_cluster)[0]]

            for i, v in enumerate(conflicted_cluster):
                if v in permuting_candidates:
                    candidates = permuting_candidates[v]
                    candidates = set([c for c in candidates if c not in list(permutation.values()) + fixed_symbols])
                else:
                    break

                for cluster in self.style_cluster:
                    if len(candidates.intersection(cluster.values())) > 0:
                        if i == 0:
                            cluster_candidates.append(cluster)
                    else:
                        if cluster in cluster_candidates:
                            cluster_candidates.remove(cluster)

            cluster_candidates = [c for c in cluster_candidates if set(cluster_entry.keys()).issubset(c.keys())]

            if len(cluster_candidates) > 0:
                cluster = random.choice(cluster_candidates)
                for k, v in cluster_entry.items():
                    if v in conflicted_cluster:
                        c = cluster[k]
                        if c in permuting_variables:
                            permutation[v] = create_sympy(c)
                        else:
                            permutation[v] = sympy.Function(c)

        # 2. create special permutations for similar variables/functions: x,y -> z_1, z_2
        if sympy.Indexed not in self.current_tex:
            special_permutation_variable_candidates = {}
            special_permutation_function_candidates = {}
            pair_iterator = lambda list: [(l1, l2) for l1 in list for l2 in list if l1 != l2]
            for x1, x2 in pair_iterator(
                [v for v in permuting_variables if v not in permutation and v not in self.indexed]):
                for cluster in self.variable_cluster:
                    if str(x1) in cluster and str(x2) in cluster:
                        if cluster in special_permutation_variable_candidates:
                            special_permutation_variable_candidates[cluster].union({x1, x2})
                        else:
                            special_permutation_variable_candidates[cluster] = {x1, x2}

            for x1, x2 in pair_iterator([v for v in permuting_functions if v not in permutation]):
                for cluster in self.function_cluster:
                    if self._strip_function(x1) in cluster and self._strip_function(x2) in cluster:
                        if cluster in special_permutation_function_candidates:
                            special_permutation_function_candidates[cluster].union({x1, x2})
                        else:
                            special_permutation_function_candidates[cluster] = {x1, x2}

            for cluster, symbols in special_permutation_variable_candidates.items():
                cluster = [c for c in cluster if c not in [str(s) for s in fixed_symbols + fixed_functions + list(
                    permutation.keys())] and all(c in permuting_candidates[s] for s in symbols)]
                if self._random(probability_true=0.2) and len(cluster) > 0:

                    # perform this permutations
                    c = [x for x in cluster if x not in fixed_symbols]
                    if len(c) > 0:
                        target_symbol = random.choice(c)  # even an already used variable is ok
                        symbols = list(symbols)
                        random.shuffle(symbols)
                        for i, s in enumerate(symbols):
                            permutation[s] = sympy.Indexed(target_symbol, i + 1)

            for cluster, symbols in special_permutation_function_candidates.items():
                cluster = [c for c in cluster if c not in [str(s) for s in fixed_symbols + fixed_functions + list(
                    permutation.keys())] and all(c in permuting_candidates[s] for s in symbols)]
                if self._random(probability_true=0.3) and len(cluster) > 0:
                    # perform this permutations
                    target_symbol = random.choice(list(cluster))  # even an already used variable is ok
                    symbols = list(symbols)
                    random.shuffle(symbols)
                    for i, s in enumerate(symbols):
                        if s in self.indexed:
                            s = self.indexed[s]
                        permutation[s] = sympy.Function('%s_%s' % (target_symbol, i + 1))

        # 3. create permutation for other variables
        for s in permuting_symbols:
            if s not in permutation and s in permuting_candidates:
                candidates = permuting_candidates[s]
                candidates = [c for c in candidates if c not in [str(v) for v in
                                                                 list(permutation.values()) + fixed_symbols + list(
                                                                     permuting_symbols)]]
                if len(candidates) > 0:
                    c = random.choice(candidates)
                    if s in permuting_variables:
                        permutation[s] = create_sympy(c)
                    else:
                        permutation[s] = sympy.Function(c)

        return permutation

    def _strip_function(self, f):
        f = str(f)
        if '_' in f:
            # this is an indexed function
            f = re.match(r"^(.*?)_", f).group(1)
        return f

    def _get_variable_permutation_candidates(self, symbol, is_function=False, exclude=None):
        if exclude is None:
            exclude = []
        symbol = str(symbol)
        candidates = set()
        cluster = self.function_cluster if is_function else self.variable_cluster
        if is_function:
            symbol = self._strip_function(symbol)

        for symbols in cluster:
            if symbol in symbols:
                for s in symbols:
                    if symbol != s:
                        candidates.add(s)

        # x is always a candidate for a variable
        if not is_function and symbol != 'x':
            candidates.add('x')

        if symbol in self.additional_cluster_entries:
            for v in self.additional_cluster_entries[symbol]:
                candidates.add(v)

        # add randomly any other variables
        # probability is determined such that if only few candidates are available, the probability is low
        prob = 1 - 0.8 * math.exp(-len(candidates) / 50)
        if not is_function and self._random(probability_true=prob):
            candidates.add(self._get_random_variable(prefer_used_var=False, include_numbers=False))

        if symbol in self.special_variables_candidates:
            for s in self.special_variables_candidates[symbol]:
                if s != symbol:
                    candidates.add(s)

        if 'e' in candidates and self._contains_pow():
            candidates.remove('e')

        if any(i in self.current_tex for i in [sympy.I, sympy.core.numbers.ICandidate()]) and 'i' in candidates:
            candidates.remove('i')

        if 'E' in candidates and any(c in self.current_tex for c in
                                     [sympy.stats.Expectation, sympy.stats.Variance, sympy.stats.Covariance,
                                      sympy.stats.ConditionalProbability, sympy.stats.BasicProbability]):
            candidates.remove('E')

        for e in exclude:
            if e in candidates:
                candidates.remove(e)

        return candidates

    def _contains_pow(self):
        return self.__contains_pow(self.current_tex)

    def __contains_pow(self, expr):
        if isinstance(expr, sympy.Pow):
            return True
        elif hasattr(expr, 'args'):
            return any(self.__contains_pow(arg) for arg in expr.args)
        else:
            return False

    def _get_random_binary_operator(self):
        # actually 3 arguments are expected, since the 3rd one is "evaluate=False"

        def minus(x, y, evaluate=False):
            return sympy.Add(x, -y, evaluate=evaluate)

        def div(x, y, evaluate=False):
            return sympy.Mul(x, 1 / y, evaluate=evaluate)

        return random.choices([
            sympy.Add,
            minus,
            sympy.Mul,
            div,
            sympy.Pow
        ], weights=[10, 5, 10, 5, 2])[0]

    def random_name(self, not_allowed_names=None):
        names = self.names
        if not_allowed_names:
            names = random.choice([n for n in self.names if n not in not_allowed_names])

        if len(names) > 0:
            return random.choice(names)
        return None

    def get_random_version(self, return_stats=False, only_true_version=False):
        if only_true_version:
            keys = [k for k in self.versions.keys() if k[1]]
        else:
            keys = list(self.versions.keys())

        if len(keys) > 0:
            random_version = random.choice(keys)
            stats = self.versions[random_version]
            name = self.random_name(self.not_allowed_names)
            if name:
                if return_stats:
                    return name, random_version, stats
                return name, random_version
        else:
            self.generate_versions(max=1, only_true_version=only_true_version)
            if len(self.versions) > 0:
                name = self.random_name()
                version, stats = random.choice(list(self.versions.items()))
                if return_stats:
                    return name, version, stats
                return name, version
        if return_stats:
            return None, None, None
        return None, None

    def _get_random_strategies(self):
        l = len(self.strategies)
        if self.max_strategies and self.max_strategies < l:
            l = self.max_strategies

        k = random.randint(1, l)
        return random.choices(self.strategies, k=k)

    def _random_current_tex(self, set_id=False):
        if set_id:
            idx = random.randint(0, len(self.texs) - 1)
            tex = self.texs[idx]
            id = self.ids[idx]
            self.original_id = id
            return tex
        return random.choice(self.texs)

    def _generate_random_false_version(self, strategies=None):
        if not strategies:
            strategies = self._get_random_strategies()
        result = self.copy_tex()

        success = False
        stats = self._create_permutation_stats(set_current=True)
        self.current_stats['original'] = self._print_expr(self.current_tex)

        if set(strategies) == {Strategy.TEXT, Strategy.RANDOM_FORMULA}:
            strategies = [random.choice(strategies)]

        if Strategy.RANDOM_FORMULA in strategies:
            success |= self._strategy_random_formula()
            result = self.current_tex
        elif Strategy.TEXT in strategies:
            for i, formula in enumerate(result.get_formulas()):
                new_formula = self._replace_formula(formula)
                result.set_arg(i, new_formula)
                success = True

        if Strategy.CONSTANTS in strategies:
            success |= self._strategy_constants(result)
            result = self.current_tex

        if Strategy.EQUALITY in strategies:
            success |= self._strategy_equality(result)

        if Strategy.INEQUALITY in strategies:
            success |= self._strategy_inequality(result)

        if Strategy.SWAP in strategies:
            success |= self._strategy_swap(result)

        if Strategy.VARIABLES in strategies:
            success |= self._strategy_variables(result)

        if Strategy.DISTRIBUTE in strategies:
            success |= self._strategy_distribute(result)

        if success:
            try:
                if self._random():
                    # additionally change some variables
                    stats = self.current_stats
                    tex, tex_stats = self._generate_random_true_version(result, is_false=True)
                    if tex is not None:
                        return tex, stats.union(tex_stats)

                    # todo merge stats
                tex, tex_stats = random_latex(result, return_stats=True)
                return tex, stats.union(tex_stats)
            except Exception as e:
                print("Error in random false version: %s" % e)

        return None, None

    def _replace_formula(self, formula):
        if isinstance(formula, Symbol):
            return self._get_random_variable(return_symbol=True, prefer_used_var=False)
        elif isinstance(formula, Number):
            return self._random_number()
        else:
            try:
                s = latex(formula)
                if len(s) <= 20:
                    return self.random_small_formula()
                return self.random_large_formula()
            except Exception:
                return self.random_small_formula()

    def _random(self, probability_true: float = None) -> bool:
        if probability_true:
            return random.random() < probability_true
        return bool(random.getrandbits(1))

    def _strategy_equality(self, expr, depth=0, max_depth=3):
        if isinstance(expr, Equality):
            indices = self._get_small_subexpression_indices(expr)
            if self._random(probability_true=0.8) or len(indices) == 0:
                # add/ multiply something to selected side
                subexpression = self._random()
                if subexpression:
                    # element from expression
                    new_element = self._random_subexpression(expr)
                else:
                    # arbitrary new element
                    new_element = self._random_expression()

                if new_element:
                    if len(indices) == 0:
                        # best we can do is to add to one side using a random binary operator (like +, *, -, /, ^)
                        creator = self._get_random_binary_operator()
                        side = int(self._random())
                        side_expr = expr.args[side]
                        if not (
                            self._is_inequality(side_expr) or isinstance(side_expr, sympy.core.relational.Relational)):
                            new_expr = creator(side_expr, new_element, evaluate=False)
                            expr.set_arg(side, new_expr)
                            self.current_stats['strategy_equality'] = {'old': self._print_expr(side_expr),
                                                                       'new': self._print_expr(new_expr),
                                                                       'subexpression': subexpression,
                                                                       'indices': indices}
                            return True
                        return False
                    else:
                        # substitute something within the expression
                        indices = random.choice(indices)
                        operator = self._get_random_binary_operator()
                        with evaluate(False):
                            arg = sympy_copy(expr.get_arg(indices))
                        new_element_ = operator(new_element, arg, evaluate=False) if self._random() else operator(arg,
                                                                                                                  new_element,
                                                                                                                  evaluate=False)
                        expr.set_arg(indices, new_element_)
                        try:
                            new_ = self._print_expr(new_element_)
                        except RecursionError as e:
                            return False
                        try:
                            # check if current tex is still valid expression
                            str(self.current_tex)
                        except Exception as e:
                            print("Something invalid happened during strategy_equality (%s)" % e)
                            raise e
                        self.current_stats['strategy_equality'] = {'old': self._print_expr(arg), 'new': new_,
                                                                   'subexpression': subexpression, 'indices': indices}
                        return True
                else:
                    return False

            else:
                # remove something from one side
                if len(indices) > 0:
                    indices = random.choice(indices)
                    current = expr
                    for i in indices[:-2]:
                        current = current.args[i]
                    if len(current.args) > 0:
                        old = self._print_expr(current)
                        old_element = current.args[indices[-2]]
                        if isinstance(old_element, (sympy.Tuple, sympy.core.BasicVector)):
                            return False
                        new_element = old_element.args[indices[-1]]
                        current.set_arg(indices[-2], new_element)
                        self.current_stats['strategy_equality'] = {'old': old, 'new': self._print_expr(current),
                                                                   'subexpression': True, 'indices': indices}
                    else:
                        # replace the symbol
                        symbol = self._get_random_variable(True)
                        old = self._print_expr(current)
                        current.set_arg(0, symbol)
                        self.current_stats['strategy_equality'] = {'old': old, 'new': self._print_expr(current),
                                                                   'subexpression': False, 'indices': indices}
                else:
                    return False

            return True
        elif depth < max_depth:
            try:
                if hasattr(expr, 'args'):
                    args = list(expr.args)
                    random.shuffle(args)
                    for arg in args:
                        success = self._strategy_equality(arg, depth=depth + 1, max_depth=max_depth)
                        if success:
                            return True
            except Exception:
                pass
        return False

    def _print_expr(self, expr):
        return latex(expr)

    def _is_inequality(self, expr):
        return isinstance(expr,
                          (sympy.LessThan, sympy.GreaterThan, sympy.StrictGreaterThan, sympy.StrictLessThan, sympy.Ne))

    def _strategy_inequality(self, expr, depth=0, max_depth=4, indices=None):
        if indices is None:
            indices = []

        if self._is_inequality(expr):
            old = self._print_expr(expr)
            if depth == 0:
                if not isinstance(expr, sympy.Ne):
                    # swap arguments
                    tmp = expr.args[0]
                    expr.set_arg(0, expr.args[1])
                    expr.set_arg(1, tmp)
                else:
                    return False
            else:
                try:
                    with evaluate(False):
                        operator = self._get_inverse_operator(expr)(expr.args[0], expr.args[1], evaluate=False)
                    self.current_tex.set_arg(indices, operator)
                except Exception:
                    # swap arguments
                    tmp = expr.args[0]
                    expr.set_arg(0, expr.args[1])
                    expr.set_arg(1, tmp)
            new_expression = self._print_expr(expr)
            self.current_stats['strategy_inequality'] = {'old': old, 'new': new_expression, 'indices': indices}
            return True
        elif depth < max_depth and hasattr(expr, 'args'):
            arg_indices = list(range(len(expr.args)))
            random.shuffle(arg_indices)
            for i in arg_indices:
                success = self._strategy_inequality(expr.args[i], depth=depth + 1, max_depth=max_depth,
                                                    indices=indices + [i])
                if success:
                    return True
        return False

    def _strategy_swap(self, expr):
        # swap or modify the arguments of non-commutative binary operators: pow, div, minus
        # also unary functions can be replaced by other unary functions

        indices = self._get_swap_indices(expr)  # todo larger subexpressions might be alright as well
        if len(indices) > 0:
            indices = random.choice(indices)
            # swap
            e = expr.get_arg(indices)
            old = self._print_expr(e)
            args = list(e.args)

            set_indices = None
            new_element = None

            if isinstance(e, sympy.Add):
                # x + y -> -x - y
                expr.set_arg(indices + [0], -args[1])
                expr.set_arg(indices + [1], -args[0])
                new_ = self._print_expr(expr.get_arg(indices))
                self.current_stats['strategy_swap'] = {'old': old, 'new': new_, 'indices': indices}
                return True
            elif isinstance(e, sympy.Mul):
                pow = args[1]
                if isinstance(pow, sympy.Pow):
                    # x * y^-1 -> y * x^-1
                    expr.set_arg(indices + [0], pow.args[0])
                    pow.set_arg(0, args[0])
                    new_ = self._print_expr(expr.get_arg(indices))
                    self.current_stats['strategy_swap'] = {'old': old, 'new': new_, 'indices': indices}
                    return True
                else:
                    # make the second argument to divisor
                    set_indices = indices + [1]
                    new_element = sympy.Pow(pow, -1)

            elif self._is_unary(e):
                new_op = self._get_random_unary_operator(exclude=[type(e), sympy.factorial, sympy.exp])
                set_indices = indices
                new_element = new_op(args[0])
            else:
                # power
                if args[1] is S.Half:
                    # swapping args[1] and args[0] would lead to major latex reformatting (sqrt -> ()^())
                    # use any unary function symbol instead
                    unary_function = self._get_random_unary_operator(exclude=[sympy.sqrt])

                    set_indices = indices
                    new_element = unary_function(args[0])
                elif args[1] is -S.One:
                    # this expression has the shape 1/x, so just remove the division
                    set_indices = indices
                    new_element = args[0]
                else:
                    # swap the args
                    expr.set_arg(indices + [0], args[1])
                    expr.set_arg(indices + [1], args[0])
                    new_ = self._print_expr(expr.get_arg(indices))
                    self.current_stats['strategy_swap'] = {'old': old, 'new': new_, 'indices': indices}
                    return True

            if set_indices and new_element:
                expr.set_arg(set_indices, new_element)
                new_ = self._print_expr(expr.get_arg(indices))
                self.current_stats['strategy_swap'] = {'old': old, 'new': new_, 'indices': indices}
                return True
        return False

    def _is_unary(self, expr, exclude=[]):
        return isinstance(expr, tuple([o for o in self.unary_operator if o not in exclude]))

    def __disjoint_subsets(self, lst):
        n = len(lst)
        k = random.randint(1, n - 1)  # at least one element is needed for subset2
        subset1 = random.sample(lst, k)
        subset2 = [elem for elem in lst if elem not in subset1]
        n = len(subset2)
        k = random.randint(1, n)
        subset2 = random.sample(subset2, k)

        return subset1, subset2

    def _strategy_variables(self, expr):
        if isinstance(expr, sympy.Tuple) or not hasattr(expr, 'args'):
            return False

        l = len(expr.args)
        if isinstance(expr, sympy.Interval):
            l = 2

        if l >= 2:
            if l == 2:
                max_tries = 1
            elif l == 3:
                max_tries = 3
            elif l == 4:
                max_tries = 4
            else:
                max_tries = 7

            for i in range(max_tries):
                try:
                    side1, side2 = self.__disjoint_subsets(list(range(l)))

                    lhs = set(a for s1 in side1 for a in expr.args[s1].free_symbols)
                    rhs = set(a for s2 in side2 for a in expr.args[s2].free_symbols)

                    common = set(lhs).intersection(rhs)
                    if len(common) > 0:
                        old = self._print_expr(expr)
                        side = random.choice([side1, side2])
                        if self._random():
                            # recurse
                            s = random.choice(side)
                            success = self._strategy_variables(expr.args[s])
                            if success:
                                return True

                        # use the found common symbol
                        element = random.choice(list(common))
                        candidates = list(self._get_variable_permutation_candidates(element, is_function=False))
                        if len(candidates) > 0:
                            new_symbol = sympy.Symbol(random.choice(candidates))
                            try:
                                with evaluate(False):
                                    for s in side:
                                        expr.set_arg(s, expr.args[s].subs(element, new_symbol))
                                new_ = self._print_expr(expr)
                                self.current_stats['strategy_variables'] = {'old': old, 'new': new_,
                                                                            'substitution': {self._print_expr(element),
                                                                                             self._print_expr(
                                                                                                 new_symbol)},
                                                                            'indices': side}
                                return True
                            except Exception as e:  # if x->x
                                pass
                except Exception as e:
                    pass
        return False

    def _strategy_distribute(self, expr):

        candidates = self._get_distribute_candidates(expr)
        if len(candidates) > 0:
            candidate = random.choice(candidates)
            op1 = candidate[1]
            op2 = candidate[2]
            lhs = expr.get_arg(candidate[3])
            rhs = expr.get_arg(candidate[4])
            indices = candidate[3][:-1]  # same as candidate[4][:-1]
            with evaluate(False):
                if candidate[0]:
                    # case like sin(x+y) -> sin(x) + sin(y)
                    new_ = op2(op1(lhs), op1(rhs))
                else:
                    # case like sin(x)+sin(y) -> sin(x+y)
                    new_ = op1(op2(lhs, rhs))
            if len(indices) > 1:
                old = self._print_expr(expr.get_arg(indices[:-1]))
                expr.set_arg(indices[:-1], new_)
                new__ = self._print_expr(new_)
                self.current_stats['strategy_distribute'] = {'old': old, 'new': new__}
                return True

        return False

    def __check_candidate(self, expr, lhs, rhs, is_add, arg_index):
        pi = sympy.pi
        pi2 = pi * 2
        is_mul = not is_add
        if isinstance(expr, sympy.sin):
            # exclude cases like sin(0 + x) = sin(x) =  sin(0) + sin(x) or sin(0*x) = sin(0) = 0 = sin(0) * sin(x)
            if is_add or (lhs in [0, pi2, -pi2] or rhs in [0, pi2, -pi2]):
                return False

        elif isinstance(expr, sympy.cos):
            # exclude cases like cos(1 * x) = cos(x) = 1 * cos(x) * cos(1) * cos(x)
            if is_mul and S.One in [lhs, rhs]:
                return False

        elif isinstance(expr, sympy.log):
            # exclude case log(1*1) = log(1) = 0 = log(1) * log(1)
            if is_mul and lhs == S.One and rhs == S.One:
                return False

        elif isinstance(expr, sympy.Pow):
            if is_add and arg_index == 0:
                # exclude (x+0)^n = x^n = x^n + 0^n
                if S.Zero in [lhs, rhs]:
                    return False

                # exclude (x-x)^n
                if lhs == -rhs:
                    return False


            elif is_mul and arg_index == 0:
                # exclude (x*1)^n = x^n = x^n * 1^n
                if S.One in [lhs, rhs]:
                    return False

        elif isinstance(expr, sympy.factorial):
            # exclude 0! and 1!
            if lhs in [S.One, S.Zero] or rhs in [S.One, S.Zero]:
                return False

        return True

    def _get_distribute_candidates(self, expr, prefix=None):
        if prefix is None:
            prefix = []

        indices = []

        non_distributive = [sympy.sin, sympy.cos, sympy.tan, sympy.acos, sympy.asin, sympy.atan, sympy.log,
                            sympy.factorial, sympy.Pow]

        is_pow = isinstance(expr, sympy.Pow)
        if is_pow:
            arg_index = random.choice([0, 1])
            is_non_distributive = True

        else:
            is_non_distributive = any(isinstance(expr, c) for c in non_distributive) or is_pow
            arg_index = 0

        def get_op(expr, arg_index):
            if isinstance(expr, sympy.log):
                if len(expr.args) >= 2:
                    return lambda x: sympy.log(x, expr.args[1])
                return sympy.log
            elif isinstance(expr, sympy.Pow):
                if arg_index == 0:
                    return lambda x: sympy.Pow(x, expr.args[1])
                else:
                    return lambda x: sympy.Pow(expr.args[0], x)
            else:
                return type(expr)

        if is_non_distributive:
            arg = expr.args[arg_index]
            is_add = isinstance(arg, sympy.Add)
            is_mul = isinstance(arg, sympy.Mul)
            op = sympy.Add if is_add else sympy.Mul

            if is_add or is_mul:
                # candidate like sin(x+y)-> sin(x)+sin(y)
                if len(arg.args) == 2:
                    lhs = arg.args[0]
                    rhs = arg.args[1]

                    candidate = self.__check_candidate(expr, lhs, rhs, is_add, arg_index)

                    if candidate:
                        if is_pow:
                            lhs = prefix + [arg_index, 0]
                            rhs = prefix + [arg_index, 1]
                        else:
                            lhs = prefix + [0, 0]
                            rhs = prefix + [0, 1]
                        candidate = (True, get_op(expr, arg_index), op, lhs, rhs)
                        indices.append(candidate)
        else:
            is_add = isinstance(expr, sympy.Add)
            is_mul = isinstance(expr, sympy.Mul)
            if is_add or is_mul:
                lhs = expr.args[0]
                rhs = expr.args[1]

                if len(expr.args) == 2:
                    candidate = any(isinstance(lhs, c) and isinstance(rhs, c) for c in non_distributive)

                    if candidate and isinstance(lhs, sympy.log):
                        if len(lhs.args) < 2 ^ len(rhs.args) < 2:
                            candidate = False  # a log has a base given, the other one not
                        elif len(lhs.args) == 2 and len(rhs.args) == 2:
                            # check that bases are equal
                            candidate = lhs.args[1] == rhs.args[1]

                    if candidate:
                        if isinstance(lhs, sympy.Pow):
                            arg_index = random.choice([0, 1])
                            other = abs(1 - arg_index)
                            candidate = lhs.args[other] == rhs.args[other]
                            is_pow = True

                        l = lhs.args[arg_index]
                        r = rhs.args[arg_index]
                        candidate &= self.__check_candidate(lhs, l, r, is_add, arg_index)

                        if candidate:
                            # candidate like sin(x)+sin(y) -> sin(x+y)
                            op1 = get_op(lhs, arg_index)
                            op2 = sympy.Add if is_add else sympy.Mul
                            if is_pow:
                                lhs = prefix + [0, arg_index]
                                rhs = prefix + [1, arg_index]
                            else:
                                # additional unwrapping of sin(x) -> x necessary
                                lhs = prefix + [0, 0]
                                rhs = prefix + [1, 0]

                            candidate = (False, op1, op2, lhs, rhs)
                            indices.append(candidate)

        if hasattr(expr, 'args'):
            # recurse
            for i, arg in enumerate(expr.args):
                indices += self._get_distribute_candidates(arg, prefix + [i])

        return indices

    def _get_operator(self, expr):
        return type(expr)

    def _random_number(self):
        standard_candidates = [S.One, S.Zero, -S.One, sympy.Number(2)]
        if self._random(probability_true=0.7):
            return random.choice(standard_candidates)
        candidates = [42, 73, 128, S.Half, random.randint(-100, 500), sympy.pi, -sympy.pi, sympy.oo, sympy.I, -1, 1 / 3,
                      random.randint(0, 20), random.randint(-20, 20),
                      int(100 * random.random() * random.randint(-2, 3)) / 100.0]

        candidate = random.choice(candidates)
        if isinstance(candidate, (int, float)):
            return sympy.Number(candidate)
        return candidate

    def _strategy_constants(self, expr):
        # replace constants (like 0, 1, oo, pi, ...) by other constants
        # and replace special variables by constants (^n in sum by \infty)

        candidates = self._get_constant_indices(expr)

        # not inequality like >0
        if len(candidates) > 0:
            candidate = random.choice(candidates)
            zero = sympy.Number(0)
            one = sympy.Number(1)
            three = sympy.Number(3)
            pi = sympy.pi
            n = sympy.Symbol('n')
            m = sympy.Symbol('m')
            inf = sympy.oo
            old = expr.get_arg(candidate)
            i = sympy.I
            e = sympy.E

            random_number = self._random_number()

            if old == inf:
                new_ = random.choice([n, m, -inf, zero])
            elif old == pi:
                new_ = random.choice([three, one, zero, -pi])
            elif old == one:
                new_ = random.choice([-one, zero, random_number])
            elif old == zero:
                new_ = random.choice([one, inf, three, random_number])
            elif old == n:
                new_ = random.choice([inf, n - 1, n + 1, random_number])
            elif old == i:
                new_ = random.choice([one, -one, i * i])
            elif old == e:
                new_ = random.choice([three, pi, i, random_number])
            else:
                new_ = random_number

            if old == new_:
                return False

            if len(candidate) == 0:
                self.current_tex = new_
            else:
                expr.set_arg(candidate, new_)
            self.current_stats['strategy_constants'] = {'old': old, 'new': self._print_expr(new_)}
            return True

        return False

    def _get_constant_indices(self, expr, prefix=None, is_super=False):
        if prefix is None:
            prefix = []

        indices = []
        # check for number (note that expr.is_number does not work here since e.g. object log(2) is also number
        try:
            if (isinstance(expr, sympy.Number) or expr in [sympy.E, sympy.pi, sympy.I]) and expr != -1:
                indices.append(prefix)
            elif is_super and isinstance(expr, sympy.Symbol) and expr.name in ['n', 'm']:
                indices.append(prefix)
            elif hasattr(expr, 'args'):
                if isinstance(expr, sympy.Tuple):
                    is_super = lambda i: i == 2
                else:
                    is_super = lambda i: False

                for i, arg in enumerate(expr.args):
                    indices += self._get_constant_indices(arg, prefix + [i], is_super(i))

        except Exception as e:
            print("Error generating constant indices ")
            print(e)
        return indices

    def _strategy_random_formula(self, max_tries=10):
        if (self._random(probability_true=0.5) or not self.random_formula) and len(self.no_versions) > 0:
            # use no_versions -> MANUAL
            for i in range(max_tries):
                no_version = random.choice(self.no_versions)
                name, formula, stats = no_version.get_random_version(only_true_version=True, return_stats=True)
                if not formula or not name:
                    continue

                if 'formula_name_id' not in stats or stats['formula_name_id'] == self.id:
                    # this formula has actually been generated based on the same formula
                    continue

                stats = stats.copy()
                if 'stats' in stats:
                    del stats['stats']

                if formula[1]:
                    old = self._print_expr(self.current_tex)
                    try:
                        tex = parse_latex(formula[0])
                        self.current_tex = tex
                        self.current_stats['strategy_random_formula'] = {'old': old, 'new': formula[0],
                                                                         'no_version': True, "stats": dict(stats)}
                        return True
                    except Exception as e:
                        pass
        elif self.random_formula:
            # true RANDOM versions
            for i in range(max_tries):
                r = self.random_formula(return_stats=True, only_true_version=True)
                if r and isinstance(r, tuple):
                    random_formula, stats = r
                    if random_formula and random_formula[2] and 'formula_name_id' in stats and stats[
                        'formula_name_id'] != self.id:
                        random_formula = parse_latex(random_formula[1])
                        if random_formula != self.current_tex:
                            stats = stats.copy()
                            if 'stats' in stats:
                                del stats['stats']
                            old = self._print_expr(self.current_tex)
                            self.current_stats['strategy_random_formula'] = {'old': old, 'new': random_formula,
                                                                             'no_version': False}
                            self.current_tex = random_formula
                            return True
                elif isinstance(r, str):
                    random_formula = parse_latex(r)
                    old = self._print_expr(self.current_tex)
                    self.current_stats['strategy_random_formula'] = {'old': old, 'new': r, 'no_version': False}
                    self.current_tex = random_formula
                    return True
                return False
        return False

    def _get_inverse_operator(self, expr):
        if isinstance(expr, sympy.LessThan):
            return sympy.StrictGreaterThan
        elif isinstance(expr, sympy.GreaterThan):
            return sympy.StrictLessThan
        elif isinstance(expr, sympy.StrictLessThan):
            return sympy.GreaterThan
        elif isinstance(expr, sympy.StrictGreaterThan):
            return sympy.LessThan
        elif isinstance(expr, sympy.Ne):
            return random.choices([sympy.Eq, sympy.GreaterThan, sympy.LessThan], weights=[3, 1, 1])[0]
        else:
            raise ValueError("Unexpected type %s" % type(expr))

    def _get_small_subexpression_indices(self, expr, prefix=None):
        if prefix is None:
            prefix = []
        indices = []
        if self._is_small_expression(expr) and len(prefix) > 1 and self._is_replacable_subexpression(expr):
            indices.append(prefix)

        if hasattr(expr, 'args'):
            for i, arg in enumerate(expr.args):
                if isinstance(arg, (sympy.core.function.MultiDerivative,)):
                    if i > 1:
                        continue
                elif isinstance(arg, (sympy.Indexed, sympy.IndexedBase, sympy.Tuple,)):
                    continue
                current_index = prefix.copy()
                current_index.append(i)
                indices += self._get_small_subexpression_indices(arg, current_index)

        return indices

    def _get_swap_indices(self, expr, prefix=None):
        if prefix is None:
            prefix = []
        elif not isinstance(prefix, list):
            prefix = list(prefix)

        indices = []
        if self._is_swap_expression(expr):
            indices.append(prefix)

        if hasattr(expr, 'args'):
            for i, arg in enumerate(expr.args):
                current_index = prefix.copy()
                current_index.append(i)
                indices += self._get_swap_indices(arg, current_index)

        return indices

    def _is_swap_expression(self, expr):
        if isinstance(expr, (sympy.Pow)):
            return True

        if isinstance(expr, sympy.Add):
            return isinstance(expr.args[1], sympy.Mul) and expr.args[1].args[
                0] == -S.One  # it is actually a subtraction

        if isinstance(expr, sympy.Mul):
            return not isinstance(expr.args[1], sympy.Pow) or isinstance(expr.args[1], sympy.Pow) and expr.args[1].args[
                0] == S.One

        if self._is_unary(expr, exclude=[sympy.factorial]):
            return True

        return False

    def _is_replacable_subexpression(self, expr):
        if hasattr(expr, 'limits'):
            return False

        if isinstance(expr, (sympy.Indexed, sympy.IndexedBase)):
            return False

        return True

    def _get_random_variable(self, return_symbol=False, prefer_used_var=True, exclude=None, include_numbers=False):
        if exclude is None:
            exclude = []
        else:
            exclude = [e.lower() for e in exclude]

        if not prefer_used_var or self._random() or len(self.current_tex.free_symbols) == 0:
            candidates = self.all_known_variables
            candidates = [c for c in candidates if c.lower() not in exclude]
            if include_numbers:
                candidates += list(range(-2, 17))
            v = random.choice(candidates)
        else:
            v = random.choice(list(self.current_tex.free_symbols))
        if return_symbol:
            if not isinstance(v, (sympy.Symbol, sympy.Indexed)):
                if isinstance(v, int):
                    return sympy.Integer(v)
                else:
                    return sympy.Symbol(v)
        elif isinstance(v, sympy.Symbol):
            v = v.name
        return v

    def _random_expression(self):
        if self._random(probability_true=0.8):
            return self._get_random_variable(True, include_numbers=True)

        # make a binary expression, like x+y or x^y
        return self._get_random_binary_operator()(self._get_random_variable(return_symbol=True, include_numbers=True),
                                                  self._get_random_variable(return_symbol=True, prefer_used_var=False,
                                                                            include_numbers=True),
                                                  evaluate=False)

    def _random_subexpression(self, expr):
        indices = self._get_small_subexpression_indices(expr)
        if len(indices) == 0:
            return None

        indices = random.choice(indices)
        sub_expression = expr
        for i in indices:
            sub_expression = sub_expression.args[i]

        return sympy_copy(sub_expression)

    def _is_small_expression(self, expr, depth=0):
        if isinstance(expr, (
        sympy.Equality, sympy.LessThan, sympy.GreaterThan, sympy.StrictLessThan, sympy.StrictGreaterThan,
        sympy.Formulas, sympy.core.BasicMatrix, sympy.Indexed, sympy.IndexedBase, sympy.Tuple, sympy.core.symbol.Str,
        sympy.Union, sympy.Intersection, sympy.sets.sets.SetComplement, sympy.sets.sets.SetMinus,
        sympy.core.BasicVector)):
            return False

        try:
            l = len(expr.args)
            if l == 0:
                return True
            elif depth < 1 and (l == 1 or l == 2):
                args = expr.args
                return all([self._is_small_expression(a, depth + 1) for a in args])
        except Exception as e:
            pass

        return False

    unary_operator = [sympy.sin, sympy.cos, sympy.tan, sympy.log, sympy.factorial, sympy.acos, sympy.asin, sympy.atan]

    def _get_random_unary_operator(self, exclude=None):
        if not exclude:
            exclude = []
        candidates = self.unary_operator + [sympy.sqrt]
        candidates = [c for c in candidates if c not in exclude]
        return random.choice(candidates)

    def __str__(self):
        return "Template %s" % self.formula

    def set_not_allowed_names(self, names):
        self.not_allowed_names = names
