# SymPy with randomized LaTeX Formula Generator

[![arXiv](https://img.shields.io/badge/arXiv-2502.20855-B31B1B.svg)](https://arxiv.org/abs/2502.20855)


# Special Features of this fork
This SymPy repository contains a FormulaGenerator which generates randomized versions of a given formula.

```python
from sympy.generator import FormulaGenerator

formula = "-1 = e^{\\pi * i}"
generator = FormulaGenerator(formula)

for version, _ in generator.generate_versions_iterator(max=4, only_true_version=True, initial_is_candidate=False):
    print(version)

>>> e^{\\pi \\cdot i} = -1
>>> e^{i \\pi} = -1
>>> e^{\\pi \\cdot \\mathrm{i}} = -1
>>> -1 = e^{\\pi\\cdot \\mathrm{i}}
```
The `FormulaGenerator' takes a LaTeX Formula as input and creates equivalent versions in this example. If variables
occur in the formula, they are substituted by random variables:
```python
from sympy.generator import FormulaGenerator
formula = "(a+b)^2 = a^2 + 2ab + b^2"
generator = FormulaGenerator(formula)
for version, _ in generator.generate_versions_iterator(max=3, only_true_version=True, initial_is_candidate=False):
    print(version)

>>> x^2 + 2*b*x + b * b = (x + b)^2
>>> b^2 + a^2 + b\\cdot a\\cdot 2 = (a + b)^2
>>> f^2 + f\\times c\\times 2 + c^2 = (c + f)^2
```
To disable the substitutions, use `FormulaGenerator(formula, max_substitution=0)`.

Furthermore, the generator can generate *falsified* versions of the given formula, meaning a similar looking but not
mathematical equivalent formula.

```python
from sympy.generator import FormulaGenerator
formula = r"a^2+b^2=c^2"
generator = FormulaGenerator(formula)
for version in generator.generate_versions_iterator(max=10, initial_is_candidate=False):
    print(version)

>>> ('x^2 = d^2 + f * f', True)
>>> ('2^c = b^2 + h^2', False)
>>> ('g^2 = a^2 + d^2', True)
>>> ('x^2 = d^2 + f * f', True)
>>> ('2^a = c^2', False)
```
The factor of falsified versions per equivalent version can be controlled by `factor_false` argument in `genertate_versions_iterator` (default `=1`). The falsified versions may be used to generate challenging ML training data, e.g., for an information retrieval model.

The randomized printing can be controlled by changing the [randomize_settings](sympy/settings.py) or providing it by `FormulaGenerator(formula, randomize_settings=custom_randomized_settings)`
```python
from sympy.generator import FormulaGenerator
from sympy.util import RandomChoice, RandomDecidedTruthValue
custom_randomized_settings = {
    "root_notation": RandomDecidedTruthValue(0.7), # use the same root notation consistent across a single formula (i.e., \sqrt in 70% of the cases and x^{1/2} otherwise)
    "frac": RandomChoice([r"\frac", r"\mycustomfrac"], weights=[2, 10]), # use \frac in 20% of the cases and \mycustomfrac otherwise (the notation might be different for different fractions within a formula as RandomChoice is used instead of RandomDecidedChoice)
}
formula = r"\frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}"
generator = FormulaGenerator(formula, randomize_settings=custom_randomized_settings)
for version, _ in generator.generate_versions_iterator(max=3, only_true_version=True, initial_is_candidate=False):
    print(version)

>>> \frac{1}{\sqrt{2}} = \mycustomfrac{\sqrt{2}}{2}
>>> \mycustomfrac{1}{\sqrt{2}} = \mycustomfrac{\sqrt{2}}{2}
>>> \frac{1}{2^{\mycustomfrac{1}{2}}} = \mycustomfrac{2^{\mycustomfrac{1}{2}}}{2}
```

The examples are provided in [examples/generator.py](examples/generator.py). Due to randomized printing, your results
may (most likely) be different when running the code.
You may need to run `pip install antlr4-python3-runtime==4.12` to enable LaTeX parsing.

## Generated Datasets

See [Math Mutator (MAMUT)](https://github.com/aieng-lab/math-mutator) for information regarding generated datasets using this special fork.

## CITATION
If you use the special features of this fork, please cite the following paper:
```bibtex
@article{
  drechsel2025mamut,
  title={{MAMUT}: A Novel Framework for Modifying Mathematical Formulas for the Generation of Specialized Datasets for Language Model Training},
  author={Jonathan Drechsel and Anja Reusch and Steffen Herbold},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=khODmRpQEx}
}
```

See also the citation section of the general SymPy repository below.

# SymPy
[![pypi version](https://img.shields.io/pypi/v/sympy.svg)](https://pypi.python.org/pypi/sympy)
[![Join the chat at https://gitter.im/sympy/sympy](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sympy/sympy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Zenodo Badge](https://zenodo.org/badge/18918/sympy/sympy.svg)](https://zenodo.org/badge/latestdoi/18918/sympy/sympy)
[![Downloads](https://pepy.tech/badge/sympy/month)](https://pepy.tech/project/sympy)
[![GitHub Issues](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/sympy/sympy/issues)
[![Git Tutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![Powered by NumFocus](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
[![Commits since last release](https://img.shields.io/github/commits-since/sympy/sympy/latest.svg?longCache=true&style=flat-square&logo=git&logoColor=fff)](https://github.com/sympy/sympy/releases)

[![SymPy Banner](https://github.com/sympy/sympy/raw/master/banner.svg)](https://sympy.org/)


See the [AUTHORS](AUTHORS) file for the list of authors.

And many more people helped on the SymPy mailing list, reported bugs,
helped organize SymPy's participation in the Google Summer of Code, the
Google Highly Open Participation Contest, Google Code-In, wrote and
blogged about SymPy...

License: New BSD License (see the [LICENSE](LICENSE) file for details) covers all
files in the sympy repository unless stated otherwise.

Our mailing list is at
<https://groups.google.com/forum/?fromgroups#!forum/sympy>.

We have a community chat at [Gitter](https://gitter.im/sympy/sympy). Feel
free to ask us anything there. We have a very welcoming and helpful
community.



## Download

The recommended installation method is through Anaconda,
<https://www.anaconda.com/products/distribution>

You can also get the latest version of SymPy from
<https://pypi.python.org/pypi/sympy/>

To get the git version do

    $ git clone https://github.com/sympy/sympy.git

For other options (tarballs, debs, etc.), see
<https://docs.sympy.org/dev/install.html>.

## Documentation and Usage

For in-depth instructions on installation and building the
documentation, see the [SymPy Documentation Style Guide](https://docs.sympy.org/dev/documentation-style-guide.html).

Everything is at:

<https://docs.sympy.org/>

You can generate everything at the above site in your local copy of
SymPy by:

    $ cd doc
    $ make html

Then the docs will be in <span class="title-ref">\_build/html</span>. If
you don't want to read that, here is a short usage:

From this directory, start Python and:

``` python
>>> from sympy import Symbol, cos
>>> x = Symbol('x')
>>> e = 1/cos(x)
>>> print(e.series(x, 0, 10))
1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + 277*x**8/8064 + O(x**10)
```

SymPy also comes with a console that is a simple wrapper around the
classic python console (or IPython when available) that loads the SymPy
namespace and executes some common commands for you.

To start it, issue:

    $ bin/isympy

from this directory, if SymPy is not installed or simply:

    $ isympy

if SymPy is installed.

## Installation

SymPy has a hard dependency on the [mpmath](http://mpmath.org/) library
(version \>= 0.19). You should install it first, please refer to the
mpmath installation guide:

<https://github.com/fredrik-johansson/mpmath#1-download--installation>

To install SymPy using PyPI, run the following command:

    $ pip install sympy

To install SymPy using Anaconda, run the following command:

    $ conda install -c anaconda sympy

To install SymPy from GitHub source, first clone SymPy using `git`:

    $ git clone https://github.com/sympy/sympy.git

Then, in the `sympy` repository that you cloned, simply run:

    $ pip install .

See <https://docs.sympy.org/dev/install.html> for more information.

## Contributing

We welcome contributions from anyone, even if you are new to open
source. Please read our [Introduction to Contributing](https://docs.sympy.org/dev/contributing/introduction-to-contributing.html)
page and the [SymPy Documentation Style Guide](https://docs.sympy.org/dev/documentation-style-guide.html). If you
are new and looking for some way to contribute, a good place to start is
to look at the issues tagged [Easy to Fix](https://github.com/sympy/sympy/issues?q=is%3Aopen+is%3Aissue+label%3A%22Easy+to+Fix%22).

Please note that all participants in this project are expected to follow
our Code of Conduct. By participating in this project you agree to abide
by its terms. See [CODE\_OF\_CONDUCT.md](CODE_OF_CONDUCT.md).

## Tests

To execute all tests, run:

    $./setup.py test

in the current directory.

For the more fine-grained running of tests or doctests, use `bin/test`
or respectively `bin/doctest`. The master branch is automatically tested
by GitHub Actions.

To test pull requests, use
[sympy-bot](https://github.com/sympy/sympy-bot).

## Regenerate Experimental <span class="title-ref">LaTeX</span> Parser/Lexer

The parser and lexer were generated with the [ANTLR4](http://antlr4.org)
toolchain in `sympy/parsing/latex/_antlr` and checked into the repo.
Presently, most users should not need to regenerate these files, but
if you plan to work on this feature, you will need the `antlr4`
command-line tool (and you must ensure that it is in your `PATH`).
One way to get it is:

    $ conda install -c conda-forge antlr=4.11.1

Alternatively, follow the instructions on the ANTLR website and download
the `antlr-4.11.1-complete.jar`. Then export the `CLASSPATH` as instructed
and instead of creating `antlr4` as an alias, make it an executable file
with the following contents:
``` bash
#!/bin/bash
java -jar /usr/local/lib/antlr-4.11.1-complete.jar "$@"
```

After making changes to `sympy/parsing/latex/LaTeX.g4`, run:

    $ ./setup.py antlr

## Clean

To clean everything (thus getting the same tree as in the repository):

    $ git clean -Xdf

which will clear everything ignored by `.gitignore`, and:

    $ git clean -df

to clear all untracked files. You can revert the most recent changes in
git with:

    $ git reset --hard

WARNING: The above commands will all clear changes you may have made,
and you will lose them forever. Be sure to check things with `git
status`, `git diff`, `git clean -Xn`, and `git clean -n` before doing any
of those.

## Bugs

Our issue tracker is at <https://github.com/sympy/sympy/issues>. Please
report any bugs that you find. Or, even better, fork the repository on
GitHub and create a pull request. We welcome all changes, big or small,
and we will help you make the pull request if you are new to git (just
ask on our mailing list or Gitter Channel). If you further have any queries, you can find answers
on Stack Overflow using the [sympy](https://stackoverflow.com/questions/tagged/sympy) tag.

## Brief History

SymPy was started by Ondřej Čertík in 2005, he wrote some code during
the summer, then he wrote some more code during summer 2006. In February
2007, Fabian Pedregosa joined the project and helped fix many things,
contributed documentation, and made it alive again. 5 students (Mateusz
Paprocki, Brian Jorgensen, Jason Gedge, Robert Schwarz, and Chris Wu)
improved SymPy incredibly during summer 2007 as part of the Google
Summer of Code. Pearu Peterson joined the development during the summer
2007 and he has made SymPy much more competitive by rewriting the core
from scratch, which has made it from 10x to 100x faster. Jurjen N.E. Bos
has contributed pretty-printing and other patches. Fredrik Johansson has
written mpmath and contributed a lot of patches.

SymPy has participated in every Google Summer of Code since 2007. You
can see <https://github.com/sympy/sympy/wiki#google-summer-of-code> for
full details. Each year has improved SymPy by bounds. Most of SymPy's
development has come from Google Summer of Code students.

In 2011, Ondřej Čertík stepped down as lead developer, with Aaron
Meurer, who also started as a Google Summer of Code student, taking his
place. Ondřej Čertík is still active in the community but is too busy
with work and family to play a lead development role.

Since then, a lot more people have joined the development and some
people have also left. You can see the full list in doc/src/aboutus.rst,
or online at:

<https://docs.sympy.org/dev/aboutus.html#sympy-development-team>

The git history goes back to 2007 when development moved from svn to hg.
To see the history before that point, look at
<https://github.com/sympy/sympy-old>.

You can use git to see the biggest developers. The command:

    $ git shortlog -ns

will show each developer, sorted by commits to the project. The command:

    $ git shortlog -ns --since="1 year"

will show the top developers from the last year.

## Citation

To cite SymPy in publications use

> Meurer A, Smith CP, Paprocki M, Čertík O, Kirpichev SB, Rocklin M,
> Kumar A, Ivanov S, Moore JK, Singh S, Rathnayake T, Vig S, Granger BE,
> Muller RP, Bonazzi F, Gupta H, Vats S, Johansson F, Pedregosa F, Curry
> MJ, Terrel AR, Roučka Š, Saboo A, Fernando I, Kulal S, Cimrman R,
> Scopatz A. (2017) SymPy: symbolic computing in Python. *PeerJ Computer
> Science* 3:e103 <https://doi.org/10.7717/peerj-cs.103>

A BibTeX entry for LaTeX users is

``` bibtex
@article{10.7717/peerj-cs.103,
 title = {SymPy: symbolic computing in Python},
 author = {Meurer, Aaron and Smith, Christopher P. and Paprocki, Mateusz and \v{C}ert\'{i}k, Ond\v{r}ej and Kirpichev, Sergey B. and Rocklin, Matthew and Kumar, Amit and Ivanov, Sergiu and Moore, Jason K. and Singh, Sartaj and Rathnayake, Thilina and Vig, Sean and Granger, Brian E. and Muller, Richard P. and Bonazzi, Francesco and Gupta, Harsh and Vats, Shivam and Johansson, Fredrik and Pedregosa, Fabian and Curry, Matthew J. and Terrel, Andy R. and Rou\v{c}ka, \v{S}t\v{e}p\'{a}n and Saboo, Ashutosh and Fernando, Isuru and Kulal, Sumith and Cimrman, Robert and Scopatz, Anthony},
 year = 2017,
 month = Jan,
 keywords = {Python, Computer algebra system, Symbolics},
 abstract = {
            SymPy is an open-source computer algebra system written in pure Python. It is built with a focus on extensibility and ease of use, through both interactive and programmatic applications. These characteristics have led SymPy to become a popular symbolic library for the scientific Python ecosystem. This paper presents the architecture of SymPy, a description of its features, and a discussion of select submodules. The supplementary material provides additional examples and further outlines details of the architecture and features of SymPy.
         },
 volume = 3,
 pages = {e103},
 journal = {PeerJ Computer Science},
 issn = {2376-5992},
 url = {https://doi.org/10.7717/peerj-cs.103},
 doi = {10.7717/peerj-cs.103}
}
```

SymPy is BSD licensed, so you are free to use it whatever you like, be
it academic, commercial, creating forks or derivatives, as long as you
copy the BSD statement if you redistribute it (see the LICENSE file for
details). That said, although not required by the SymPy license, if it
is convenient for you, please cite SymPy when using it in your work and
also consider contributing all your changes back, so that we can
incorporate it and all of us will benefit in the end.
