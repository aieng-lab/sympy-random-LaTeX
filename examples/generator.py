from sympy.generator import FormulaGenerator
from sympy.parsing.latex import _parse_latex_antlr
from sympy.util import RandomChoice, RandomDecidedTruthValue

formula = "e^{\mathrm{i}\pi} = -1"

e = _parse_latex_antlr.parse_latex(formula)

generator = FormulaGenerator(formula)

for version in generator.generate_versions_iterator(max=10, only_true_version=True):
    print(version)


formula = "(a+b)^2 = a^2 + 2ab + b^2"
generator = FormulaGenerator(formula)
for version in generator.generate_versions_iterator(max=10, only_true_version=True):
    print(version)



formula = r"a^2+b^2=c^2"
generator = FormulaGenerator(formula)
for version in generator.generate_versions_iterator(max=10):
    print(version)


custom_randomized_settings = {
    "root_notation": RandomDecidedTruthValue(0.7),
    "frac": RandomChoice([r"\frac", r"\mycustomfrac", r"\tfrac"], weights=[2, 10, 1]),
}
formula = r"\frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}"
generator = FormulaGenerator(formula, randomize_settings=custom_randomized_settings)
for version in generator.generate_versions_iterator(max=10, only_true_version=True):
    print(version)
