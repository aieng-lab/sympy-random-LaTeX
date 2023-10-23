/*
 ANTLR4 LaTeX Math Grammar

 Ported from latex2sympy by @augustt198 https://github.com/augustt198/latex2sympy See license in
 LICENSE.txt
 */

/*
 After changing this file, it is necessary to run `python setup.py antlr` in the root directory of
 the repository. This will regenerate the code in `sympy/parsing/latex/_antlr/*.py`.
 */

parser grammar LaTeX_test;

options {
	language = Python3;
	tokenVocab=LaTeXLexer;
}



math: TEXT_OPEN TEXT_BODY TEXT_CLOSE | LETTER*;