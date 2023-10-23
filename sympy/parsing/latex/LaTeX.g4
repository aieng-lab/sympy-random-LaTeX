/*
 ANTLR4 LaTeX Math Grammar

 Ported from latex2sympy by @augustt198 https://github.com/augustt198/latex2sympy See license in
 LICENSE.txt
 */

/*
 After changing this file, it is necessary to run `python setup.py antlr` in the root directory of
 the repository. This will regenerate the code in `sympy/parsing/latex/_antlr/*.py`.
 */

grammar LaTeX;

options {
	language = Python3;
}

WS: [ \t\r\n]+ -> skip;
THINSPACE: ('\\,' | '\\thinspace') -> skip;
MEDSPACE: ('\\:' | '\\medspace') -> skip;
THICKSPACE: ('\\;' | '\\thickspace') -> skip;
QUAD: '\\quad' -> skip;
QQUAD: '\\qquad' -> skip;
NEGTHINSPACE: ('\\!' | '\\negthinspace') -> skip;
NEGMEDSPACE: '\\negmedspace' -> skip;
NEGTHICKSPACE: '\\negthickspace' -> skip;
CMD_LEFT: ('\\left' | '\\biggl') -> skip;
CMD_RIGHT: ('\\right' | '\\biggr') -> skip;

MODIFIER: '\\bar' | '\\tilde' | '\\widetilde';

IGNORE:
	(
		'\\vrule'
		| '\\vcenter'
		| '\\vbox'
		| '\\vskip'
		| '\\vspace'
		| '\\hfil'
		| '\\*'
		| '\\-'
		| '\\.'
		| '\\/'
		| '\\"'
		| '\\('
		| '\\='
		| '\\cancel'
	) -> skip;

ADD: '+';
SUB: '-';
MUL: '*' | '×';
DIV: '/';
SET_MINUS: '\\setminus' | '\\backslash';
PM: '\\pm' | '+-' | '±';

L_PAREN: '(';
R_PAREN: ')';
L_BRACE: '{';
R_BRACE: '}';
L_BRACE_LITERAL: '\\{' | '\\lbrace';
R_BRACE_LITERAL: '\\}' | '\\rbrace';
L_BRACKET: '[';
R_BRACKET: ']';

BAR: '|';

MATRIX_DELIMITER : '&';

R_BAR: '\\right|';
L_BAR: '\\left|';

L_ANGLE: '\\langle';
R_ANGLE: '\\rangle';

L_VERT: '\\lvert';
R_VERT: '\\rvert';

FUNC_LIM: '\\lim';
FUNC_MIN: '\\min' | 'min';
FUNC_MAX: '\\max' | 'max';
FUNC_SUP: '\\sup' | 'sup';
FUNC_INF: '\\inf' | 'inf';

LIM_APPROACH_SYM:
	'\\to'
	| '\\rightarrow'
	| '\\longrightarrow'
	| '\\Longrightarrow';
FUNC_INT:
    '\\int'
    | '\\int\\limits';
FUNC_SUM: '\\sum' | '\\sum\\limits' | '∑';
FUNC_PROD: '\\prod';

FUNC_EXP: '\\exp' | 'exp';
FUNC_LOG: '\\log' | 'log';
FUNC_LG: '\\lg';
FUNC_LN: '\\ln' | 'ln';
FUNC_SIN: '\\sin' | 'sin';
FUNC_COS: '\\cos' | 'cos';
FUNC_TAN: '\\tan' | 'tan';
FUNC_CSC: '\\csc';
FUNC_SEC: '\\sec';
FUNC_COT: '\\cot';

FUNC_ARCSIN: '\\arcsin' | '\\asin';
FUNC_ARCCOS: '\\arccos' | '\\acos';
FUNC_ARCTAN: '\\arctan' | '\\atan';
FUNC_ARCCSC: '\\arccsc' | '\\acsc';
FUNC_ARCSEC: '\\arcsec' | '\\asec';
FUNC_ARCCOT: '\\arccot' | '\\acot';

FUNC_SINH: '\\sinh' | '\\operatorname{sinh}';
FUNC_COSH: '\\cosh' | '\\operatorname{cosh}';
FUNC_TANH: '\\tanh' | '\\operatorname{tanh}';
FUNC_ARSINH: '\\arsinh' | '\\operatorname{asinh}';
FUNC_ARCOSH: '\\arcosh' | '\\operatorname{acosh}';
FUNC_ARTANH: '\\artanh' | '\\operatorname{atanh}';

FUNC_VAR: 'Var' | 'VAR' | '\\mathbb{Var}' | '\\operatorname{Var}' | '\\Var';
FUNC_COV: 'Cov' | 'COV' | '\\mathbb{Cov}' | '\\operatorname{Cov}' | '\\Cov';
FUNC_EXPECTED_VALUE: 'E' | '\\mathbb{E}' | '\\operatorname{E}' | '\\mathbb E';
PROBABILITY: 'P' | '\\mathbb{P}' | '\\operatorname{P}';

FUNC_DET: 'det' | '\\operatorname{det}' | '\\det';
FUNC_RE: '\\Re' | 'Re' | '\\operatorname{re}';
FUNC_IM: '\\Im' | 'Im' | '\\operatorname{im}';

RIGHT_ARROW: '\\Rightarrow' | '\\implies';
LEFT_RIGHT_ARROW: '\\Leftrightarrow' | '\\iff' | '\\Longleftrightarrow';

L_FLOOR: '\\lfloor';
R_FLOOR: '\\rfloor';
L_CEIL: '\\lceil';
R_CEIL: '\\rceil';

FUNC_SQRT: '\\sqrt';
FUNC_OVERLINE: '\\overline';

FUNC_OVER: '\\over';

CMD_TIMES: '\\times';
CMD_CDOT: '\\cdot';
CMD_DIV: '\\div';
CMD_FRAC:
    '\\frac'
    | '\\dfrac'
    | '\\tfrac'
    | '\\cfrac';
CMD_BINOM: '\\binom';
CMD_DBINOM: '\\dbinom';
CMD_TBINOM: '\\tbinom';

CMD_CAP: '\\cap' | '∩' | '\\bigcap';
CMD_CUP: '\\cup' | '∪' | '\\bigcup';

CMD_MATHIT: '\\mathit';

UNDERSCORE: '_';
CARET: '^';
COLON: ':' | '\\colon';

DOTS: '\\dots' | '\\cdots' | '...' | '\\ldots' | '\\dotsm';

PERCENTAGE: '\\%';

AND: '\\land' | '\\wedge';
OR: '\\lor' | '\\vee';
TEXT_OR: 'or' | '\\text{ or }';
NEGATE: '\\neg';

fragment WS_CHAR: [ \t\r\n];
GREEK: '\\alpha' | '\\beta' | '\\gamma' | '\\delta'
        | '\\' ('alpha' | 'beta' | 'gamma' | 'delta' | 'epsilon' | 'varepsilon' | 'zeta' | 'eta' | 'theta'
                    | 'vartheta' | 'iota' | 'kappa' | 'lambda' | 'mu' | 'nu' | 'xi' | 'pi' | 'varpi' | 'rho' | 'varrho'
                    | 'sigma' | 'varsigma' | 'tau' | 'upsilon' | 'phi' | 'varphi' | 'chi' | 'psi' | 'omega' | 'zeta'
                    | 'Gamma' | 'Delta' | 'Theta' | 'Lambda' | 'Xi' | 'Sigma' | 'Upsilon' | 'Phi' | 'Psi'
                    | 'Omega')
        | PI ;
PI: '\\pi' | 'π';

TEXT_IF: ('\\text{' ' '* 'if' ' '* '}') | '\\text{' ' '* 'for' ' '* '}';
TEXT_OTHERWISE: '\\text{' ' '* 'otherwise' ' '* '}' | '\\text{' ' '* 'else' ' '* '}';
DDX: 'd/dx' | 'd/dy' | 'd/dz';

DERIVATIVE_SYMBOLS: '\\text{d}' | '\\mathrm{d}';
fragment DERIVATIVE_SYMBOL: 'd' | DERIVATIVE_SYMBOLS;
DIFFERENTIAL: DERIVATIVE_SYMBOL WS_CHAR*? ([a-zA-Z] | GREEK);
LETTER: [a-zA-Z];
DIGIT: [0-9];

DEFINES: ':=' | '\\coloneqq';
EQUAL: (('&' WS_CHAR*?)? ('=' | ':=')) | (('=' | ':=') (WS_CHAR*? '&')?);
NEQ: '\\neq' | '≠';

LT: '<' | '\\lt';
LTE: ('\\leq' | '\\le' | LTE_Q | LTE_S) | '≤';
LTE_Q: '\\leqq';
LTE_S: '\\leqslant';

GT: '>' | '\\gt';
GTE: '\\geq' | '\\ge' | GTE_Q | GTE_S | '≥';
GTE_Q: '\\geqq';
GTE_S: '\\geqslant';

SUBSET: '\\subset' | '\\subseteq';
SUPSET: '\\supset' | '\\supseteq';

APPROX: '\\approx' | '~' | '≈' | '\\sim';

BANG: '!';
MID: '\\mid' | BAR | '\\middle|';

SINGLE_QUOTES: '\''+;
BEGIN: '\\begin';
END: '\\end';

INFINITY: '\\infty' | '∞';

SPECIAL_SETS: '\\RR' | '\\mathbb{R}' | '\\R' | '\\mathbf{R}'
                | '\\N' | '\\mathbb{N}' | '\\mathbf{N}' | '\\mathbb N'
                | '\\Z' | '\\mathbb{Z}' | '\\mathbf{Z}' | '\\mathbb Z'
                | '\\Q' | '\\mathbb{Q}' | '\\mathbf{Q}' | '\\mathbb Q'
                | '\\C' | '\\mathbb{C}' | '\\mathbf{C}' | '\\mathbb C'
                | EMPTY_SET;
EMPTY_SET: '\\varnothing' | '∅' | (L_BRACE R_BRACE) | '\\O' | '\\emptyset';


BINARY_OPERATORS: '\\' ('oplus' | 'otimes' | 'triangleleft' | 'oslash' | 'boxdot' | 'triangleright' | 'barwedge'
                        | 'veebar');

CHOOSE: '\\choose';
//OPERATORNAME: '\\operatorname';
IN: '\\in' | '∈';

SYMBOL: GREEK | '\\' [a-zA-Z]+;

NORM: '\\|' | '||' | '\\lVert' | '\\rVert';

IMAGINARY_UNIT: '\\text{i}' | '\\mathrm{i}'; // not 'i' here, since this is deduced by the context

COMMA: ',';


math: implication;

implication: implication (RIGHT_ARROW | LEFT_RIGHT_ARROW) implication | L_BRACKET implication (RIGHT_ARROW | LEFT_RIGHT_ARROW) implication R_BRACKET | formulas;

formulas: relation ((',' | AND | TEXT_OR) relation)*;
l_interval: L_PAREN | R_PAREN | L_BRACKET | R_BRACKET;
r_interval: L_PAREN | R_PAREN | L_BRACKET | R_BRACKET;


variable: (LETTER | SYMBOL) in_set?;

set: (L_BRACE_LITERAL atom (',' atom)* R_BRACE_LITERAL) | (L_BRACE_LITERAL variable (MID | BAR | COLON) formulas | R_BRACE_LITERAL) | SPECIAL_SETS subexpr? supexpr? | (l_interval SUB? atom ',' SUB? atom r_interval);

in_set: (IN (set | LETTER));


relation:
	relation (EQUAL | LT | LTE | GT | GTE | NEQ | APPROX | SUPSET | SUBSET | DEFINES) relation
	| set
	| expr
	| variable;

equality: expr EQUAL expr;

text: ('\\text' | '\\textrm' | '\textbf') L_BRACE (LETTER | 'c' | 'r' | 'l')* R_BRACE;

expr: NEGATE? additive;
env:
    BEGIN L_BRACE 'cases' R_BRACE cases END L_BRACE 'cases' R_BRACE
    | BEGIN L_BRACE 'array' R_BRACE (L_BRACE ('c' | 'l' | 'r')* R_BRACE)? matrix END L_BRACE 'array' R_BRACE
    | BEGIN L_BRACE ('pmatrix' | 'matrix' | 'array') R_BRACE matrix END L_BRACE ('pmatrix' | 'matrix' | 'array') R_BRACE
    //| '\\begin' L_BRACE 'pmatrix' R_BRACE matrix '\\end' L_BRACE 'pmatrix' R_BRACE supexpr
    | '[[' matrix_row ']' (',' '[' matrix_row ']')* ']'
    | NORM expr NORM subexpr?
    | BAR expr BAR subexpr
   // | additive (ADD | SUB | PM | MUL | CMD_TIMES | CMD_CDOT | DIV | CMD_DIV | COLON) '\\begin' L_BRACE 'pmatrix' R_BRACE matrix '\\end' L_BRACE 'pmatrix' R_BRACE supexpr
    //| '\\begin'  ENVIRONMENT relation '\\end' ENVIRONMENT
  ;//  | additive;

cases_row: expr ('&'? ((TEXT_IF? relation) | TEXT_OTHERWISE));

cases: cases_row ('\\\\' cases_row)*;

matrix: matrix_row ('\\\\' matrix_row)*;
matrix_row: expr (('&' | ',') expr)*;



additive: additive (ADD | SUB | PM | CMD_CUP | CMD_CAP | OR) additive | mp;

// mult part
mp:
	mp (MUL | CMD_TIMES | CMD_CDOT | DIV | CMD_DIV | COLON | SET_MINUS) mp
	| operator;

operator: operator BINARY_OPERATORS operator | unary;

mp_nofunc:
	mp_nofunc (
		MUL
		| CMD_TIMES
		| CMD_CDOT
		| DIV
		| CMD_DIV
		| COLON
	) mp_nofunc
	| unary_nofunc;

unary: (ADD | SUB) unary | (LETTER CHOOSE LETTER) | L_BRACE expr CHOOSE expr R_BRACE| postfix+;

unary_nofunc:
	(ADD | SUB) unary_nofunc
	| postfix postfix_nofunc*;

postfix: exp postfix_op*;
postfix_nofunc: exp_nofunc postfix_op*;
postfix_op: BANG | eval_at;

eval_at:
	BAR (eval_at_sup | eval_at_sub | eval_at_sup eval_at_sub);

eval_at_sub: UNDERSCORE L_BRACE (expr | equality) R_BRACE;

eval_at_sup: CARET L_BRACE (expr | equality) R_BRACE;

exp: exp CARET (atom | L_BRACE expr R_BRACE) subexpr? | comp;

exp_nofunc:
	exp_nofunc CARET (atom | L_BRACE expr R_BRACE) subexpr?
	| comp_nofunc;

//quantorized: (('\\forall' | '\\exists') (LETTER | SYMBOL | relation))+ COLON? relation;

comp:
	group
	//| quantorized
	| abs_group
	| func
	| atom
	| floor
	| ceil;

comp_nofunc:
	group
	| abs_group
	| atom
	| floor
	| ceil;

group:
	L_PAREN expr R_PAREN
	| L_BRACKET expr R_BRACKET SINGLE_QUOTES?
	| ('\\phantom' | '\\hspace' | '\\vspace' | '\\textbf' | '\\mbox')? L_BRACE expr R_BRACE
	| L_BRACE_LITERAL expr R_BRACE_LITERAL;

abs_group: BAR expr BAR | L_VERT expr R_VERT;

number: DIGIT+ (',' DIGIT DIGIT DIGIT)* ('.' DIGIT+)? PERCENTAGE?;

special_symbols: RIGHT_ARROW;

modifier: MODIFIER L_BRACE expr R_BRACE;

tuple: L_PAREN expr (COMMA expr)+ R_PAREN;

atom:
	frac
    | (LETTER | SYMBOL | GREEK | 'c' | 'r' | 'l' | 'E') (subexpr? SINGLE_QUOTES? | SINGLE_QUOTES? subexpr?)
    | DERIVATIVE_SYMBOLS
	| env
	| number
	| DIFFERENTIAL
	| mathit
	| binom
	| bra
	| bars
	| ket
	| modifier
	| IMAGINARY_UNIT
	| INFINITY
	| text
	| tuple
	| DOTS;

probability: PROBABILITY (L_PAREN | L_BRACKET) (LETTER subexpr? | (LETTER EQUAL LETTER subexpr?) | (atom BAR atom)) (R_PAREN | R_BRACKET);

bars: L_BAR expr R_BAR;
bra: L_ANGLE expr (R_BAR | BAR);
ket: (L_BAR | BAR) expr R_ANGLE;

mathit: CMD_MATHIT L_BRACE mathit_text R_BRACE;
mathit_text: LETTER*;

frac:
    //CMD_FRAC L_BRACE DERIVATIVE_SYMBOL R_BRACE L_BRACE DIFFERENTIAL R_BRACE |
    CMD_FRAC (upperd = DIGIT | L_BRACE upper = expr R_BRACE) (lowerd = DIGIT | lowers = (LETTER | 'r' | 'c' | 'l') | L_BRACE lower = expr R_BRACE)
    //| derivative
    | L_BRACE expr FUNC_OVER expr R_BRACE
    ;

binom:
	(CMD_BINOM | CMD_DBINOM | CMD_TBINOM) L_BRACE n = expr R_BRACE L_BRACE k = expr R_BRACE;

floor: L_FLOOR val = expr R_FLOOR;
ceil: L_CEIL val = expr R_CEIL;


func_normal:
	FUNC_EXP
	| FUNC_LOG
	| FUNC_LG
	| FUNC_LN
	| FUNC_SIN
	| FUNC_COS
	| FUNC_TAN
	| FUNC_CSC
	| FUNC_SEC
	| FUNC_COT
	| FUNC_ARCSIN
	| FUNC_ARCCOS
	| FUNC_ARCTAN
	| FUNC_ARCCSC
	| FUNC_ARCSEC
	| FUNC_ARCCOT
	| FUNC_SINH
	| FUNC_COSH
	| FUNC_TANH
	| FUNC_ARSINH
	| FUNC_ARCOSH
	| FUNC_ARTANH
	| FUNC_DET
	| FUNC_RE
	| FUNC_IM;

func_bracket: FUNC_VAR | FUNC_EXPECTED_VALUE;

func_binary_bracket:
	FUNC_COV;

//derivative_power: CARET (SYMBOL | LETTER | number);
//derivative: (DERIVATIVE_SYMBOL derivative_power? '/' DIFFERENTIAL derivative_power?)
//derivative: (DERIVATIVE_SYMBOL derivative_power? '/' DIFFERENTIAL derivative_power?)
    //| ('\\frac{' DERIVATIVE_SYMBOL derivative_power? '}{' DIFFERENTIAL derivative_power? '}');

func:
	DDX? (func_normal (subexpr? supexpr? | supexpr? subexpr?) (
		(L_PAREN | L_BRACE) func_arg (R_PAREN | R_BRACE)
		| func_arg_noparens
	)
	| func_bracket (
		(L_PAREN | L_BRACKET) func_arg (R_PAREN | R_BRACKET)
		| func_arg_noparens
	)
    | func_binary_bracket(
		(L_BRACKET | L_PAREN) func_arg ',' func_arg (R_BRACKET | R_PAREN)
	)
	| (LETTER | SYMBOL | GREEK ) (subexpr? (SINGLE_QUOTES? | CARET L_BRACE L_PAREN atom R_PAREN R_BRACE) | SINGLE_QUOTES? subexpr?)  supexpr? // e.g. f(x), f_1'(x)
	L_PAREN args R_PAREN // | (OPERATORNAME L_BRACE LETTER R_BRACE)
	| FUNC_INT (subexpr supexpr | supexpr subexpr)? (
		additive? DIFFERENTIAL
		| frac
		| additive
	)
	| FUNC_SQRT (L_BRACKET root = expr R_BRACKET)? L_BRACE base = expr R_BRACE
	| FUNC_OVERLINE L_BRACE base = expr R_BRACE
	| (FUNC_SUM | FUNC_PROD) (subeq supexpr | supexpr subeq | subeq) mp
	| (FUNC_LIM) limit_sub mp)
	| (FUNC_MAX | FUNC_MIN | FUNC_SUP | FUNC_INF) subexpr? (L_PAREN | L_BRACE) (formulas | set) (R_PAREN | R_BRACE)
	| DDX group
	| probability;

args: (expr ',' args) | expr;

limit_sub:
	UNDERSCORE L_BRACE (LETTER | SYMBOL) LIM_APPROACH_SYM expr (
		CARET ((L_BRACE (ADD | SUB) R_BRACE) | ADD | SUB)
	)? R_BRACE;

func_arg: expr | (expr ',' func_arg);
func_arg_noparens: mp_nofunc;

subexpr: UNDERSCORE (atom | L_BRACE (expr | variable) R_BRACE | L_BRACE expr ',' expr R_BRACE);
supexpr: CARET (atom | L_BRACE expr R_BRACE);

subeq: UNDERSCORE L_BRACE (equality | atom text?) R_BRACE;
supeq: UNDERSCORE L_BRACE equality R_BRACE;
