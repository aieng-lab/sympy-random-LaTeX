lexer grammar LaTeXLexer;

TEXT_OPEN : '\\text{' -> pushMode(TextMode);

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

NORM: '\\|' | '||';

IMAGINARY_UNIT: '\\text{i}' | '\\mathrm{i}'; // not 'i' here, since this is deduced by the context

COMMA: ',';


mode TextMode;

TEXT_CLOSE: '}' -> popMode;
TEXT_BODY: ~[{}]+;
SPACE: [ \t\r\n]+;