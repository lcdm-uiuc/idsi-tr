<span style="font-variant:small-caps;">Illinois Data Science
Initiative</span>\
\[1.5cm\]<span style="font-variant:small-caps;">Technical
Reports</span>\
\[0.4cm\][****]{}\
\
\[1.5cm\]*Author:* Cameron Dart\
\[3cm\]

Introduction
============

This template is designed to assist with creating a two-column research
article or letter to submit to *iDSI*.

If you have a question while using this template on writeLaTeX, please
use the help menu (“?”) on the top bar to search for help or ask us a
question using the option in the lower right of the editor.

Examples of Article Components {#sec:examples}
==============================

The sections below show examples of different article components.

Figures and Tables
==================

It is not necessary to place figures and tables at the back of the
manuscript. Figures and tables should be sized as they are to appear in
the final article. Do not include a separate list of figure captions and
table titles.

Figures and Tables should be labelled and referenced in the standard way
using the `\label{}` and `\ref{}` commands.

Sample Figure
-------------

Figure \[fig:false-color\] shows an example figure.

Sample Table
------------

Table \[tab:shape-functions\] shows an example table.

   local node     $\{N\}_m$     $\{\Phi_i\}_m$ $(i=x,y,z)$
  ------------ --------------- ----------------------------
    $m = 1$     $L_1(2L_1-1)$          $\Phi_{i1}$
    $m = 2$     $L_2(2L_2-1)$          $\Phi_{i2}$
    $m = 3$     $L_3=4L_1L_2$          $\Phi_{i3}$

  : **Shape Functions for Quadratic Line Elements**

\[tab:shape-functions\]

Sample Equation
===============

Let $X_1, X_2, \ldots, X_n$ be a sequence of independent and identically
distributed random variables with $\text{E}[X_i] = \mu$ and
$\text{Var}[X_i] = \sigma^2 < \infty$, and let
$$S_n = \frac{X_1 + X_2 + \cdots + X_n}{n}
      = \frac{1}{n}\sum_{i}^{n} X_i
\label{eq:refname1}$$ denote their mean. Then as $n$ approaches
infinity, the random variables $\sqrt{n}(S_n - \mu)$ converge in
distribution to a normal $\mathcal{N}(0, \sigma^2)$.

Sample Algorithm
================

Algorithms can be included using the commands as shown in algorithm
\[alg:euclid\].

$r\gets a\bmod b$ $a\gets b$ $b\gets r$ $r\gets a\bmod b$
\[euclidendwhile\] **return** $b$

Funding Information {#funding-information .unnumbered}
===================

National Science Foundation (NSF) (1263236, 0968895, 1102301); The 863
Program (2013AA014402).

Acknowledgments {#acknowledgments .unnumbered}
===============

Formal funding declarations should not be included in the
acknowledgments but in a Funding Information section as shown above. The
acknowledgments may contain information that is not related to funding:

The authors thank H. Haase, C. Wiede, and J. Gabler for technical
support.

Supplemental Documents {#supplemental-documents .unnumbered}
======================

*Optica* authors may include supplemental documents with the primary
manuscript. For details, see [Supplementary Materials in
Optica](http://www.opticsinfobase.org/submit/style/supplementary-materials-optica.cfm).
To reference the supplementary document, the statement “See Supplement 1
for supporting content.” should appear at the bottom of the manuscript
(above the references).

References {#references .unnumbered}
==========

For references, you may add citations manually or use BibTeX. E.g.
[@Zhang:14].

Note that letter submissions to *Optica* use an abbreviated reference
style. Citations to journal articles should omit the article title and
final page number; this abbreviated reference style is produced
automatically when the `\setminussetboolean{shortarticle}{true}` option
is selected in the template, if you are using a .bib file for your
references.

However, full references (to aid the editor and reviewers) must be
included as well on an informational page that will not count against
page length; again this will be produced automatically if you are using
a .bib file and have the `\setminussetboolean{shortarticle}{true}`
option selected.
