##
## Copyright (c) 2006-2019 of Toni Giorgino
##
## This file is part of the DTW package.
##
## DTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with DTW.  If not, see <http://www.gnu.org/licenses/>.
##


# Author: Toni Giorgino 2018
#
# If you use this software in academic work, please cite:
#  * T. Giorgino. Computing and Visualizing Dynamic Time Warping
#    Alignments in R: The dtw Package. Journal of Statistical
#    Software, v. 31, Issue 7, p. 1 - 24, aug. 2009. ISSN
#    1548-7660. doi:10.18637/jss.v031.i07. http://www.jstatsoft.org/v31/i07/

"""Main dtw module"""

import numpy
import sys

##
## Copyright (c) 2006-2019 of Toni Giorgino
##
## This file is part of the DTW package.
##
## DTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with DTW.  If not, see <http://www.gnu.org/licenses/>.
##


import numpy

_INT_MIN = numpy.iinfo(numpy.int32).min

# This is O(n). Let's not make it unreadable.
def _backtrack(gcm):
    n = gcm.costMatrix.shape[0]
    m = gcm.costMatrix.shape[1]
    i = n - 1
    j = gcm.jmin

    # Drop null deltas
    dir = gcm.stepPattern.mx
    dir = dir[numpy.bitwise_or(dir[:, 1] != 0,
                               dir[:, 2] != 0), :]

    # Split by 1st column
    npat = gcm.stepPattern.get_n_patterns()
    stepsCache = dict()
    for q in range(1, npat + 1):
        tmp = dir[dir[:, 0] == q,]
        stepsCache[q] = numpy.array(tmp[:, [1, 2]],
                                    dtype=int)
        stepsCache[q] = numpy.flip(stepsCache[q],0)

    # Mapping lists
    iis = [i]
    ii = [i]
    jjs = [j]
    jj = [j]
    ss = []

    while True:
        if i == 0 and j == 0: break

        # Direction taken, 1-based
        s = gcm.directionMatrix[i, j]

        if s == _INT_MIN: break  # int nan in R

        # undo the steps
        ss.insert(0, s)
        steps = stepsCache[s]

        ns = steps.shape[0]
        for k in range(ns):
            ii.insert(0, i - steps[k, 0])
            jj.insert(0, j - steps[k, 1])

        i -= steps[k, 0]
        j -= steps[k, 1]

        iis.insert(0, i)
        jjs.insert(0, j)

    out = {'index1': numpy.array(ii),
           'index2': numpy.array(jj),
           'index1s': numpy.array(iis),
           'index2s': numpy.array(jjs),
           'stepsTaken': numpy.array(ss)}

    return (out)
from dtw._dtw_utils import _computeCM_wrapper


def _globalCostMatrix(lm,
                      step_pattern,
                      window_function,
                      seed,
                      win_args):

    ITYPE = numpy.int32
    n, m = lm.shape

    if window_function == noWindow:  # for performance
        wm = numpy.full_like(lm, 1, dtype=ITYPE)
    else:
        ix, jx = numpy.indices((n,m))
        wm = window_function(ix, jx,
                             query_size=n,
                             reference_size=m,
                             **win_args)
        wm = wm.astype(ITYPE)   # Convert False/True to 0/1

    nsteps = numpy.array([step_pattern.get_n_rows()], dtype=ITYPE)

    dir = numpy.array(step_pattern._get_p(), dtype=numpy.double)

    if seed is not None:
        cm = seed
    else:
        cm = numpy.full_like(lm, numpy.nan, dtype=numpy.double)
        cm[0, 0] = lm[0, 0]

    # All input arguments
    out = _computeCM_wrapper(wm,
                             lm,
                             nsteps,
                             dir,
                             cm)

    out['stepPattern'] = step_pattern;
    return out


def _test_computeCM2(TS=5):
    import numpy as np
    ITYPE = np.int32

    twm = np.ones((TS, TS), dtype=ITYPE)

    tlm = np.zeros((TS, TS), dtype=np.double)
    for i in range(TS):
        for j in range(TS):
            tlm[i, j] = (i + 1) * (j + 1)

    tnstepsp = np.array([6], dtype=ITYPE)

    tdir = np.array((1, 1, 2, 2, 3, 3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, -1, 1, -1, 1, -1, 1),
                    dtype=np.double)

    tcm = np.full_like(tlm, np.nan, dtype=np.double)
    tcm[0, 0] = tlm[0, 0]

    out = _computeCM_wrapper(twm,
                             tlm,
                             tnstepsp,
                             tdir,
                             tcm)
    return out
##
## Copyright (c) 2006-2019 of Toni Giorgino
##
## This file is part of the DTW package.
##
## DTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with DTW.  If not, see <http://www.gnu.org/licenses/>.
##

"""Step Pattern handling

See documentation for the StepPattern class.
"""

import numpy


class StepPattern:
    # IMPORT_RDOCSTRING stepPattern
    """Step patterns for DTW

A ``stepPattern`` object lists the transitions allowed while searching
for the minimum-distance path. DTW variants are implemented by passing
one of the objects described in this page to the ``stepPattern``
argument of the [dtw()] call.

**Details**

A step pattern characterizes the matching model and slope constraint
specific of a DTW variant. They also known as local- or
slope-constraints, transition types, production or recursion rules
(GiorginoJSS).

**Pre-defined step patterns**

::

      ## Well-known step patterns
      symmetric1
      symmetric2
      asymmetric

      ## Step patterns classified according to Rabiner-Juang (Rabiner1993)
      rabinerJuangStepPattern(type,slope_weighting="d",smoothed=False)

      ## Slope-constrained step patterns from Sakoe-Chiba (Sakoe1978)
      symmetricP0;  asymmetricP0
      symmetricP05; asymmetricP05
      symmetricP1;  asymmetricP1
      symmetricP2;  asymmetricP2

      ## Step patterns classified according to Rabiner-Myers (Myers1980)
      typeIa;   typeIb;   typeIc;   typeId;
      typeIas;  typeIbs;  typeIcs;  typeIds;  # smoothed
      typeIIa;  typeIIb;  typeIIc;  typeIId;
      typeIIIc; typeIVc;

      ## Miscellaneous
      mori2006;
      rigid;

A variety of classification schemes have been proposed for step
patterns, including Sakoe-Chiba (Sakoe1978); Rabiner-Juang
(Rabiner1993); and Rabiner-Myers (Myers1980). The ``dtw`` package
implements all of the transition types found in those papers, with the
exception of Itakura’s and Velichko-Zagoruyko’s steps, which require
subtly different algorithms (this may be rectified in the future).
Itakura recursion is almost, but not quite, equivalent to ``typeIIIc``.

For convenience, we shall review pre-defined step patterns grouped by
classification. Note that the same pattern may be listed under different
names. Refer to paper (GiorginoJSS) for full details.

**1. Well-known step patterns**

Common DTW implementations are based on one of the following transition
types.

``symmetric2`` is the normalizable, symmetric, with no local slope
constraints. Since one diagonal step costs as much as the two equivalent
steps along the sides, it can be normalized dividing by ``N+M``
(query+reference lengths). It is widely used and the default.

``asymmetric`` is asymmetric, slope constrained between 0 and 2. Matches
each element of the query time series exactly once, so the warping path
``index2~index1`` is guaranteed to be single-valued. Normalized by ``N``
(length of query).

``symmetric1`` (or White-Neely) is quasi-symmetric, no local constraint,
non-normalizable. It is biased in favor of oblique steps.

**2. The Rabiner-Juang set**

A comprehensive table of step patterns is proposed in Rabiner-Juang’s
book (Rabiner1993), tab. 4.5. All of them can be constructed through the
``rabinerJuangStepPattern(type,slope_weighting,smoothed)`` function.

The classification foresees seven families, labelled with Roman numerals
I-VII; here, they are selected through the integer argument ``type``.
Each family has four slope weighting sub-types, named in sec. 4.7.2.5 as
“Type (a)” to “Type (d)”; they are selected passing a character argument
``slope_weighting``, as in the table below. Furthermore, each subtype
can be either plain or smoothed (figure 4.44); smoothing is enabled
setting the logical argument ``smoothed``. (Not all combinations of
arguments make sense.)

::

     Subtype | Rule       | Norm | Unbiased 
     --------|------------|------|---------
        a    | min step   |  --  |   NO 
        b    | max step   |  --  |   NO 
        c    | Di step    |   N  |  YES 
        d    | Di+Dj step | N+M  |  YES 

**3. The Sakoe-Chiba set**

Sakoe-Chiba (Sakoe1978) discuss a family of slope-constrained patterns;
they are implemented as shown in page 47, table I. Here, they are called
``symmetricP<x>`` and ``asymmetricP<x>``, where ``<x>`` corresponds to
Sakoe’s integer slope parameter *P*. Values available are accordingly:
``0`` (no constraint), ``1``, ``05`` (one half) and ``2``. See
(Sakoe1978) for details.

**4. The Rabiner-Myers set**

The ``type<XX><y>`` step patterns follow the older Rabiner-Myers’
classification proposed in (Myers1980) and (MRR1980). Note that this is
a subset of the Rabiner-Juang set (Rabiner1993), and the latter should
be preferred in order to avoid confusion. ``<XX>`` is a Roman numeral
specifying the shape of the transitions; ``<y>`` is a letter in the
range ``a-d`` specifying the weighting used per step, as above;
``typeIIx`` patterns also have a version ending in ``s``, meaning the
smoothing is used (which does not permit skipping points). The
``typeId, typeIId`` and ``typeIIds`` are unbiased and symmetric.

**5. Others**

The ``rigid`` pattern enforces a fixed unitary slope. It only makes
sense in combination with ``open_begin=True``, ``open_end=True`` to find
gapless subsequences. It may be seen as the ``P->inf`` limiting case in
Sakoe’s classification.

``mori2006`` is Mori’s asymmetric step-constrained pattern (Mori2006).
It is normalized by the matched reference length.

[mvmStepPattern()] implements Latecki’s Minimum Variance Matching
algorithm, and it is described in its own page.

**Methods**

``print_stepPattern`` prints an user-readable description of the
recurrence equation defined by the given pattern.

``plot_stepPattern`` graphically displays the step patterns productions
which can lead to element (0,0). Weights are shown along the step
leading to the corresponding element.

``t_stepPattern`` transposes the productions and normalization hint so
that roles of query and reference become reversed.

Parameters
----------
x : 
    a step pattern object
type : 
    path specification, integer 1..7 (see (Rabiner1993), table 4.5)
slope_weighting : 
    slope weighting rule: character `"a"` to `"d"` (see (Rabiner1993), sec. 4.7.2.5)
smoothed : 
    logical, whether to use smoothing (see (Rabiner1993), fig. 4.44)
... : 
    additional arguments to [print()].

Notes
-----

Constructing ``stepPattern`` objects is tricky and thus undocumented.
For a commented example please see source code for ``symmetricP1``.

References
----------

-  (GiorginoJSS) Toni Giorgino. *Computing and Visualizing Dynamic Time
   Warping Alignments in R: The dtw Package.* Journal of Statistical
   Software, 31(7), 1-24.
   `doi:10.18637/jss_v031.i07 <https://doi.org/10.18637/jss_v031.i07>`__
-  (Itakura1975) Itakura, F., *Minimum prediction residual principle
   applied to speech recognition,* Acoustics, Speech, and Signal
   Processing, IEEE Transactions on , vol.23, no.1, pp. 67-72, Feb 1975.
   `doi:10.1109/TASSP.1975.1162641 <https://doi.org/10.1109/TASSP.1975.1162641>`__
-  (MRR1980) Myers, C.; Rabiner, L. & Rosenberg, A. *Performance
   tradeoffs in dynamic time warping algorithms for isolated word
   recognition*, IEEE Trans. Acoust., Speech, Signal Process., 1980, 28,
   623-635.
   `doi:10.1109/TASSP.1980.1163491 <https://doi.org/10.1109/TASSP.1980.1163491>`__
-  (Mori2006) Mori, A.; Uchida, S.; Kurazume, R.; Taniguchi, R.;
   Hasegawa, T. & Sakoe, H. Early Recognition and Prediction of Gestures
   Proc. 18th International Conference on Pattern Recognition ICPR 2006,
   2006, 3, 560-563.
   `doi:10.1109/ICPR.2006.467 <https://doi.org/10.1109/ICPR.2006.467>`__
-  (Myers1980) Myers, Cory S. *A Comparative Study Of Several Dynamic
   Time Warping Algorithms For Speech Recognition*, MS and BS thesis,
   Dept. of Electrical Engineering and Computer Science, Massachusetts
   Institute of Technology, archived Jun 20 1980,
   https://hdl_handle_net/1721.1/27909
-  (Rabiner1993) Rabiner, L. R., & Juang, B.-H. (1993). *Fundamentals of
   speech recognition.* Englewood Cliffs, NJ: Prentice Hall.
-  (Sakoe1978) Sakoe, H.; Chiba, S., *Dynamic programming algorithm
   optimization for spoken word recognition,* Acoustics, Speech, and
   Signal Processing, IEEE Transactions on , vol.26, no.1, pp. 43-49,
   Feb 1978
   `doi:10.1109/TASSP.1978.1163055 <https://doi.org/10.1109/TASSP.1978.1163055>`__

Examples
--------
>>> from dtw import *
>>> import numpy as np

The usual (normalizable) symmetric step pattern
Step pattern recursion, defined as:
 g[i,j] = min(
   g[i,j-1] + d[i,j] ,
   g[i-1,j-1] + 2 * d[i,j] ,
   g[i-1,j] + d[i,j] ,
)

>>> print(symmetric2)		 #doctest: +NORMALIZE_WHITESPACE
Step pattern recursion:
 g[i,j] = min(
     g[i-1,j-1] + 2 * d[i  ,j  ] ,
     g[i  ,j-1] +     d[i  ,j  ] ,
     g[i-1,j  ] +     d[i  ,j  ] ,
 )
<BLANKLINE>
Normalization hint: N+M
<BLANKLINE>

The well-known plotting style for step patterns

>>> import matplotlib.pyplot as plt;		# doctest: +SKIP
... symmetricP2.plot().set_title("Sakoe's Symmetric P=2 recursion")

Same example seen in ?dtw , now with asymmetric step pattern

>>> (query, reference) = dtw_test_data.sin_cos()

Do the computation

>>> asy = dtw(query, reference, keep_internals=True,
... 	  	     step_pattern=asymmetric);

>>> dtwPlot(asy,type="density"			# doctest: +SKIP
...         ).set_title("Sine and cosine, asymmetric step")

Hand-checkable example given in [Myers1980] p 61 - see JSS paper

>>> tm = numpy.reshape( [1, 3, 4, 4, 5, 2, 2, 3, 3, 4, 3, 1, 1, 1, 3, 4, 2,
...                      3, 3, 2, 5, 3, 4, 4, 1], (5,5), "F" )

"""
    # ENDIMPORT

    def __init__(self, mx, hint="NA"):
        self.mx = numpy.array(mx, dtype=numpy.double)
        self.hint = hint

    def get_n_rows(self):
        """Total number of steps in the recursion."""
        return self.mx.shape[0]

    def get_n_patterns(self):
        """Number of rules in the recursion."""
        return int(numpy.max(self.mx[:, 0]))

    def T(self):
        """Transpose a step pattern."""
        tsp = self
        tsp.mx = tsp.mx[:, [0, 2, 1, 3]]
        if tsp.hint == "N":
            tsp.hint = "M"
        elif tsp.hint == "M":
            tsp.hint = "N"
        return tsp

    def __str__(self):
        np = self.get_n_patterns()
        head = " g[i,j] = min(\n"

        body = ""
        for p in range(1, np + 1):
            steps = self._extractpattern(p)
            ns = steps.shape[0]
            steps = numpy.flip(steps, 0)

            for s in range(ns):
                di, dj, cc = steps[s, :]
                dis = "" if di == 0 else f"{-int(di)}"
                djs = "" if dj == 0 else f"{-int(dj)}"
                dijs = f"i{dis:2},j{djs:2}"

                if cc == -1:
                    gs = f"    g[{dijs}]"
                    body = body + " " + gs
                else:
                    ccs = "    " if cc == 1 else f"{cc:2.2g} *"
                    ds = f"+{ccs} d[{dijs}]"
                    body = body + " " + ds
            body = body + " ,\n"

        tail = " ) \n\n"
        ntxt = f"Normalization hint: {self.hint}\n"

        return "Step pattern recursion:\n" + head + body + tail + ntxt

    def plot(self):
        """Provide a visual description of a StepPattern object"""
        import matplotlib.pyplot as plt
        x = self.mx
        pats = numpy.arange(1, 1 + self.get_n_patterns())

        alpha = .5
        fudge = [0, 0]

        fig, ax = plt.subplots(figsize=(6, 6))
        for i in pats:
            ss = x[:, 0] == i
            ax.plot(-x[ss, 1], -x[ss, 2], lw=2, color="tab:blue")
            ax.plot(-x[ss, 1], -x[ss, 2], 'o', color="black", marker="o", fillstyle="none")

            if numpy.sum(ss) == 1: continue

            xss = x[ss, :]
            xh = alpha * xss[:-1, 1] + (1 - alpha) * xss[1:, 1]
            yh = alpha * xss[:-1, 2] + (1 - alpha) * xss[1:, 2]

            for xx, yy, tt in zip(xh, yh, xss[1:, 3]):
                ax.annotate("{:.2g}".format(tt), (-xx - fudge[0],
                                                  -yy - fudge[1]))

        endpts = x[:, 3] == -1
        ax.plot(-x[endpts, 1], -x[endpts, 2], 'o', color="black")

        ax.set_xlabel("Query index")
        ax.set_ylabel("Reference index")
        ax.set_xticks(numpy.unique(-x[:, 1]))
        ax.set_yticks(numpy.unique(-x[:, 2]))
        return ax

    def _extractpattern(self, sn):
        sp = self.mx
        sbs = sp[:, 0] == sn
        spl = sp[sbs, 1:]
        return numpy.flip(spl, 0)

    def _mkDirDeltas(self):
        out = numpy.array(self.mx, dtype=numpy.int32)
        out = out[out[:, 3] == -1, :]
        out = out[:, [1, 2]]
        return out

    def _get_p(self):
        # Dimensions are reversed wrt R
        s = self.mx[:, [0, 2, 1, 3]]
        return s.T.reshape(-1)



# Alternate constructor for ease of R import
def _c(*v):
    va = numpy.array([*v])
    if len(va) % 4 != 0:
        _error("Internal error in _c constructor")
    va = va.reshape((-1, 4))
    return (va)


# Kludge because lambda: raise doesn't work
def _error(s):
    raise ValueError(s)


##################################################
##################################################

# Reimplementation of the building process

class _P:
    def __init__(self, pid, subtype, smoothing):
        self.subtype = subtype
        self.smoothing = smoothing
        self.pid = pid
        self.i = [0]
        self.j = [0]

    def step(self, di, dj):  # equivalent to .Pstep
        self.i.append(di)
        self.j.append(dj)
        return self

    def get(self):  # eqv to .Pend
        ia = numpy.array(self.i, dtype=numpy.double)
        ja = numpy.array(self.j, dtype=numpy.double)
        si = numpy.cumsum(ia)
        sj = numpy.cumsum(ja)
        ni = numpy.max(si) - si  # ?
        nj = numpy.max(sj) - sj
        if self.subtype == "a":
            w = numpy.minimum(ia, ja)
        elif self.subtype == "b":
            w = numpy.maximum(ia, ja)
        elif self.subtype == "c":
            w = ia
        elif self.subtype == "d":
            w = ia + ja
        else:
            _error("Unsupported subtype")

        if self.smoothing:
            # if self.pid==3:                import ipdb; ipdb.set_trace()
            w[1:] = numpy.mean(w[1:])

        w[0] = -1.0

        nr = len(w)
        mx = numpy.zeros((nr, 4))
        mx[:, 0] = self.pid
        mx[:, 1] = ni
        mx[:, 2] = nj
        mx[:, 3] = w
        return mx


def rabinerJuangStepPattern(ptype, slope_weighting="d", smoothed=False):
    """Construct a pattern classified according to the Rabiner-Juang scheme (Rabiner1993)

See documentation for the StepPattern class.
"""

    f = {
        1: _RJtypeI,
        2: _RJtypeII,
        3: _RJtypeIII,
        4: _RJtypeIV,
        5: _RJtypeV,
        6: _RJtypeVI,
        7: _RJtypeVII
    }.get(ptype, lambda: _error("Invalid type"))

    r = f(slope_weighting, smoothed)
    norm = "NA"
    if slope_weighting == "c":
        norm = "N"
    elif slope_weighting == "d":
        norm = "N+M"

    return StepPattern(r, norm)


def _RJtypeI(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(0, 1).get()])


def _RJtypeII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 1).step(0, 1).get()])


def _RJtypeIII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(2, 1).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 2).get()])


def _RJtypeIV(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 2).step(1, 0).get(),
        _P(3, s, m).step(1, 1).get(),
        _P(4, s, m).step(1, 2).get(),
    ])


def _RJtypeV(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).step(1, 0).get(),
        _P(2, s, m).step(1, 1).step(1, 0).get(),
        _P(3, s, m).step(1, 1).get(),
        _P(4, s, m).step(1, 1).step(0, 1).get(),
        _P(5, s, m).step(1, 1).step(0, 1).step(0, 1).get(),
    ])


def _RJtypeVI(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 1).step(1, 1).step(0, 1).get()
    ])


def _RJtypeVII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).step(1, 0).get(),
        _P(2, s, m).step(1, 2).step(1, 0).step(1, 0).get(),
        _P(3, s, m).step(1, 3).step(1, 0).step(1, 0).get(),
        _P(4, s, m).step(1, 1).step(1, 0).get(),
        _P(5, s, m).step(1, 2).step(1, 0).get(),
        _P(6, s, m).step(1, 3).step(1, 0).get(),
        _P(7, s, m).step(1, 1).get(),
        _P(8, s, m).step(1, 2).get(),
        _P(9, s, m).step(1, 3).get(),
    ])


##########################################################################################
##########################################################################################

## Everything here is semi auto-generated from the R source. Don't
## edit!


##################################################
##################################################


##
## Various step patterns, defined as internal variables
##
## First column: enumerates step patterns.
## Second   	 step in query index
## Third	 step in reference index
## Fourth	 weight if positive, or -1 if starting point
##
## For \cite{} see dtw.bib in the package
##


## Widely-known variants

## White-Neely symmetric (default)
## aka Quasi-symmetric \cite{White1976}
## normalization: no (N+M?)
symmetric1 = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 0, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
));

## Normal symmetric
## normalization: N+M
symmetric2 = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 2,
    2, 0, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N+M");

## classic asymmetric pattern: max slope 2, min slope 0
## normalization: N
asymmetric = StepPattern(_c(
    1, 1, 0, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 0, 1
), "N");

# % \item{\code{symmetricVelichkoZagoruyko}}{symmetric, reproduced from %
# [Sakoe1978]. Use distance matrix \code{1-d}}
# 

## normalization: max[N,M]
## note: local distance matrix is 1-d
## \cite{Velichko}
_symmetricVelichkoZagoruyko = StepPattern(_c(
    1, 0, 1, -1,
    2, 1, 1, -1,
    2, 0, 0, -1.001,
    3, 1, 0, -1));

# % \item{\code{asymmetricItakura}}{asymmetric, slope contrained 0.5 -- 2
# from reference [Itakura1975]. This is the recursive definition % that
# generates the Itakura parallelogram; }
# 

## Itakura slope-limited asymmetric \cite{Itakura1975}
## Max slope: 2; min slope: 1/2
## normalization: N
_asymmetricItakura = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1,
    4, 2, 2, -1,
    4, 1, 0, 1,
    4, 0, 0, 1
));

#############################
## Slope-limited versions
##
## Taken from Table I, page 47 of "Dynamic programming algorithm
## optimization for spoken word recognition," Acoustics, Speech, and
## Signal Processing, vol.26, no.1, pp. 43-49, Feb 1978 URL:
## http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1163055
##
## Mostly unchecked


## Row P=0
symmetricP0 = symmetric2;

## normalization: N ?
asymmetricP0 = StepPattern(_c(
    1, 0, 1, -1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N");

## alternative implementation
_asymmetricP0b = StepPattern(_c(
    1, 0, 1, -1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N");

## Row P=1/2
symmetricP05 = StepPattern(_c(
    1, 1, 3, -1,
    1, 0, 2, 2,
    1, 0, 1, 1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 1, 2,
    2, 0, 0, 1,
    3, 1, 1, -1,
    3, 0, 0, 2,
    4, 2, 1, -1,
    4, 1, 0, 2,
    4, 0, 0, 1,
    5, 3, 1, -1,
    5, 2, 0, 2,
    5, 1, 0, 1,
    5, 0, 0, 1
), "N+M");

asymmetricP05 = StepPattern(_c(
    1, 1, 3, -1,
    1, 0, 2, 1 / 3,
    1, 0, 1, 1 / 3,
    1, 0, 0, 1 / 3,
    2, 1, 2, -1,
    2, 0, 1, .5,
    2, 0, 0, .5,
    3, 1, 1, -1,
    3, 0, 0, 1,
    4, 2, 1, -1,
    4, 1, 0, 1,
    4, 0, 0, 1,
    5, 3, 1, -1,
    5, 2, 0, 1,
    5, 1, 0, 1,
    5, 0, 0, 1
), "N");

## Row P=1
## Implementation of Sakoe's P=1, Symmetric algorithm

symmetricP1 = StepPattern(_c(
    1, 1, 2, -1,  # First branch: g(i-1,j-2)+
    1, 0, 1, 2,  # + 2d(i  ,j-1)
    1, 0, 0, 1,  # +  d(i  ,j)
    2, 1, 1, -1,  # Second branch: g(i-1,j-1)+
    2, 0, 0, 2,  # +2d(i,  j)
    3, 2, 1, -1,  # Third branch: g(i-2,j-1)+
    3, 1, 0, 2,  # + 2d(i-1,j)
    3, 0, 0, 1  # +  d(  i,j)
), "N+M");

asymmetricP1 = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 1, .5,
    1, 0, 0, .5,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1
), "N");

## Row P=2
symmetricP2 = StepPattern(_c(
    1, 2, 3, -1,
    1, 1, 2, 2,
    1, 0, 1, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 3, 2, -1,
    3, 2, 1, 2,
    3, 1, 0, 2,
    3, 0, 0, 1
), "N+M");

asymmetricP2 = StepPattern(_c(
    1, 2, 3, -1,
    1, 1, 2, 2 / 3,
    1, 0, 1, 2 / 3,
    1, 0, 0, 2 / 3,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 3, 2, -1,
    3, 2, 1, 1,
    3, 1, 0, 1,
    3, 0, 0, 1
), "N");

################################
## Taken from Table III, page 49.
## Four varieties of DP-algorithm compared

## 1st row:  asymmetric

## 2nd row:  symmetricVelichkoZagoruyko

## 3rd row:  symmetric1

## 4th row:  asymmetricItakura


#############################
## Classified according to Rabiner
##
## Taken from chapter 2, Myers' thesis [4]. Letter is
## the weighting function:
##
##      rule       norm   unbiased
##   a  min step   ~N     NO
##   b  max step   ~N     NO
##   c  x step     N      YES
##   d  x+y step   N+M    YES
##
## Mostly unchecked

# R-Myers     R-Juang
# type I      type II   
# type II     type III
# type III    type IV
# type IV     type VII


typeIa = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 0
));

typeIb = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 1
));

typeIc = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 0
), "N");

typeId = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 1, 2, -1,
    3, 0, 1, 2,
    3, 0, 0, 1
), "N+M");

## ----------
## smoothed variants of above

typeIas = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, .5,
    1, 0, 0, .5,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, .5,
    3, 0, 0, .5
));

typeIbs = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 1
));

typeIcs = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, .5,
    3, 0, 0, .5
), "N");

typeIds = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1.5,
    1, 0, 0, 1.5,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 1, 2, -1,
    3, 0, 1, 1.5,
    3, 0, 0, 1.5
), "N+M");

## ----------

typeIIa = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 0, 0, 1
));

typeIIb = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 2,
    3, 2, 1, -1,
    3, 0, 0, 2
));

typeIIc = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 0, 0, 2
), "N");

typeIId = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 2,
    2, 1, 2, -1,
    2, 0, 0, 3,
    3, 2, 1, -1,
    3, 0, 0, 3
), "N+M");

## ----------

## Rabiner [3] discusses why this is not equivalent to Itakura's

typeIIIc = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1,
    4, 2, 2, -1,
    4, 1, 0, 1,
    4, 0, 0, 1
), "N");

## ----------

## numbers follow as production rules in fig 2.16

typeIVc = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 1, 3, -1,
    3, 0, 0, 1,
    4, 2, 1, -1,
    4, 1, 0, 1,
    4, 0, 0, 1,
    5, 2, 2, -1,
    5, 1, 0, 1,
    5, 0, 0, 1,
    6, 2, 3, -1,
    6, 1, 0, 1,
    6, 0, 0, 1,
    7, 3, 1, -1,
    7, 2, 0, 1,
    7, 1, 0, 1,
    7, 0, 0, 1,
    8, 3, 2, -1,
    8, 2, 0, 1,
    8, 1, 0, 1,
    8, 0, 0, 1,
    9, 3, 3, -1,
    9, 2, 0, 1,
    9, 1, 0, 1,
    9, 0, 0, 1
), "N");

#############################
## 
## Mori's asymmetric step-constrained pattern. Normalized in the
## reference length.
##
## Mori, A.; Uchida, S.; Kurazume, R.; Taniguchi, R.; Hasegawa, T. &
## Sakoe, H. Early Recognition and Prediction of Gestures Proc. 18th
## International Conference on Pattern Recognition ICPR 2006, 2006, 3,
## 560-563
##

mori2006 = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 3,
    3, 1, 2, -1,
    3, 0, 1, 3,
    3, 0, 0, 3
), "M");

## Completely unflexible: fixed slope 1. Only makes sense with
## open.begin and open.end
rigid = StepPattern(_c(1, 1, 1, -1,
                       1, 0, 0, 1), "N")

##
## Copyright (c) 2006-2019 of Toni Giorgino
##
## This file is part of the DTW package.
##
## DTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with DTW.  If not, see <http://www.gnu.org/licenses/>.
##


# IMPORT_RDOCSTRING dtwWindowingFunctions
"""Global constraints and windowing functions for DTW

Various global constraints (windows) which can be applied to the
``window_type`` argument of [dtw()], including the Sakoe-Chiba band, the
Itakura parallelogram, and custom functions.

**Details**

Windowing functions can be passed to the ``window_type`` argument in
[dtw()] to put a global constraint to the warping paths allowed. They
take two integer arguments (plus optional parameters) and must return a
boolean value ``True`` if the coordinates fall within the allowed region
for warping paths, ``False`` otherwise.

User-defined functions can read variables ``reference_size``,
``query_size`` and ``window_size``; these are pre-set upon invocation.
Some functions require additional parameters which must be set (e_g.
``window_size``). User-defined functions are free to implement any
window shape, as long as at least one path is allowed between the
initial and final alignment points, i_e., they are compatible with the
DTW constraints.

The ``sakoeChibaWindow`` function implements the Sakoe-Chiba band, i_e.
``window_size`` elements around the ``main`` diagonal. If the window
size is too small, i_e. if ``reference_size``-``query_size`` >
``window_size``, warping becomes impossible.

An ``itakuraWindow`` global constraint is still provided with this
package. See example below for a demonstration of the difference between
a local the two.

The ``slantedBandWindow`` (package-specific) is a band centered around
the (jagged) line segment which joins element ``[1,1]`` to element
``[query_size,reference_size]``, and will be ``window_size`` columns
wide. In other words, the “diagonal” goes from one corner to the other
of the possibly rectangular cost matrix, therefore having a slope of
``M/N``, not 1.

``dtwWindow_plot`` visualizes a windowing function. By default it plots
a 200 x 220 rectangular region, which can be changed via
``reference_size`` and ``query_size`` arguments.

Parameters
----------
iw : 
    index in the query (row) -- automatically set
jw : 
    index in the reference (column) -- automatically set
query_size : 
    size of the query time series -- automatically set
reference_size : 
    size of the reference time series -- automatically set
window_size : 
    window size, used by some windowing functions -- must be set
fun : 
    a windowing function
... : 
    additional arguments passed to windowing functions

Returns
-------

Windowing functions return ``True`` if the coordinates passed as
arguments fall within the chosen warping window, ``False`` otherwise.
User-defined functions should do the same.

Notes
-----

Although ``dtwWindow_plot`` resembles object-oriented notation, there is
not a such a dtwWindow class currently.

A widely held misconception is that the “Itakura parallelogram” (as
described in reference 2) is a *global* constraint, i_e. a window. To
the author’s knowledge, it instead arises from the local slope
restrictions imposed to the warping path, such as the one implemented by
the [typeIIIc()] step pattern.

References
----------

1. Sakoe, H.; Chiba, S., *Dynamic programming algorithm optimization for
   spoken word recognition,* Acoustics, Speech, and Signal Processing,
   IEEE Transactions on , vol.26, no.1, pp. 43-49, Feb 1978
   `doi:10.1109/TASSP.1978.1163055 <https://doi.org/10.1109/TASSP.1978.1163055>`__
2. Itakura, F., *Minimum prediction residual principle applied to speech
   recognition,* Acoustics, Speech, and Signal Processing, IEEE
   Transactions on , vol.23, no.1, pp. 67-72, Feb 1975.
   `doi:10.1109/TASSP.1975.1162641 <https://doi.org/10.1109/TASSP.1975.1162641>`__

Examples
--------
>>> from dtw import *
>>> import numpy as np

Default test data

>>> (query, reference) = dtw_test_data.sin_cos()

Asymmetric step with Sakoe-Chiba band

>>> asyband = dtw(query,reference,
...     keep_internals=True, step_pattern=asymmetric,
...     window_type=sakoeChibaWindow,
...     window_args={'window_size': 30}                  );

>>> dtwPlot(asyband,type="density")  # doctest: +SKIP

Display some windowing functions 

>>> #TODO dtwWindow_plot(itakuraWindow, main="So-called Itakura parallelogram window")
>>> #TODO dtwWindow_plot(slantedBandWindow, window_size=2,
>>> #TODO reference=13, query=17, main="The slantedBandWindow at window_size=2")

"""
# ENDIMPORT



# The functions must be vectorized! The first 2 args are matrices of row and column indices.

def noWindow(iw, jw, query_size, reference_size):
    return (iw | True)


def sakoeChibaWindow(iw, jw, query_size, reference_size, window_size):
    ok = abs(jw - iw) <= window_size
    return ok


def itakuraWindow(iw, jw, query_size, reference_size):
    n = query_size
    m = reference_size
    ok = (jw < 2 * iw) & \
         (iw <= 2 * jw) & \
         (iw >= n - 1 - 2 * (m - jw)) & \
         (jw > m - 1 - 2 * (n - iw))
    return ok


def slantedBandWindow(iw, jw, query_size, reference_size, window_size):
    diagj = (iw * reference_size / query_size)
    return abs(jw - diagj) <= window_size;


import scipy.spatial.distance
##
## Copyright (c) 2006-2019 of Toni Giorgino
##
## This file is part of the DTW package.
##
## DTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with DTW.  If not, see <http://www.gnu.org/licenses/>.
##

"""DTW plotting functions"""

import numpy

def dtwPlot(x, type="alignment", **kwargs):
    # IMPORT_RDOCSTRING plot.dtw
    """Plotting of dynamic time warp results

Methods for plotting dynamic time warp alignment objects returned by
[dtw()].

**Details**

``dtwPlot`` displays alignment contained in ``dtw`` objects.

Various plotting styles are available, passing strings to the ``type``
argument (may be abbreviated):

-  ``alignment`` plots the warping curve in ``d``;
-  ``twoway`` plots a point-by-point comparison, with matching lines;
   see [dtwPlotTwoWay()];
-  ``threeway`` vis-a-vis inspection of the timeseries and their warping
   curve; see [dtwPlotThreeWay()];
-  ``density`` displays the cumulative cost landscape with the warping
   path overimposed; see [dtwPlotDensity()]

Additional parameters are passed to the plotting functions: use with
care.

Parameters
----------
x,d : 
    `dtw` object, usually result of call to [dtw()]
xlab : 
    label for the query axis
ylab : 
    label for the reference axis
type : 
    general style for the plot, see below
plot_type : 
    type of line to be drawn, used as the `type` argument in the underlying `plot` call
... : 
    additional arguments, passed to plotting functions

"""
    # ENDIMPORT

    if type == "alignment":
        return dtwPlotAlignment(x, **kwargs)
    elif type == "twoway":
        return dtwPlotTwoWay(x, **kwargs)
    elif type == "threeway":
        return dtwPlotThreeWay(x, **kwargs)
    elif type == "density":
        return dtwPlotDensity(x, **kwargs)
    else:
        raise ValueError("Unknown plot type: " + type)


def dtwPlotAlignment(d, xlab="Query index", ylab="Reference index", **kwargs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(d.index1, d.index2, **kwargs)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    return ax


def dtwPlotTwoWay(d, xts=None, yts=None,
                  offset=0,
                  ts_type="l",
                  match_indices=None,
                  match_col="gray",
                  xlab="Index",
                  ylab="Query value",
                  **kwargs):
    # IMPORT_RDOCSTRING dtwPlotTwoWay
    """Plotting of dynamic time warp results: pointwise comparison

Display the query and reference time series and their alignment,
arranged for visual inspection.

**Details**

The two vectors are displayed via the [matplot()] functions; their
appearance can be customized via the ``type`` and ``pch`` arguments
(constants or vectors of two elements). If ``offset`` is set, the
reference is shifted vertically by the given amount; this will be
reflected by the *right-hand* axis.

Argument ``match_indices`` is used to draw a visual guide to matches; if
a vector is given, guides are drawn for the corresponding indices in the
warping curve (match lines). If integer, it is used as the number of
guides to be plotted. The corresponding style is customized via the
``match_col`` and ``match_lty`` arguments.

If ``xts`` and ``yts`` are not supplied, they will be recovered from
``d``, as long as it was created with the two-argument call of [dtw()]
with ``keep_internals=True``. Only single-variate time series can be
plotted this way.

Parameters
----------
d : 
    an alignment result, object of class `dtw`
xts : 
    query vector
yts : 
    reference vector
xlab,ylab : 
    axis labels
offset : 
    displacement between the timeseries, summed to reference
match_col,match_lty : 
    color and line type of the match guide lines
match_indices : 
    indices for which to draw a visual guide
ts_type,pch : 
    graphical parameters for timeseries plotting, passed to `matplot`
... : 
    additional arguments, passed to `matplot`

Notes
-----

When ``offset`` is set values on the left axis only apply to the query.

"""
    # ENDIMPORT

    import matplotlib.pyplot as plt
    from matplotlib import collections  as mc

    if xts is None or yts is None:
        try:
            xts = d.query
            yts = d.reference
        except:
            raise ValueError("Original timeseries are required")

    # ytso = yts + offset
    offset = -offset

    xtimes = numpy.arange(len(xts))
    ytimes = numpy.arange(len(yts))

    fig, ax = plt.subplots()
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    ax.plot(xtimes, numpy.array(xts), color='k', **kwargs)
    ax.plot(ytimes, numpy.array(yts) - offset, **kwargs)      # Plot with offset applied

    if offset != 0:
        # Create an offset axis
        ax2 = ax.twinx()
        ax2.tick_params('y', colors='b')
        ql, qh = ax.get_ylim()
        ax2.set_ylim(ql + offset, qh + offset)

    # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
    if match_indices is None:
        idx = numpy.linspace(0, len(d.index1) - 1)
    elif not hasattr(match_indices, "__len__"):
        idx = numpy.linspace(0, len(d.index1) - 1, num=match_indices)
    else:
        idx = match_indices
    idx = numpy.array(idx).astype(int)

    col = []
    for i in idx:
        col.append([(d.index1[i], xts[d.index1[i]]),
                    (d.index2[i], -offset + yts[d.index2[i]])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    ax.add_collection(lc)

    return ax


def dtwPlotThreeWay(d, xts=None, yts=None,
                    match_indices=None,
                    match_col="gray",
                    xlab="Query index",
                    ylab="Reference index", **kwargs):
    # IMPORT_RDOCSTRING dtwPlotThreeWay
    """Plotting of dynamic time warp results: annotated warping function

Display the query and reference time series and their warping curve,
arranged for visual inspection.

**Details**

The query time series is plotted in the bottom panel, with indices
growing rightwards and values upwards. Reference is in the left panel,
indices growing upwards and values leftwards. The warping curve panel
matches indices, and therefore element (1,1) will be at the lower left,
(N,M) at the upper right.

Argument ``match_indices`` is used to draw a visual guide to matches; if
a vector is given, guides are drawn for the corresponding indices in the
warping curve (match lines). If integer, it is used as the number of
guides to be plotted. The corresponding style is customized via the
``match_col`` and ``match_lty`` arguments.

If ``xts`` and ``yts`` are not supplied, they will be recovered from
``d``, as long as it was created with the two-argument call of [dtw()]
with ``keep_internals=True``. Only single-variate time series can be
plotted.

Parameters
----------
d : 
    an alignment result, object of class `dtw`
xts : 
    query vector
yts : 
    reference vector
xlab : 
    label for the query axis
ylab : 
    label for the reference axis
main : 
    main title
type_align : 
    line style for warping curve plot
type_ts : 
    line style for timeseries plot
match_indices : 
    indices for which to draw a visual guide
margin : 
    outer figure margin
inner_margin : 
    inner figure margin
title_margin : 
    space on the top of figure
... : 
    additional arguments, used for the warping curve

"""
    # ENDIMPORT
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import collections  as mc

    if xts is None or yts is None:
        try:
            xts = d.query
            yts = d.reference
        except:
            raise ValueError("Original timeseries are required")

    nn = len(xts)
    mm = len(yts)
    nn1 = numpy.arange(nn)
    mm1 = numpy.arange(mm)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 3],
                           height_ratios=[3, 1])
    axr = plt.subplot(gs[0])
    ax = plt.subplot(gs[1])
    axq = plt.subplot(gs[3])

    axq.plot(nn1, xts)  # query, horizontal, bottom
    axq.set_xlabel(xlab)

    axr.plot(yts, mm1)  # ref, vertical
    axr.invert_xaxis()
    axr.set_ylabel(ylab)

    ax.plot(d.index1, d.index2)

    if match_indices is None:
        idx = []
    elif not hasattr(match_indices, "__len__"):
        idx = numpy.linspace(0, len(d.index1) - 1, num=match_indices)
    else:
        idx = match_indices
    idx = numpy.array(idx).astype(int)

    col = []
    for i in idx:
        col.append([(d.index1[i], 0),
                    (d.index1[i], d.index2[i])])
        col.append([(0, d.index2[i]),
                    (d.index1[i], d.index2[i])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    ax.add_collection(lc)

    return ax


def dtwPlotDensity(d, normalize=False,
                   xlab="Query index",
                   ylab="Reference index", **kwargs):
    # IMPORT_RDOCSTRING dtwPlotDensity
    """Display the cumulative cost density with the warping path overimposed

The plot is based on the cumulative cost matrix. It displays the optimal
alignment as a “ridge” in the global cost landscape.

**Details**

The alignment must have been constructed with the
``keep_internals=True`` parameter set.

If ``normalize`` is ``True``, the *average* cost per step is plotted
instead of the cumulative one. Step averaging depends on the
[stepPattern()] used.

Parameters
----------
d : 
    an alignment result, object of class `dtw`
normalize : 
    show per-step average cost instead of cumulative cost
xlab : 
    label for the query axis
ylab : 
    label for the reference axis
... : 
    additional parameters forwarded to plotting functions

Examples
--------
>>> from dtw import *

A study of the "Itakura" parallelogram

A widely held misconception is that the "Itakura parallelogram" (as
described in the original article) is a global constraint.  Instead,
it arises from local slope restrictions. Anyway, an "itakuraWindow",
is provided in this package. A comparison between the two follows.

The local constraint: three sides of the parallelogram are seen

>>> (query, reference) = dtw_test_data.sin_cos()
>>> ita = dtw(query, reference, keep_internals=True, step_pattern=typeIIIc)

>>> dtwPlotDensity(ita)				     # doctest: +SKIP

Symmetric step with global parallelogram-shaped constraint. Note how
long (>2 steps) horizontal stretches are allowed within the window.

>>> ita = dtw(query, reference, keep_internals=True, window_type=itakuraWindow)

>>> dtwPlotDensity(ita)				     # doctest: +SKIP

"""
    # ENDIMPORT
    import matplotlib.pyplot as plt

    try:
        cm = d.costMatrix
    except:
        raise ValueError("dtwPlotDensity requires dtw internals (set keep.internals=True on dtw() call)")

    if normalize:
        norm = d.stepPattern.hint
        row, col = numpy.indices(cm.shape)
        if norm == "NA":
            raise ValueError("Step pattern has no normalization")
        elif norm == "N":
            cm = cm / (row + 1)
        elif norm == "N+M":
            cm = cm / (row + col + 2)
        elif norm == "M":
            cm = cm / (col + 1)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(cm.T, origin="lower", cmap=plt.get_cmap("terrain"))
    co = ax.contour(cm.T, colors="black", linewidths = 1)
    ax.clabel(co)

    ax.plot(d.index1, d.index2, color="blue", linewidth=2)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    return ax

##
## Copyright (c) 2006-2019 of Toni Giorgino
##
## This file is part of the DTW package.
##
## DTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with DTW.  If not, see <http://www.gnu.org/licenses/>.
##

"""Step Pattern handling

See documentation for the StepPattern class.
"""

import numpy


class StepPattern:
    # IMPORT_RDOCSTRING stepPattern
    """Step patterns for DTW

A ``stepPattern`` object lists the transitions allowed while searching
for the minimum-distance path. DTW variants are implemented by passing
one of the objects described in this page to the ``stepPattern``
argument of the [dtw()] call.

**Details**

A step pattern characterizes the matching model and slope constraint
specific of a DTW variant. They also known as local- or
slope-constraints, transition types, production or recursion rules
(GiorginoJSS).

**Pre-defined step patterns**

::

      ## Well-known step patterns
      symmetric1
      symmetric2
      asymmetric

      ## Step patterns classified according to Rabiner-Juang (Rabiner1993)
      rabinerJuangStepPattern(type,slope_weighting="d",smoothed=False)

      ## Slope-constrained step patterns from Sakoe-Chiba (Sakoe1978)
      symmetricP0;  asymmetricP0
      symmetricP05; asymmetricP05
      symmetricP1;  asymmetricP1
      symmetricP2;  asymmetricP2

      ## Step patterns classified according to Rabiner-Myers (Myers1980)
      typeIa;   typeIb;   typeIc;   typeId;
      typeIas;  typeIbs;  typeIcs;  typeIds;  # smoothed
      typeIIa;  typeIIb;  typeIIc;  typeIId;
      typeIIIc; typeIVc;

      ## Miscellaneous
      mori2006;
      rigid;

A variety of classification schemes have been proposed for step
patterns, including Sakoe-Chiba (Sakoe1978); Rabiner-Juang
(Rabiner1993); and Rabiner-Myers (Myers1980). The ``dtw`` package
implements all of the transition types found in those papers, with the
exception of Itakura’s and Velichko-Zagoruyko’s steps, which require
subtly different algorithms (this may be rectified in the future).
Itakura recursion is almost, but not quite, equivalent to ``typeIIIc``.

For convenience, we shall review pre-defined step patterns grouped by
classification. Note that the same pattern may be listed under different
names. Refer to paper (GiorginoJSS) for full details.

**1. Well-known step patterns**

Common DTW implementations are based on one of the following transition
types.

``symmetric2`` is the normalizable, symmetric, with no local slope
constraints. Since one diagonal step costs as much as the two equivalent
steps along the sides, it can be normalized dividing by ``N+M``
(query+reference lengths). It is widely used and the default.

``asymmetric`` is asymmetric, slope constrained between 0 and 2. Matches
each element of the query time series exactly once, so the warping path
``index2~index1`` is guaranteed to be single-valued. Normalized by ``N``
(length of query).

``symmetric1`` (or White-Neely) is quasi-symmetric, no local constraint,
non-normalizable. It is biased in favor of oblique steps.

**2. The Rabiner-Juang set**

A comprehensive table of step patterns is proposed in Rabiner-Juang’s
book (Rabiner1993), tab. 4.5. All of them can be constructed through the
``rabinerJuangStepPattern(type,slope_weighting,smoothed)`` function.

The classification foresees seven families, labelled with Roman numerals
I-VII; here, they are selected through the integer argument ``type``.
Each family has four slope weighting sub-types, named in sec. 4.7.2.5 as
“Type (a)” to “Type (d)”; they are selected passing a character argument
``slope_weighting``, as in the table below. Furthermore, each subtype
can be either plain or smoothed (figure 4.44); smoothing is enabled
setting the logical argument ``smoothed``. (Not all combinations of
arguments make sense.)

::

     Subtype | Rule       | Norm | Unbiased 
     --------|------------|------|---------
        a    | min step   |  --  |   NO 
        b    | max step   |  --  |   NO 
        c    | Di step    |   N  |  YES 
        d    | Di+Dj step | N+M  |  YES 

**3. The Sakoe-Chiba set**

Sakoe-Chiba (Sakoe1978) discuss a family of slope-constrained patterns;
they are implemented as shown in page 47, table I. Here, they are called
``symmetricP<x>`` and ``asymmetricP<x>``, where ``<x>`` corresponds to
Sakoe’s integer slope parameter *P*. Values available are accordingly:
``0`` (no constraint), ``1``, ``05`` (one half) and ``2``. See
(Sakoe1978) for details.

**4. The Rabiner-Myers set**

The ``type<XX><y>`` step patterns follow the older Rabiner-Myers’
classification proposed in (Myers1980) and (MRR1980). Note that this is
a subset of the Rabiner-Juang set (Rabiner1993), and the latter should
be preferred in order to avoid confusion. ``<XX>`` is a Roman numeral
specifying the shape of the transitions; ``<y>`` is a letter in the
range ``a-d`` specifying the weighting used per step, as above;
``typeIIx`` patterns also have a version ending in ``s``, meaning the
smoothing is used (which does not permit skipping points). The
``typeId, typeIId`` and ``typeIIds`` are unbiased and symmetric.

**5. Others**

The ``rigid`` pattern enforces a fixed unitary slope. It only makes
sense in combination with ``open_begin=True``, ``open_end=True`` to find
gapless subsequences. It may be seen as the ``P->inf`` limiting case in
Sakoe’s classification.

``mori2006`` is Mori’s asymmetric step-constrained pattern (Mori2006).
It is normalized by the matched reference length.

[mvmStepPattern()] implements Latecki’s Minimum Variance Matching
algorithm, and it is described in its own page.

**Methods**

``print_stepPattern`` prints an user-readable description of the
recurrence equation defined by the given pattern.

``plot_stepPattern`` graphically displays the step patterns productions
which can lead to element (0,0). Weights are shown along the step
leading to the corresponding element.

``t_stepPattern`` transposes the productions and normalization hint so
that roles of query and reference become reversed.

Parameters
----------
x : 
    a step pattern object
type : 
    path specification, integer 1..7 (see (Rabiner1993), table 4.5)
slope_weighting : 
    slope weighting rule: character `"a"` to `"d"` (see (Rabiner1993), sec. 4.7.2.5)
smoothed : 
    logical, whether to use smoothing (see (Rabiner1993), fig. 4.44)
... : 
    additional arguments to [print()].

Notes
-----

Constructing ``stepPattern`` objects is tricky and thus undocumented.
For a commented example please see source code for ``symmetricP1``.

References
----------

-  (GiorginoJSS) Toni Giorgino. *Computing and Visualizing Dynamic Time
   Warping Alignments in R: The dtw Package.* Journal of Statistical
   Software, 31(7), 1-24.
   `doi:10.18637/jss_v031.i07 <https://doi.org/10.18637/jss_v031.i07>`__
-  (Itakura1975) Itakura, F., *Minimum prediction residual principle
   applied to speech recognition,* Acoustics, Speech, and Signal
   Processing, IEEE Transactions on , vol.23, no.1, pp. 67-72, Feb 1975.
   `doi:10.1109/TASSP.1975.1162641 <https://doi.org/10.1109/TASSP.1975.1162641>`__
-  (MRR1980) Myers, C.; Rabiner, L. & Rosenberg, A. *Performance
   tradeoffs in dynamic time warping algorithms for isolated word
   recognition*, IEEE Trans. Acoust., Speech, Signal Process., 1980, 28,
   623-635.
   `doi:10.1109/TASSP.1980.1163491 <https://doi.org/10.1109/TASSP.1980.1163491>`__
-  (Mori2006) Mori, A.; Uchida, S.; Kurazume, R.; Taniguchi, R.;
   Hasegawa, T. & Sakoe, H. Early Recognition and Prediction of Gestures
   Proc. 18th International Conference on Pattern Recognition ICPR 2006,
   2006, 3, 560-563.
   `doi:10.1109/ICPR.2006.467 <https://doi.org/10.1109/ICPR.2006.467>`__
-  (Myers1980) Myers, Cory S. *A Comparative Study Of Several Dynamic
   Time Warping Algorithms For Speech Recognition*, MS and BS thesis,
   Dept. of Electrical Engineering and Computer Science, Massachusetts
   Institute of Technology, archived Jun 20 1980,
   https://hdl_handle_net/1721.1/27909
-  (Rabiner1993) Rabiner, L. R., & Juang, B.-H. (1993). *Fundamentals of
   speech recognition.* Englewood Cliffs, NJ: Prentice Hall.
-  (Sakoe1978) Sakoe, H.; Chiba, S., *Dynamic programming algorithm
   optimization for spoken word recognition,* Acoustics, Speech, and
   Signal Processing, IEEE Transactions on , vol.26, no.1, pp. 43-49,
   Feb 1978
   `doi:10.1109/TASSP.1978.1163055 <https://doi.org/10.1109/TASSP.1978.1163055>`__

Examples
--------
>>> from dtw import *
>>> import numpy as np

The usual (normalizable) symmetric step pattern
Step pattern recursion, defined as:
 g[i,j] = min(
   g[i,j-1] + d[i,j] ,
   g[i-1,j-1] + 2 * d[i,j] ,
   g[i-1,j] + d[i,j] ,
)

>>> print(symmetric2)		 #doctest: +NORMALIZE_WHITESPACE
Step pattern recursion:
 g[i,j] = min(
     g[i-1,j-1] + 2 * d[i  ,j  ] ,
     g[i  ,j-1] +     d[i  ,j  ] ,
     g[i-1,j  ] +     d[i  ,j  ] ,
 )
<BLANKLINE>
Normalization hint: N+M
<BLANKLINE>

The well-known plotting style for step patterns

>>> import matplotlib.pyplot as plt;		# doctest: +SKIP
... symmetricP2.plot().set_title("Sakoe's Symmetric P=2 recursion")

Same example seen in ?dtw , now with asymmetric step pattern

>>> (query, reference) = dtw_test_data.sin_cos()

Do the computation

>>> asy = dtw(query, reference, keep_internals=True,
... 	  	     step_pattern=asymmetric);

>>> dtwPlot(asy,type="density"			# doctest: +SKIP
...         ).set_title("Sine and cosine, asymmetric step")

Hand-checkable example given in [Myers1980] p 61 - see JSS paper

>>> tm = numpy.reshape( [1, 3, 4, 4, 5, 2, 2, 3, 3, 4, 3, 1, 1, 1, 3, 4, 2,
...                      3, 3, 2, 5, 3, 4, 4, 1], (5,5), "F" )

"""
    # ENDIMPORT

    def __init__(self, mx, hint="NA"):
        self.mx = numpy.array(mx, dtype=numpy.double)
        self.hint = hint

    def get_n_rows(self):
        """Total number of steps in the recursion."""
        return self.mx.shape[0]

    def get_n_patterns(self):
        """Number of rules in the recursion."""
        return int(numpy.max(self.mx[:, 0]))

    def T(self):
        """Transpose a step pattern."""
        tsp = self
        tsp.mx = tsp.mx[:, [0, 2, 1, 3]]
        if tsp.hint == "N":
            tsp.hint = "M"
        elif tsp.hint == "M":
            tsp.hint = "N"
        return tsp

    def __str__(self):
        np = self.get_n_patterns()
        head = " g[i,j] = min(\n"

        body = ""
        for p in range(1, np + 1):
            steps = self._extractpattern(p)
            ns = steps.shape[0]
            steps = numpy.flip(steps, 0)

            for s in range(ns):
                di, dj, cc = steps[s, :]
                dis = "" if di == 0 else f"{-int(di)}"
                djs = "" if dj == 0 else f"{-int(dj)}"
                dijs = f"i{dis:2},j{djs:2}"

                if cc == -1:
                    gs = f"    g[{dijs}]"
                    body = body + " " + gs
                else:
                    ccs = "    " if cc == 1 else f"{cc:2.2g} *"
                    ds = f"+{ccs} d[{dijs}]"
                    body = body + " " + ds
            body = body + " ,\n"

        tail = " ) \n\n"
        ntxt = f"Normalization hint: {self.hint}\n"

        return "Step pattern recursion:\n" + head + body + tail + ntxt

    def plot(self):
        """Provide a visual description of a StepPattern object"""
        import matplotlib.pyplot as plt
        x = self.mx
        pats = numpy.arange(1, 1 + self.get_n_patterns())

        alpha = .5
        fudge = [0, 0]

        fig, ax = plt.subplots(figsize=(6, 6))
        for i in pats:
            ss = x[:, 0] == i
            ax.plot(-x[ss, 1], -x[ss, 2], lw=2, color="tab:blue")
            ax.plot(-x[ss, 1], -x[ss, 2], 'o', color="black", marker="o", fillstyle="none")

            if numpy.sum(ss) == 1: continue

            xss = x[ss, :]
            xh = alpha * xss[:-1, 1] + (1 - alpha) * xss[1:, 1]
            yh = alpha * xss[:-1, 2] + (1 - alpha) * xss[1:, 2]

            for xx, yy, tt in zip(xh, yh, xss[1:, 3]):
                ax.annotate("{:.2g}".format(tt), (-xx - fudge[0],
                                                  -yy - fudge[1]))

        endpts = x[:, 3] == -1
        ax.plot(-x[endpts, 1], -x[endpts, 2], 'o', color="black")

        ax.set_xlabel("Query index")
        ax.set_ylabel("Reference index")
        ax.set_xticks(numpy.unique(-x[:, 1]))
        ax.set_yticks(numpy.unique(-x[:, 2]))
        return ax

    def _extractpattern(self, sn):
        sp = self.mx
        sbs = sp[:, 0] == sn
        spl = sp[sbs, 1:]
        return numpy.flip(spl, 0)

    def _mkDirDeltas(self):
        out = numpy.array(self.mx, dtype=numpy.int32)
        out = out[out[:, 3] == -1, :]
        out = out[:, [1, 2]]
        return out

    def _get_p(self):
        # Dimensions are reversed wrt R
        s = self.mx[:, [0, 2, 1, 3]]
        return s.T.reshape(-1)



# Alternate constructor for ease of R import
def _c(*v):
    va = numpy.array([*v])
    if len(va) % 4 != 0:
        _error("Internal error in _c constructor")
    va = va.reshape((-1, 4))
    return (va)


# Kludge because lambda: raise doesn't work
def _error(s):
    raise ValueError(s)


##################################################
##################################################

# Reimplementation of the building process

class _P:
    def __init__(self, pid, subtype, smoothing):
        self.subtype = subtype
        self.smoothing = smoothing
        self.pid = pid
        self.i = [0]
        self.j = [0]

    def step(self, di, dj):  # equivalent to .Pstep
        self.i.append(di)
        self.j.append(dj)
        return self

    def get(self):  # eqv to .Pend
        ia = numpy.array(self.i, dtype=numpy.double)
        ja = numpy.array(self.j, dtype=numpy.double)
        si = numpy.cumsum(ia)
        sj = numpy.cumsum(ja)
        ni = numpy.max(si) - si  # ?
        nj = numpy.max(sj) - sj
        if self.subtype == "a":
            w = numpy.minimum(ia, ja)
        elif self.subtype == "b":
            w = numpy.maximum(ia, ja)
        elif self.subtype == "c":
            w = ia
        elif self.subtype == "d":
            w = ia + ja
        else:
            _error("Unsupported subtype")

        if self.smoothing:
            # if self.pid==3:                import ipdb; ipdb.set_trace()
            w[1:] = numpy.mean(w[1:])

        w[0] = -1.0

        nr = len(w)
        mx = numpy.zeros((nr, 4))
        mx[:, 0] = self.pid
        mx[:, 1] = ni
        mx[:, 2] = nj
        mx[:, 3] = w
        return mx


def rabinerJuangStepPattern(ptype, slope_weighting="d", smoothed=False):
    """Construct a pattern classified according to the Rabiner-Juang scheme (Rabiner1993)

See documentation for the StepPattern class.
"""

    f = {
        1: _RJtypeI,
        2: _RJtypeII,
        3: _RJtypeIII,
        4: _RJtypeIV,
        5: _RJtypeV,
        6: _RJtypeVI,
        7: _RJtypeVII
    }.get(ptype, lambda: _error("Invalid type"))

    r = f(slope_weighting, smoothed)
    norm = "NA"
    if slope_weighting == "c":
        norm = "N"
    elif slope_weighting == "d":
        norm = "N+M"

    return StepPattern(r, norm)


def _RJtypeI(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(0, 1).get()])


def _RJtypeII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 1).step(0, 1).get()])


def _RJtypeIII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(2, 1).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 2).get()])


def _RJtypeIV(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 2).step(1, 0).get(),
        _P(3, s, m).step(1, 1).get(),
        _P(4, s, m).step(1, 2).get(),
    ])


def _RJtypeV(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).step(1, 0).get(),
        _P(2, s, m).step(1, 1).step(1, 0).get(),
        _P(3, s, m).step(1, 1).get(),
        _P(4, s, m).step(1, 1).step(0, 1).get(),
        _P(5, s, m).step(1, 1).step(0, 1).step(0, 1).get(),
    ])


def _RJtypeVI(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 1).step(1, 0).get(),
        _P(2, s, m).step(1, 1).get(),
        _P(3, s, m).step(1, 1).step(1, 1).step(0, 1).get()
    ])


def _RJtypeVII(s, m):
    return numpy.vstack([
        _P(1, s, m).step(1, 1).step(1, 0).step(1, 0).get(),
        _P(2, s, m).step(1, 2).step(1, 0).step(1, 0).get(),
        _P(3, s, m).step(1, 3).step(1, 0).step(1, 0).get(),
        _P(4, s, m).step(1, 1).step(1, 0).get(),
        _P(5, s, m).step(1, 2).step(1, 0).get(),
        _P(6, s, m).step(1, 3).step(1, 0).get(),
        _P(7, s, m).step(1, 1).get(),
        _P(8, s, m).step(1, 2).get(),
        _P(9, s, m).step(1, 3).get(),
    ])


##########################################################################################
##########################################################################################

## Everything here is semi auto-generated from the R source. Don't
## edit!


##################################################
##################################################


##
## Various step patterns, defined as internal variables
##
## First column: enumerates step patterns.
## Second   	 step in query index
## Third	 step in reference index
## Fourth	 weight if positive, or -1 if starting point
##
## For \cite{} see dtw.bib in the package
##


## Widely-known variants

## White-Neely symmetric (default)
## aka Quasi-symmetric \cite{White1976}
## normalization: no (N+M?)
symmetric1 = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 0, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
));

## Normal symmetric
## normalization: N+M
symmetric2 = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 2,
    2, 0, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N+M");

## classic asymmetric pattern: max slope 2, min slope 0
## normalization: N
asymmetric = StepPattern(_c(
    1, 1, 0, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 0, 1
), "N");

# % \item{\code{symmetricVelichkoZagoruyko}}{symmetric, reproduced from %
# [Sakoe1978]. Use distance matrix \code{1-d}}
# 

## normalization: max[N,M]
## note: local distance matrix is 1-d
## \cite{Velichko}
_symmetricVelichkoZagoruyko = StepPattern(_c(
    1, 0, 1, -1,
    2, 1, 1, -1,
    2, 0, 0, -1.001,
    3, 1, 0, -1));

# % \item{\code{asymmetricItakura}}{asymmetric, slope contrained 0.5 -- 2
# from reference [Itakura1975]. This is the recursive definition % that
# generates the Itakura parallelogram; }
# 

## Itakura slope-limited asymmetric \cite{Itakura1975}
## Max slope: 2; min slope: 1/2
## normalization: N
_asymmetricItakura = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1,
    4, 2, 2, -1,
    4, 1, 0, 1,
    4, 0, 0, 1
));

#############################
## Slope-limited versions
##
## Taken from Table I, page 47 of "Dynamic programming algorithm
## optimization for spoken word recognition," Acoustics, Speech, and
## Signal Processing, vol.26, no.1, pp. 43-49, Feb 1978 URL:
## http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1163055
##
## Mostly unchecked


## Row P=0
symmetricP0 = symmetric2;

## normalization: N ?
asymmetricP0 = StepPattern(_c(
    1, 0, 1, -1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N");

## alternative implementation
_asymmetricP0b = StepPattern(_c(
    1, 0, 1, -1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N");

## Row P=1/2
symmetricP05 = StepPattern(_c(
    1, 1, 3, -1,
    1, 0, 2, 2,
    1, 0, 1, 1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 1, 2,
    2, 0, 0, 1,
    3, 1, 1, -1,
    3, 0, 0, 2,
    4, 2, 1, -1,
    4, 1, 0, 2,
    4, 0, 0, 1,
    5, 3, 1, -1,
    5, 2, 0, 2,
    5, 1, 0, 1,
    5, 0, 0, 1
), "N+M");

asymmetricP05 = StepPattern(_c(
    1, 1, 3, -1,
    1, 0, 2, 1 / 3,
    1, 0, 1, 1 / 3,
    1, 0, 0, 1 / 3,
    2, 1, 2, -1,
    2, 0, 1, .5,
    2, 0, 0, .5,
    3, 1, 1, -1,
    3, 0, 0, 1,
    4, 2, 1, -1,
    4, 1, 0, 1,
    4, 0, 0, 1,
    5, 3, 1, -1,
    5, 2, 0, 1,
    5, 1, 0, 1,
    5, 0, 0, 1
), "N");

## Row P=1
## Implementation of Sakoe's P=1, Symmetric algorithm

symmetricP1 = StepPattern(_c(
    1, 1, 2, -1,  # First branch: g(i-1,j-2)+
    1, 0, 1, 2,  # + 2d(i  ,j-1)
    1, 0, 0, 1,  # +  d(i  ,j)
    2, 1, 1, -1,  # Second branch: g(i-1,j-1)+
    2, 0, 0, 2,  # +2d(i,  j)
    3, 2, 1, -1,  # Third branch: g(i-2,j-1)+
    3, 1, 0, 2,  # + 2d(i-1,j)
    3, 0, 0, 1  # +  d(  i,j)
), "N+M");

asymmetricP1 = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 1, .5,
    1, 0, 0, .5,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1
), "N");

## Row P=2
symmetricP2 = StepPattern(_c(
    1, 2, 3, -1,
    1, 1, 2, 2,
    1, 0, 1, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 3, 2, -1,
    3, 2, 1, 2,
    3, 1, 0, 2,
    3, 0, 0, 1
), "N+M");

asymmetricP2 = StepPattern(_c(
    1, 2, 3, -1,
    1, 1, 2, 2 / 3,
    1, 0, 1, 2 / 3,
    1, 0, 0, 2 / 3,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 3, 2, -1,
    3, 2, 1, 1,
    3, 1, 0, 1,
    3, 0, 0, 1
), "N");

################################
## Taken from Table III, page 49.
## Four varieties of DP-algorithm compared

## 1st row:  asymmetric

## 2nd row:  symmetricVelichkoZagoruyko

## 3rd row:  symmetric1

## 4th row:  asymmetricItakura


#############################
## Classified according to Rabiner
##
## Taken from chapter 2, Myers' thesis [4]. Letter is
## the weighting function:
##
##      rule       norm   unbiased
##   a  min step   ~N     NO
##   b  max step   ~N     NO
##   c  x step     N      YES
##   d  x+y step   N+M    YES
##
## Mostly unchecked

# R-Myers     R-Juang
# type I      type II   
# type II     type III
# type III    type IV
# type IV     type VII


typeIa = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 0
));

typeIb = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 1
));

typeIc = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 0
), "N");

typeId = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 1, 2, -1,
    3, 0, 1, 2,
    3, 0, 0, 1
), "N+M");

## ----------
## smoothed variants of above

typeIas = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, .5,
    1, 0, 0, .5,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, .5,
    3, 0, 0, .5
));

typeIbs = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 1
));

typeIcs = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, .5,
    3, 0, 0, .5
), "N");

typeIds = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1.5,
    1, 0, 0, 1.5,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 1, 2, -1,
    3, 0, 1, 1.5,
    3, 0, 0, 1.5
), "N+M");

## ----------

typeIIa = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 0, 0, 1
));

typeIIb = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 2,
    3, 2, 1, -1,
    3, 0, 0, 2
));

typeIIc = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 0, 0, 2
), "N");

typeIId = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 2,
    2, 1, 2, -1,
    2, 0, 0, 3,
    3, 2, 1, -1,
    3, 0, 0, 3
), "N+M");

## ----------

## Rabiner [3] discusses why this is not equivalent to Itakura's

typeIIIc = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1,
    4, 2, 2, -1,
    4, 1, 0, 1,
    4, 0, 0, 1
), "N");

## ----------

## numbers follow as production rules in fig 2.16

typeIVc = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 1, 3, -1,
    3, 0, 0, 1,
    4, 2, 1, -1,
    4, 1, 0, 1,
    4, 0, 0, 1,
    5, 2, 2, -1,
    5, 1, 0, 1,
    5, 0, 0, 1,
    6, 2, 3, -1,
    6, 1, 0, 1,
    6, 0, 0, 1,
    7, 3, 1, -1,
    7, 2, 0, 1,
    7, 1, 0, 1,
    7, 0, 0, 1,
    8, 3, 2, -1,
    8, 2, 0, 1,
    8, 1, 0, 1,
    8, 0, 0, 1,
    9, 3, 3, -1,
    9, 2, 0, 1,
    9, 1, 0, 1,
    9, 0, 0, 1
), "N");

#############################
## 
## Mori's asymmetric step-constrained pattern. Normalized in the
## reference length.
##
## Mori, A.; Uchida, S.; Kurazume, R.; Taniguchi, R.; Hasegawa, T. &
## Sakoe, H. Early Recognition and Prediction of Gestures Proc. 18th
## International Conference on Pattern Recognition ICPR 2006, 2006, 3,
## 560-563
##

mori2006 = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 3,
    3, 1, 2, -1,
    3, 0, 1, 3,
    3, 0, 0, 3
), "M");

## Completely unflexible: fixed slope 1. Only makes sense with
## open.begin and open.end
rigid = StepPattern(_c(1, 1, 1, -1,
                       1, 0, 0, 1), "N")

# --------------------

class DTW:
    """The results of an alignment operation.

Objects of class DTW contain alignments computed by the [dtw()]
function. 

**Attributes:**

- ``distance`` the minimum global distance computed, *not* normalized.
- ``normalizedDistance`` distance computed, *normalized* for path
  length, if normalization is known for chosen step pattern.
- ``N,M`` query and reference length
- ``call`` the function call that created the object
- ``index1`` matched elements: indices in ``x``
- ``index2`` corresponding mapped indices in ``y``
- ``stepPattern`` the ``stepPattern`` object used for the computation
- ``jmin`` last element of reference matched, if ``open_end=True``
- ``directionMatrix`` if ``keep_internals=True``, the directions of
  steps that would be taken at each alignment pair (integers indexing
  production rules in the chosen step pattern)
- ``stepsTaken`` the list of steps taken from the beginning to the end
  of the alignment (integers indexing chosen step pattern)
- ``index1s, index2s`` same as ``index1/2``, excluding intermediate
  steps for multi-step patterns like [asymmetricP05()]
- ``costMatrix`` if ``keep_internals=True``, the cumulative cost matrix
- ``query, reference`` if ``keep_internals=True`` and passed as the
  ``x`` and ``y`` arguments, the query and reference timeseries.

"""
    
    def __init__(self, obj):
        self.__dict__.update(obj)  # Convert dict to object

    def __repr__(self):
        s = "DTW alignment object of size (query x reference): {:d} x {:d}".format(self.N, self.M)
        return (s)

    def plot(self, type="alignment", **kwargs):
        # IMPORT_RDOCSTRING plot.dtw
        """Plotting of dynamic time warp results

Methods for plotting dynamic time warp alignment objects returned by
[dtw()].

**Details**

``dtwPlot`` displays alignment contained in ``dtw`` objects.

Various plotting styles are available, passing strings to the ``type``
argument (may be abbreviated):

-  ``alignment`` plots the warping curve in ``d``;
-  ``twoway`` plots a point-by-point comparison, with matching lines;
   see [dtwPlotTwoWay()];
-  ``threeway`` vis-a-vis inspection of the timeseries and their warping
   curve; see [dtwPlotThreeWay()];
-  ``density`` displays the cumulative cost landscape with the warping
   path overimposed; see [dtwPlotDensity()]

Additional parameters are passed to the plotting functions: use with
care.

Parameters
----------
x,d : 
    `dtw` object, usually result of call to [dtw()]
xlab : 
    label for the query axis
ylab : 
    label for the reference axis
type : 
    general style for the plot, see below
plot_type : 
    type of line to be drawn, used as the `type` argument in the underlying `plot` call
... : 
    additional arguments, passed to plotting functions

"""
        # ENDIMPORT
        return dtwPlot(self, type, **kwargs)


# --------------------


def dtw(x, y=None,
        dist_method="euclidean",
        step_pattern="symmetric2",
        window_type=None,
        window_args={},
        keep_internals=False,
        distance_only=False,
        open_end=False,
        open_begin=False):
    """Compute Dynamic Time Warp and find optimal alignment between two time
series.

**Details**

The function performs Dynamic Time Warp (DTW) and computes the optimal
alignment between two time series ``x`` and ``y``, given as numeric
vectors. The “optimal” alignment minimizes the sum of distances between
aligned elements. Lengths of ``x`` and ``y`` may differ.

The local distance between elements of ``x`` (query) and ``y``
(reference) can be computed in one of the following ways:

1. if ``dist_method`` is a string, ``x`` and ``y`` are passed to the
   `scipy.spatial.distance.cdist` function with the method given;
2. multivariate time series and arbitrary distance metrics can be
   handled by supplying a local-distance matrix. Element ``[i,j]`` of
   the local-distance matrix is understood as the distance between
   element ``x[i]`` and ``y[j]``. The distance matrix has therefore
   ``n=length(x)`` rows and ``m=length(y)`` columns (see note below).

Several common variants of the DTW recursion are supported via the
``step_pattern`` argument, which defaults to ``symmetric2``. Step
patterns are commonly used to *locally* constrain the slope of the
alignment function. See [stepPattern()] for details.

Windowing enforces a *global* constraint on the envelope of the warping
path. It is selected by passing a string or function to the
``window_type`` argument. Commonly used windows are (abbreviations
allowed):

-  ``"none"`` No windowing (default)
-  ``"sakoechiba"`` A band around main diagonal
-  ``"slantedband"`` A band around slanted diagonal
-  ``"itakura"`` So-called Itakura parallelogram

``window_type`` can also be an user-defined windowing function. See
[dtwWindowingFunctions()] for all available windowing functions, details
on user-defined windowing, and a discussion of the (mis)naming of the
“Itakura” parallelogram as a global constraint. Some windowing functions
may require parameters, such as the ``window_size`` argument.

Open-ended alignment, i_e. semi-unconstrained alignment, can be selected
via the ``open_end`` switch. Open-end DTW computes the alignment which
best matches all of the query with a *leading part* of the reference.
This is proposed e_g. by Mori (2006), Sakoe (1979) and others.
Similarly, open-begin is enabled via ``open_begin``; it makes sense when
``open_end`` is also enabled (subsequence finding). Subsequence
alignments are similar e_g. to UE2-1 algorithm by Rabiner (1978) and
others. Please find a review in Tormene et al. (2009).

If the warping function is not required, computation can be sped up
enabling the ``distance_only=True`` switch, which skips the backtracking
step. The output object will then lack the ``index{1,2,1s,2s}`` and
``stepsTaken`` fields.


Parameters
----------
x : 
    query vector *or* local cost matrix
y : 
    reference vector, unused if `x` given as cost matrix
dist_method : 
    pointwise (local) distance function to use. 
step_pattern : 
    a stepPattern object describing the local warping steps
    allowed with their cost (see [stepPattern()])
window_type : 
    windowing function. Character: "none", "itakura",
    "sakoechiba", "slantedband", or a function (see details).
open_begin,open_end : 
    perform open-ended alignments
keep_internals : 
    preserve the cumulative cost matrix, inputs, and other
    internal structures
distance_only : 
    only compute distance (no backtrack, faster)
window_args : 
    additional arguments, passed to the windowing function

Returns
-------

An object of class ``DTW``. See docs for the corresponding properties. 


Notes
-----

Cost matrices (both input and output) have query elements arranged
row-wise (first index), and reference elements column-wise (second
index). They print according to the usual convention, with indexes
increasing down- and rightwards. Many DTW papers and tutorials show
matrices according to plot-like conventions, i_e. reference index
growing upwards. This may be confusing.

A fast compiled version of the function is normally used. Should it be
unavailable, the interpreted equivalent will be used as a fall-back with
a warning.

References
----------

1. Toni Giorgino. *Computing and Visualizing Dynamic Time Warping
   Alignments in R: The dtw Package.* Journal of Statistical Software,
   31(7), 1-24. http://www.jstatsoft.org/v31/i07/
2. Tormene, P.; Giorgino, T.; Quaglini, S. & Stefanelli, M. *Matching
   incomplete time series with dynamic time warping: an algorithm and an
   application to post-stroke rehabilitation.* Artif Intell Med, 2009,
   45, 11-34. http://dx.doi.org/10.1016/j.artmed.2008.11.007
3. Sakoe, H.; Chiba, S., *Dynamic programming algorithm optimization for
   spoken word recognition,* Acoustics, Speech, and Signal Processing,
   IEEE Transactions on , vol.26, no.1, pp. 43-49, Feb 1978.
   http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1163055
4. Mori, A.; Uchida, S.; Kurazume, R.; Taniguchi, R.; Hasegawa, T. &
   Sakoe, H. *Early Recognition and Prediction of Gestures* Proc. 18th
   International Conference on Pattern Recognition ICPR 2006, 2006, 3,
   560-563
5. Sakoe, H. *Two-level DP-matching–A dynamic programming-based pattern
   matching algorithm for connected word recognition* Acoustics, Speech,
   and Signal Processing, IEEE Transactions on, 1979, 27, 588-595
6. Rabiner L, Rosenberg A, Levinson S (1978). *Considerations in dynamic
   time warping algorithms for discrete word recognition.* IEEE Trans.
   Acoust., Speech, Signal Process., 26(6), 575-582. ISSN 0096-3518.
7. Muller M. *Dynamic Time Warping* in *Information Retrieval for Music
   and Motion*. Springer Berlin Heidelberg; 2007. p. 69-84.
   http://link.springer.com/chapter/10.1007/978-3-540-74048-3_4

Examples
--------



>>> import numpy as np
>>> from dtw import *

A noisy sine wave as query

>>> idx = np.linspace(0,6.28,num=100)
>>> query = np.sin(idx) + np.random.uniform(size=100)/10.0

A cosine is for reference; sin and cos are offset by 25 samples

>>> reference = np.cos(idx)

Find the best match

>>> alignment = dtw(query,reference)

Display the mapping, AKA warping function - may be multiple-valued
Equivalent to: plot(alignment,type="alignment")

>>> import matplotlib.pyplot as plt;
... plt.plot(alignment.index1, alignment.index2)	# doctest: +SKIP


Partial alignments are allowed.

>>> alignmentOBE = dtw(query[44:88], reference,
...                      keep_internals=True,
...                      step_pattern=asymmetric,
...                      open_end=True,open_begin=True)

>>> alignmentOBE.plot(type="twoway",offset=1)		# doctest: +SKIP


Subsetting allows warping and unwarping of
timeseries according to the warping curve. 
See first example below.

Most useful: plot the warped query along with reference 

>>> plt.plot(reference);
... plt.plot(alignment.index2,query[alignment.index1])	# doctest: +SKIP

Plot the (unwarped) query and the inverse-warped reference

>>> plt.plot(query)					# doctest: +SKIP
... plt.plot(alignment.index1,reference[alignment.index2]) 








A hand-checkable example

>>> ldist = np.ones((6,6))		      # Matrix of ones
>>> ldist[1,:] = 0; ldist[:,4] = 0;           # Mark a clear path of zeroes
>>> ldist[1,4] = .01;		              # Forcely cut the corner

>>> ds = dtw(ldist);			      # DTW with user-supplied local

>>> da = dtw(ldist,step_pattern=asymmetric)   # Also compute the asymmetric 

Symmetric: alignment follows the low-distance marked path

>>> plt.plot(ds.index1,ds.index2)	      # doctest: +SKIP

Asymmetric: visiting 1 is required twice

>>> plt.plot(da.index1,da.index2,'ro')	      # doctest: +SKIP	

>>> ds.distance
2.0
>>> da.distance
2.0

"""


    if y is None:
        x = numpy.array(x)
        if len(x.shape) != 2:
            _error("A 2D local distance matrix was expected")
        lm = numpy.array(x)
    else:
        x2, y2 = numpy.atleast_2d(x), numpy.atleast_2d(y)
        if x2.shape[0] == 1:
            x2 = x2.T
        if y2.shape[0] == 1:
            y2 = y2.T
        lm = scipy.spatial.distance.cdist(x2, y2, metric=dist_method)

    wfun = _canonicalizeWindowFunction(window_type)

    step_pattern = _canonicalizeStepPattern(step_pattern)
    norm = step_pattern.hint

    n, m = lm.shape

    if open_begin:
        if norm != "N":
            _error(
                "Open-begin requires step patterns with 'N' normalization (e.g. asymmetric, or R-J types (c)). See Tormene et al.")
        lm = numpy.vstack([numpy.zeros((1, lm.shape[1])), lm])  # prepend null row
        np = n + 1
        precm = numpy.full_like(lm, numpy.nan, dtype=numpy.double)
        precm[0, :] = 0
    else:
        precm = None
        np = n

    gcm = _globalCostMatrix(lm,
                            step_pattern=step_pattern,
                            window_function=wfun,
                            seed=precm,
                            win_args=window_args)
    gcm = DTW(gcm)  # turn into an object, use dot to access properties

    gcm.N = n
    gcm.M = m

    gcm.openEnd = open_end
    gcm.openBegin = open_begin
    gcm.windowFunction = wfun
    gcm.windowArgs = window_args  # py

    # misnamed
    lastcol = gcm.costMatrix[-1,]

    if norm == "NA":
        pass
    elif norm == "N+M":
        lastcol = lastcol / (n + numpy.arange(m) + 1)
    elif norm == "N":
        lastcol = lastcol / n
    elif norm == "M":
        lastcol = lastcol / (1 + numpy.arange(m))

    gcm.jmin = m - 1

    if open_end:
        if norm == "NA":
            _error("Open-end alignments require normalizable step patterns")
        gcm.jmin = numpy.nanargmin(lastcol)

    gcm.distance = gcm.costMatrix[-1, gcm.jmin]

    if numpy.isnan(gcm.distance):
        _error("No warping path found compatible with the local constraints")

    if step_pattern.hint != "NA":
        gcm.normalizedDistance = lastcol[gcm.jmin]
    else:
        gcm.normalizedDistance = numpy.nan

    if not distance_only:
        mapping = _backtrack(gcm)
        gcm.__dict__.update(mapping)

    if open_begin:
        gcm.index1 = gcm.index1[1:] - 1
        gcm.index1s = gcm.index1s[1:] - 1
        gcm.index2 = gcm.index2[1:]
        gcm.index2s = gcm.index2s[1:]
        lm = lm[1:, :]
        gcm.costMatrix = gcm.costMatrix[1:, :]
        gcm.directionMatrix = gcm.directionMatrix[1:, :]

    if not keep_internals:
        del gcm.costMatrix
        del gcm.directionMatrix
    else:
        gcm.localCostMatrix = lm
        if y is not None:
            gcm.query = x
            gcm.reference = y

    return gcm


# Return a callable object representing the window
def _canonicalizeWindowFunction(window_type):
    if callable(window_type):
        return window_type

    if window_type is None:
        return noWindow

    return {
        "none": noWindow,
        "sakoechiba": sakoeChibaWindow,
        "itakura": itakuraWindow,
        "slantedband": slantedBandWindow
    }.get(window_type, lambda: _error("Window function undefined"))


def _canonicalizeStepPattern(s):
    """Return object by string"""
    if hasattr(s,"mx"):
        return s
    else:
        return getattr(sys.modules["dtw.stepPattern"], s)


# Kludge because lambda: raise doesn't work
def _error(s):
    raise ValueError(s)

