"""DESCRIPTION."""
# License: GNU AGPLv3

from functools import reduce
from operator import and_
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numpy.ma import masked_invalid
from numpy.ma.core import MaskedArray
from scipy.sparse import csr, csgraph, issparse, isspmatrix_csr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import PlotterMixin
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import check_graph

# -------------------------------------------------------------------
# ---------------------------- FUNCTIONS ----------------------------
# -------------------------------------------------------------------

@njit
def edge_to_single_value(C, u, v):
    F_row = C[u]-C[v]
    ranks = np.empty_like(F_row)
    ranks[np.argsort(F_row)] = np.arange(1, len(F_row)+1)
    return np.sum((2*ranks-1-len(F_row))*F_row)

@njit
def resistance_to_flow(C, row_idx, col_idx):
    """
    [TODO: RETURN MATRIX INSTEAD OF LIST]
    [TODO: DOCSTRING + TESTING]
    """
    return [edge_to_single_value(C, u, v)
                    for (u,v) in zip(row_idx, col_idx) if u <= v]

def current_flow(A):
    """[TODO: DOCSTRING + TESTING]"""
    L = csgraph.laplacian(A).A
    C = np.zeros(L.shape)
    C[1:,1:] = np.linalg.inv(L[1:,1:])
    A_coo = A.tocoo()
    values = resistance_to_flow(C, A.indptr, A.indices)
    return values


# -------------------------------------------------------------------
# ----------------------------- Classes -----------------------------
# -------------------------------------------------------------------

# @adapt_fit_transform_docs --> not sure what this is doing..
class GraphCurrentFlowCentrality(BaseEstimator, TransformerMixin):
    """Weighted adjacency matrices arising from current-flow centrality measure
    on graphs.

    For each (unweighted) graph in a collection, this transformer calculates
    the (edge-based) current flow centrality measure for each edge in the graph.
    Given two vertices s and t the st-current flow of an edge (i,j) is
    defined to be the expected number of times a random walker is trespassing
    the edge (i,j) when starting at the node s and stopping at node t. The
    (overall) current-flow is defined as the average of all st-current flows,
    for all possible s and t.

    The graphs are represented by their adjacency matrices which can be dense
    arrays, sparse matrices or masked arrays. The following rules apply:

    - In dense arrays of Boolean type, entries which are ``False`` represent
      absent edges.
    - In dense arrays of integer or float type, zero entries represent edges
      of length 0. Absent edges must be indicated by ``numpy.inf``.
    - In sparse matrices, non-stored values represent absent edges. Explicitly
      stored zero or ``False`` edges represent edges of length 0.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.


    Examples [TODO: EXAMPLE]
    --------
    >>> import numpy as np
    >>> from gtda.graphs import TransitionGraph, GraphGeodesicDistance
    >>> X = np.arange(4).reshape(1, -1, 1)
    >>> X_tg = TransitionGraph(func=None).fit_transform(X)
    >>> print(X_tg[0].toarray())
    [[0 1 0 0]
     [0 0 1 0]
     [0 0 0 1]
     [0 0 0 0]]
    >>> X_ggd = GraphGeodesicDistance(directed=False).fit_transform(X_tg)
    >>> print(X_ggd[0])
    [[0. 1. 2. 3.]
     [1. 0. 1. 2.]
     [2. 1. 0. 1.]
     [3. 2. 1. 0.]]

    See also
    --------
    TransitionGraph, KNeighborsGraph, GraphGeodesicDistance

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _current_flow(self, X, i=None):
        if not issparse(X):
            if not isinstance(X, MaskedArray):
                # Convert to a masked array with mask given by positions in
                # which infs or NaNs occur.
                if X.dtype != bool:
                    X = masked_invalid(X)
        elif X.shape[0] != X.shape[1]:
            n_vertices = max(X.shape)
            X = X.copy() if isspmatrix_csr(X) else X.tocsr()
            X.resize(n_vertices, n_vertices)

        return current_flow(X)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Input data: a collection of adjacency matrices of graphs. Each
            adjacency matrix may be a dense or a sparse array.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_graph(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute the lengths of graph shortest paths between any two
        vertices.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Input data: a collection of ``n_samples`` adjacency matrices of
            graphs. Each adjacency matrix may be a dense array, a sparse
            matrix, or a masked array.

        y : None
            Ignored.

        Returns
        -------
        Xt : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Output collection of dense distance matrices. If the distance
            matrices all have the same shape, a single 3D ndarray is returned.

        """
        check_is_fitted(self, '_is_fitted')
        X = check_graph(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._current_flow)(x, i=i) for i, x in enumerate(X))

        x0_shape = Xt[0].shape
        if reduce(and_, (x.shape == x0_shape for x in Xt), True):
            Xt = np.asarray(Xt)

        return Xt
