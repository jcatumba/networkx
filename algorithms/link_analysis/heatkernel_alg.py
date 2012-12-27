"""Heat Kernel analysis method of graph structure. """
#    Copyright (C) 2012 by
#    Jorge Catumba <jorgerev90@gmail.com>
#    All rights reserved.
#    BSD license.
#    NetworkX:http://networkx.lanl.gov/
import networkx as nx
from networkx.exception import NetworkXError
__author__ = """Jorge Catumba (jorgerev90@gmail.com)"""
__all__ = ['heatkernel']

def heatkernel(G, k=1):
    """Return the Heat Kernel indirect influences and indirect
     dependences of the nodes in the graph.

    Heat Kernel computes the indirect influences and dependencias
    via the exponential formula described in the reference.

    Parameters
    -----------
    G : graph
      A NetworkX graph

    k : integer, optional
      Paramenter of the Heat Kernel method, default 1

    Returns
    -------
    ind: dictionary
        Dictionary of Heat Kernel indirect influences and
        dependences

    Examples
    --------
    >>> G=nx.DiGraph(nx.path_graph(4))
    >>> pr=nx.heatkernel(G)

    Notes
    -----
    This function uses the numpy library to make the computations

    See Also
    --------
    micmac, pwp

    References
    ----------
    .. [2] F. Chung, 
    The heat kernel as the pagerank of a graph, Proc. Natl. Acad.
    Sci. 104 (50) 19735-19740.
    .. [1] R. Diaz,
       Indirect Influences, preprint, arXiv:0906.1610.
    """
    
    try:
        import scipy as sp
    except ImportError:
        raise ImportError(\
            "heatkernel() requires SciPy: http://scipy.org/")
            
    try:
        import scipy.linalg as lin
    except ImportError:
        raise ImportError(\
            "heatkernel() requires scipy.linalg: http://scipy.org")
            
    try:
        import scipy.linalg.matfuncs as matf
    except ImportError:
        raise ImportError(\
            "heatkernel() requires scipy.linalg.matfuncs: http://scipy.org")
            
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("heatkernel() not defined for graphs with multiedges.")

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
    
    # get the adjacency matrix of the graph
    A = nx.adjacency_matrix(D)
    # transpose de adjacency matrix
    A = sp.transpose(A)
    # get the size of the matrix
    j = sp.shape(A)[1]
    
    # get the k-th power of A
    I = matf.expm(k * (A - sp.eye(j, j)))
    
    # get the indirect influences vector
    f = sp.sum(I, axis=0)
    f = f/sp.sum(f)
    
    # get the indirect dependences vector
    d = sp.sum(I, axis=1)
    d = d/sp.sum(d)
    
    #create the dictionary with the result
    fdict = {i: number for number, i in zip(f, range(j))}
    ddict = {i: number for number, i in zip(d, range(j))}
    x = {'influences': fdict, 'dependences': ddict}
    return x

# fixture for nose tests
def setup_module(module):
    from nose import SkipTest
    try:
        import scipy
    except:
        raise SkipTest("SciPy not available")
