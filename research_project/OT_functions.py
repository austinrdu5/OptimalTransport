import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, csc_matrix, hstack
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import cvxpy as cp

# Functions for calculating neighbors:

def get_k_neighbors(X, k):
    # Inputs:
    # X - a cluster of points in the form of a 2D numpy array
    # k - number of neighbors desired
    #
    # Outputs:
    # indices - list of indices of nearest neighbors, with a list for each input point

    nbrs = NearestNeighbors(n_neighbors = k + 1, algorithm = "kd_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    return np.delete(indices, 0, axis = 1)

def get_k_partitioned_neighbors(X, k, num_partitions):
    # Inputs:
    # X - a cluster of points in the form of a 2D numpy array
    # k - number of neighbors desired
    # num_partitions - number of partitions into which X will be divided
    #
    # Outputs:
    # neighbors - list of indices of nearest neighbors, with a list for each input point
    
    partitions = np.array_split(X, num_partitions) # list of partitions
    
    # assign sink-partition not equal to source partition for each point in X.
    sinks = np.zeros(0)
    for source in range(num_partitions):
        possible_sinks = [i for i in range(num_partitions) if i != source]
        sinks = np.append(sinks, np.random.choice(a = possible_sinks, size = len(partitions[source])))
    
    neighbors = np.zeros((len(X), k))
    offset = 0
    for sink in range(num_partitions):
        # within the current sink partition, get all indices and points
        indices = (sinks == sink).nonzero()[0]
        points = X[indices]
        
        # fit kNN on current partition
        nbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(partitions[sink])
        # find neighbors (within partition) of the points to be compared
        partition_ind = nbrs.kneighbors(points)[1]
        
        #store indices of neighbors
        for i in range(len(indices)):
            neighbors[indices[i]] = partition_ind[i] + offset
        offset += len(partitions[sink])
        
    return neighbors


# Functions for Optimal Transport Algorithm:

def make_QQ(neighbors):
    n, k = neighbors.shape
    ind = np.tile(np.arange(n), k).reshape(-1,1)
    collapsed_nbrs = neighbors.reshape(-1,1, order = 'F')
    
    edges = np.concatenate((ind, collapsed_nbrs), axis = 1)
    # sort edges (!?) to eliminate identical edges
    unique_edges = np.unique(np.sort(edges, axis = 1), axis = 0)

    n_edges = len(unique_edges)
    vals = np.array([1, -1]*n_edges)
    row_ind = unique_edges.flatten()
    col_ptr = np.arange(n_edges + 1) * 2
    
    return hstack((csc_matrix((vals, row_ind, col_ptr)), csc_matrix((-vals, row_ind, col_ptr))))

def make_Q(neighbors):
    n, k = neighbors.shape
    ind = np.tile(np.arange(n), k).reshape(-1,1)
    collapsed_nbrs = neighbors.reshape(-1,1, order = 'F')
    
    edges = np.concatenate((ind, collapsed_nbrs), axis = 1)
    # sort edges (!?) to eliminate identical edges
    unique_edges = np.unique(np.sort(edges, axis = 1), axis = 0)

    n_edges = len(unique_edges)
    vals = np.array([1, -1]*n_edges)
    row_ind = unique_edges.flatten()
    col_ptr = np.arange(n_edges + 1) * 2
    
    return csc_matrix((vals, row_ind, col_ptr))

def get_edge_distances(G, QQ):
    
    n_edges = QQ.shape[1]
    
    def edge_euc_distance(edge):
        u_pos = G.nodes[edge[0]]['pos']
        v_pos = G.nodes[edge[1]]['pos']
        return np.linalg.norm([u_pos[0]-v_pos[0], u_pos[1]-v_pos[1]])
    
    distances = np.apply_along_axis(edge_euc_distance, 1, QQ.indices.reshape(-1,2)[:(QQ.shape[1])//2,:])
    return np.tile(distances, 2)
    
def get_weights(QQ, pmf, edge_distances, alpha = 1):
    output = linprog(c = edge_distances, A_eq = QQ, b_eq = -pmf)
    
    return np.round(output.x, 8), output.success

def get_reg_weights(X, Q_sparse, pmf, alpha = 1, eps = 1e-5):
    Q = Q_sparse.toarray()

    def euc_dist(tup1, tup2):
        arr1 = np.array(tup1)
        arr2 = np.array(tup2)
        return np.linalg.norm(arr1 - arr2)

    n_edges = Q_sparse.shape[1]
    l = np.empty(n_edges)
    for i in range(len(l)):
        u, v = Q_sparse.indices.reshape(-1, 2)[i]
        l[i] = euc_dist(X[u], X[v])

    # get transport weights with quadratic programming
    P = np.diag(np.ones(n_edges))

    x = cp.Variable(n_edges)
    prob = cp.Problem(cp.Minimize(l.T @ cp.atoms.elementwise.abs.abs(x) + \
                                  alpha/2 * cp.quad_form(x, P)), 
                      [Q @ x == pmf])
    prob.solve(solver=cp.SCS, eps = eps)

    w = x.value
    return w, l


# Functions for creating and visualizing graphs
def random_X(n = 500, d = 2):
    return np.random.uniform(0, 1, d*n).reshape(n, d)

def skewed_X(a, b, n = 500, d = 2):
    return stats.beta.rvs(a, b, size = d*n).reshape(n, d)

def make_uniform_pmf(X, source_center, sink_center, r_source = 0.5, r_sink = 0.5):
    
    assert len(source_center) == X.shape[1], 'dimension of source_center must match that of points in X'
    assert len(sink_center) == X.shape[1], 'dimension of sink_center must match that of points in X'
    
    # create binary arrays that identify whether a node is source or sink
    is_source = np.apply_along_axis(lambda x: 1 if np.linalg.norm(x - source_center) <= r_source else 0, 1, X)
    is_sink = np.apply_along_axis(lambda x: 1 if np.linalg.norm(x - sink_center) <= r_sink else 0, 1, X)
    
    # normalize arrays to sum to 1
    source_pmf = is_source / is_source.sum()
    sink_pmf = is_sink / is_sink.sum()
    
    return source_pmf - sink_pmf
    
def create_graph(X, neighbors):
    # Inputs:
    # X - a cluster of points in the form of a 2D numpy array
    # k - a list of neighbors to be connected to each point in X, given by their indices
    #
    # Output:
    # G - the resulting graph
    #
    # Displays a graph with nodes at the points of X and connected neighbors
    
    G = nx.Graph()
    # insert nodes, with position stored as a tuple in attribute 'pos'
    G.add_nodes_from(enumerate([{'pos': tuple(i)} for i in X]))
    
    # for each node, add an edge for each neighbor 
    for n in range(len(X)):
        G.add_edges_from([(n, i) for i in neighbors[n]])
    
    return G

def draw_graph(G):
    plt.cla()
    nx.drawing.nx_pylab.draw_networkx(G, pos = nx.get_node_attributes(G,'pos'), 
                                      with_labels = False, node_size = 20, linewidths = 0.5)

    return G

def create_digraph(X, neighbors, pmf, use_distances = True, alpha = 1):
    # Inputs:
    # X - a cluster of points in the form of a 2D numpy array
    # neighbors - matrix of neighbors for each node in X, gotten from either get_k_neighbors or get_k_partitioned_neighbors
    # pmf - array of real numbers summing to zero, with positive representing source mass and negative representing sink mass
    # use_distances - boolean on whether to increase transport cost according to length
    # alpha - scale factor for total transport length
    #
    # Output:
    # G - the resulting directed graph, with transport weights assigned to each edge
    
    assert math.isclose(pmf.sum(), 0, abs_tol = 1e-06), 'total source mass must equal total sink mass'
    
    G = nx.DiGraph()
    
    # insert nodes, with position stored as a tuple in attribute 'pos'
    G.add_nodes_from(enumerate([{'pos': tuple(i)} for i in X]))
    
    # assign attribute 'is_source' to each nodes, with a positive value for sources and negative value for sinks
    G.add_nodes_from(enumerate([{'is_source': i} for i in pmf]))
    
    # create |V| x 2|E| incidence matrix
    QQ = make_QQ(neighbors)
    
    # get distances, if applicable
    if use_distances:
        distances = get_edge_distances(G, QQ)
    else:
        distances = np.ones(QQ.shape[1])

    # get transport weights with linear programming
    weights, successful_transport = get_weights(QQ, pmf, distances, alpha)
    
    if not successful_transport:
        return G
    
    #insert edges
    num_edges = QQ.shape[1]//2
    for i in range(num_edges):
        edge = QQ[:,i].indices
        weight = weights[num_edges + i] - weights[i]
        if weight < 0:
            G.add_edge(edge[1], edge[0], weight = -weight, euc_dist = distances[i])
        else:
            G.add_edge(edge[0], edge[1], weight = weight, euc_dist = distances[i])
    
    G.add_nodes_from(enumerate([{'pos': tuple(i)} for i in X]))

    return G

def create_reg_digraph(X, neighbors, pmf, alpha = 1, eps = 1e-5):
    G_Q = nx.DiGraph()
    G_Q.add_nodes_from(enumerate([{'pos': tuple(i)} for i in X]))
    G_Q.add_nodes_from(enumerate([{'is_source': i} for i in pmf]))

    # get weights from QP
    Q_sparse = make_Q(neighbors)
    weights, lengths = get_reg_weights(X, Q_sparse, pmf, alpha, eps = eps)

    #insert edges with respective weights
    n_edges = len(weights)
    for i in range(n_edges):
        edge = Q_sparse[:,i].indices
        weight = weights[i]
        if weight < 0:
            G_Q.add_edge(edge[1], edge[0], weight = -weight)
        else:
            G_Q.add_edge(edge[0], edge[1], weight = weight)
            
    return G_Q, weights, lengths

def draw_digraph(G):

    # set colors and sizes for nodes
    source_color = '#FFBC38'   #orange
    sink_color = '#5D92F3'     #blue
    zero_color = '#B1B1B1'     #gray
    
    # classify whether nodes are sources, sinks, or neither
    is_source = np.array(list(nx.get_node_attributes(G,'is_source').values()))
    node_colors = np.where(is_source > 0, source_color, np.where(is_source == 0, zero_color, sink_color))
    
    node_sizes = 400*(np.abs(is_source)) + 10
    
    # get positions of nodes from node attributes
    pos = nx.get_node_attributes(G,'pos')

    # set colors (opacity) for edges
    edge_colors = [(.5,.5,.5,min(1,np.sqrt(G[u][v]['weight']))) for u,v in G.edges()]
    
    # draw graph
    nx.drawing.nx_pylab.draw_networkx(G, pos, with_labels = False, arrows = True, 
                                      arrowsize = 10, arrowstyle = '->', edge_color = edge_colors, 
                                      node_color = node_colors, 
                                      width = 3, 
                                      node_size = node_sizes)

    # display warning if graph was not connected
    if len(G.edges()) == 0:
        plt.text(.3, .5, 'graph was not connected', style='italic',
                 bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    return G
