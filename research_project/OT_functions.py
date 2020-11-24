import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, csc_matrix, hstack
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

# Functions for calculating neighbors
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


# Functions for Optimal Transport Algorithm
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

def get_edge_distances(G, QQ):
    
    n_edges = QQ.shape[1]
    
    def edge_euc_distance(edge):
        u_pos = G.nodes[edge[0]]['pos']
        v_pos = G.nodes[edge[1]]['pos']
        return np.linalg.norm([u_pos[0]-v_pos[0], u_pos[1]-v_pos[1]])
    
    distances = np.apply_along_axis(edge_euc_distance, 1, QQ.indices.reshape(-1,2)[:(QQ.shape[1])//2,:])
    return np.tile(distances, 2)
    
def get_weights(QQ, pmf, edge_distances = 'None', alpha = 1):
    if edge_distances == 'None':
        edge_distances = np.ones(QQ.shape[1])
    output = linprog(c = edge_distances, A_eq = QQ, b_eq = -pmf)
    
    return np.round(output.x, 8), output.success


# Functions for creating and visualizing graphs
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

def create_digraph(X, neighbors, pmf, alpha = 1):
    # Inputs:
    # X - a cluster of points in the form of a 2D numpy array
    # QQ - a sparse matrix of -1, 0, 1 that encodes vertex adjacencies
    # weights - an array of weights to be assigned to each edge in QQ
    #
    # Output:
    # G - the resulting graph
    #
    # Displays a graph with nodes at the points of X, connected with weighted edges
    
    assert math.isclose(pmf.sum(), 0, abs_tol = 1e-06), 'total source mass must equal total sink mass'
    
    G = nx.DiGraph()
    
    # insert nodes, with position stored as a tuple in attribute 'pos'
    G.add_nodes_from(enumerate([{'pos': tuple(i)} for i in X]))
    # assign attribute 'is_source' to each nodes, with a positive value for sources and negative value for sinks
    G.add_nodes_from(enumerate([{'is_source': i} for i in pmf]))
    
    # create |V| x 2|E| incidence matrix
    QQ = make_QQ(neighbors)
    
    # get Euclidean distance of all edges within QQ
    distances = get_edge_distances(G, QQ)
    
    # get edge weights
    weights, successful_transport = get_weights(QQ, pmf, distances, alpha)
    
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

    if successful_transport:
        return G
    else:
        return None

def draw_digraph(G):
    
    # set colors and sizes for nodes
    source_color = '#FFBC38'   #orange
    sink_color = '#5D92F3'     #blue
    zero_color = '#B1B1B1'     #gray
    
    # classify whether nodes are sources, sinks, or neither
    is_source = np.array(list(nx.get_node_attributes(G,'is_source').values()))
    node_colors = np.where(is_source > 0, source_color, np.where(is_source == 0, zero_color, sink_color))
    
    node_sizes = 400*(np.abs(is_source)) + 10
    
    # set colors (opacity) for edges
    edge_colors = [(.5,.5,.5,min(1,np.sqrt(G[u][v]['weight']))) for u,v in G.edges()]
    
    # get positions of nodes from node attributes
    pos = nx.get_node_attributes(G,'pos')
    
    # draw graph
    nx.drawing.nx_pylab.draw_networkx(G, pos, with_labels = False, arrows = True, 
                                      arrowsize = 10, arrowstyle = '->', edge_color = edge_colors, 
                                      node_color = node_colors, 
                                      width = 3, 
                                      node_size = node_sizes)