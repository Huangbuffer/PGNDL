import numpy as np
import networkx as nx
from networkx.algorithms import community
import scipy.sparse as sp
import torch
import os
def generate_node_mapping(G, type=None):
    """
    :param G:
    :param type:
    :return:
    """
    if type == 'degree':
        s = sorted(G.degree, key=lambda x: x[1], reverse=True)
        new_map = {s[i][0]: i for i in range(len(s))}
    elif type == 'community':
        cs = list(community.greedy_modularity_communities(G))
        l = []
        for c in cs:
            l += list(c)
        new_map = {l[i]:i for i in range(len(l))}
    else:
        new_map = None

    return new_map

def torch_sensor_to_torch_sparse_tensor(mx):
    """ Convert a torch.tensor to a torch sparse tensor.
    :param torch tensor mx
    :return: torch.sparse
    """
    index = mx.nonzero().t()
    value = mx.masked_select(mx != 0)
    shape = mx.shape
    return torch.sparse.FloatTensor(index, value, shape)

def networkx_reorder_nodes(G, type=None):
    """
    :param G:  networkX only adjacency matrix without attrs
    :param nodes_map:  nodes mapping dictionary
    :return:
    """
    nodes_map = generate_node_mapping(G, type)
    if nodes_map is None:
        return G
    C = nx.to_scipy_sparse_matrix(G, format='coo')
    new_row = np.array([nodes_map[x] for x in C.row], dtype=np.int32)
    new_col = np.array([nodes_map[x] for x in C.col], dtype=np.int32)
    new_C = sp.coo_matrix((C.data, (new_row, new_col)), shape=C.shape)
    new_G = nx.from_scipy_sparse_matrix(new_C)
    return new_G
def zipf_smoothing(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    A_prime = A + np.eye(A.shape[0])
    out_degree = np.array(A_prime.sum(1), dtype=np.float32)
    int_degree = np.array(A_prime.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A_prime @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_plus(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D ^-1/2 * ( A + I ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ (A + np.eye(A.shape[0])) @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_laplacian(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_adj(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 *  A   * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total {:d} Trainable {:d}'.format(total_num, trainable_num))
    return {'Total': total_num, 'Trainable': trainable_num}


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)