import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = I - N
    return A

def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = np.stack((I, N))
    return A

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_DAD_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A

def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    A = I - normalize_undigraph(edge2mat(neighbor, num_node))
    return A

# Joint index:
# {0:  'Nose'}
# {1:  'Neck'},
# {2:  'RShoulder'},
# {3:  'RElbow'},
# {4:  'RWrist'},
# {5:  'LShoulder'},
# {6:  'LElbow'},
# {7:  'LWrist'},
# {8:  'Hip'},
# {9:  'RHip'},
# {10: 'RKnee'},
# {11: 'RAnkle'},
# {12: 'LHip'},
# {13: 'LKnee'},
# {14: 'LAnkle'},
# {15: 'REye'},
# {16: 'LEye'},
# {17: 'REar'},
# {18: 'LEar'}
# {19: 'LFoot1'}
# {20: 'LFoot2'}
# {21: 'LFoot3'}
# {22: 'RFoot1'}
# {23: 'RFoot2'}
# {24: 'RFoot3'}

# Edge format: (origin, neighbor)
num_node = 25
self_link = [(i, i) for i in range(num_node)]

inward =[(0,1),(0,15),(0,16),(1,2),(1,5),
    (1,8),(2,3),(3,4),(5,6),(6,7),
    (8,9),(8,12),(9,10),(10,11),(12,13),
    (13,14),(15,17),(16,18),(14,21),(14,19),
    (19,20),(11,24),(11,22),(22,23)]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix
    For more information, please refer to the section 'Partition Strategies' in our paper.
    """
    def __init__(self, labeling_mode='uniform'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        return A

# def main():
#     mode = ['uniform', 'distance*', 'distance', 'spatial', 'DAD', 'DLD']
#     np.set_printoptions(threshold=np.nan)
#     for m in mode:
#         print('=' * 10 + m + '=' * 10)
#         print(Graph(m).get_adjacency_matrix())


# if __name__ == '__main__':
#     main()