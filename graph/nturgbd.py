import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

inward_ori_index_B = [(1, 2), (2, 21), (3, 21), (4, 3), (6, 5), (7, 6),
                     (8, 7), (10, 9), (11, 10), (12, 11),
                     (14, 13), (15, 14), (16, 15), (18, 17), (19, 18),
                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_B = [(i - 1, j - 1) for (i, j) in inward_ori_index_B]
outward_B = [(j, i) for (i, j) in inward_B]
neighbor_B = inward_B + outward_B

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.B = self.get_adjacency_matrix_B(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.neighbor_B = neighbor_B 

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

    def get_adjacency_matrix_B(self, labeling_mode=None):
        if labeling_mode is None:
            return self.B
        if labeling_mode == 'spatial':
            B = tools.get_spatial_graph(num_node, self_link, inward_B, outward_B)
        else:
            raise ValueError()
        return B


class Hypergraph:
    def __init__(self):
        self.G_part = self.generate_G_part()
        self.G_body = self.generate_G_body()

    def generate_G_part(self, variable_weight=False):
        H = np.zeros((25, 10))
        H[0][0], H[1][0], H[20][0] = 1, 1, 1
        H[2][1], H[3][1], H[20][1] = 1, 1, 1
        H[9][2], H[10][2], H[11][2], H[23][2], H[24][2] = 1, 1, 1, 1, 1
        H[8][3], H[9][3], H[20][3] = 1, 1, 1
        H[4][4], H[5][4], H[20][4] = 1, 1, 1
        H[5][5], H[6][5], H[7][5], H[21][5], H[22][5] = 1, 1, 1, 1, 1
        H[17][6], H[18][6], H[19][6] = 1, 1, 1
        H[0][7], H[16][7], H[17][7] = 1, 1, 1
        H[0][8], H[12][8], H[13][8] = 1, 1, 1
        H[13][9], H[14][9], H[15][9] = 1, 1, 1

        H = np.array(H)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            return G

    def generate_G_body(self, variable_weight=False):
        H = np.zeros((25, 5))
        H[0][0], H[1][0], H[2][0], H[3][0], H[20][0], H[4][0], H[8][0], H[12][0], H[16][0] = 1, 1, 1, 1, 1, 1, 1, 1, 1
        H[8][1], H[9][1], H[10][1], H[11][1], H[23][1], H[24][1] = 1, 1, 1, 1, 1, 1
        H[4][2], H[5][2], H[6][2], H[7][2], H[21][2], H[22][2] = 1, 1, 1, 1, 1, 1
        H[16][3], H[17][3], H[18][3], H[19][3] = 1, 1, 1, 1
        H[12][4], H[13][4], H[14][4], H[15][4] = 1, 1, 1, 1

        H = np.array(H)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            return G

