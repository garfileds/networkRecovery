#! /usr/bin/python
# -*- coding: utf-8 -*-

'''@file overview: find shared path based on hop with predict method'''


class RecoveryTopo():

    '''recovery network topo with missed hop'''

    def __init__(self):
        self.network_raw = ''

    def find_shared_path_based_on_hop(likehood_map_leaf_leaf, alpha_map_c_measure, path, leaf_nodes, measure_nodes):
        shared_path_predict = {}

        for i in range(len(leaf_nodes) - 1):
            leaf1 = leaf_nodes[i]

            for j in range(i + 1, len(leaf_nodes)):
                leaf2 = leaf_nodes[j]

                for k in range(len(measure_nodes)):
                    measure = measure_nodes[k]

                    if leaf1 not in shared_path_predict:
                        shared_path_predict[leaf1] = {}
                        shared_path_predict[leaf1][leaf2] = {}
                    elif leaf2 not in shared_path_predict[leaf1]:
                        shared_path_predict[leaf1][leaf2] = {}

                    likehood = likehood_map_leaf_leaf[leaf1][leaf2]
                    alpha = alpha_map_c_measure[likehood][measure]
                    shared_path_predict[leaf1][leaf2][measure] = round(alpha * min(path[leaf1][measure], path[leaf2][measure]))

        return shared_path_predict
