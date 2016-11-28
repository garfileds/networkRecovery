#! /usr/bin/python
# -*- coding: utf-8 -*-

'''@file overview: find shared path based on hop with predict method'''

import adoug.estimation as estimation
import adoug.predict as predict


class RecoveryTopo():

    '''recovery network topo with missed hop'''

    def __init__(self):
        self.network_raw = ''

    def find_shared_path_based_on_hop(hoplist, shared_path_known, path, leaf_nodes, measure_nodes, Epsilon):
        sharedPath = shared_path_known

        likehood = predict.caculateLikehoodMap(hoplist, leaf_nodes, measure_nodes, Epsilon)
        alphaMap = predict.getAlpha(likehood['map_measure'], sharedPath, path, measure_nodes)

        estimation_predict, estimation_predict_percent, shared_path_predict = estimation.estimateByPredict(leaf_nodes, measure_nodes[:1], likehood['map'], alphaMap, path, sharedPath)
        shared_path_predict['RMSE'] = estimation_predict
        # print('estimation with approach of predict\'s result is(RMSE): %s when Epsilon is %s.' % (str(estimation_predict), '1'))
        print('estimation with approach of predict\'s result is(RMSE): %s when Epsilon is %s.' % (str(1 - estimation_predict_percent), '1'))
