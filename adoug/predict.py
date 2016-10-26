# -*- coding:utf-8 -*-

'''file overview: Predictive Shared Path Length Estimation, caculate params(alpha with different c and k)'''


def caculateLikehoodMap(hoplist, leafNodes, measureNodes, Epsilon):
    likehoodMap = {}
    likehood_measure = {}
    for i in range(len(leafNodes)):
        for j in range(i + 1, len(leafNodes)):
            source = leafNodes[i]
            dest = leafNodes[j]
            for k in range(len(measureNodes)):
                numOfMeaureAnother = 0
                measure = measureNodes[k]

                for l in range(len(measureNodes)):
                    if abs(abs(hoplist[i][k]) - hoplist[j][k]) - abs(hoplist[i][l] - hoplist[j][l]) < Epsilon:
                        numOfMeaureAnother += 1

                if source not in likehoodMap:
                    likehoodMap[source] = {}
                    likehoodMap[source][dest] = {}
                elif dest not in likehoodMap[source]:
                    likehoodMap[source][dest] = {}
                likehoodMap[source][dest][measure] = numOfMeaureAnother

                if measure not in likehood_measure:
                    likehood_measure[measure] = {}
                    likehood_measure[measure][numOfMeaureAnother] = []
                elif numOfMeaureAnother not in likehood_measure[measure]:
                    likehood_measure[measure][numOfMeaureAnother] = []
                likehood_measure[measure][numOfMeaureAnother].append([source, dest])

    return {'map': likehoodMap, 'map_measure': likehood_measure}


def getAlpha(likehood_measure, sharedPath, path, measureNodes):
    alphaMap = {}

    for C in range(1, len(measureNodes) + 1):
        alphaMap[C] = {}

        for measure in measureNodes:
            alphaMap[C][measure] = {}
            sum_inter = 0

            if C in likehood_measure[measure]:
                pairlist = likehood_measure[measure][C]
            elif str(C) in likehood_measure[measure]:
                pairlist = likehood_measure[measure][str(C)]
            else:
                continue

            pairlist_selected = pairlist[:10]
            for pair in pairlist:
                pair0 = pair[0]
                pair1 = pair[1]
                if pair0 == pair1:
                    continue

                try:
                    shared = sharedPath[pair0][pair1][measure]
                except KeyError:
                    shared = sharedPath[pair1][pair0][measure]
                sum_inter += shared / min(len(path[pair0][measure]), len(path[pair1][measure]))

            alphaMap[C][measure] = sum_inter / len(pairlist)

    return alphaMap
