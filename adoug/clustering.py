# -*- coding: utf-8 -*-

'''file overview: clustering leaf nodes'''


def getLabel(target, source):
    for i in range(len(source)):
        if target == source[i]:
            return i


def byUniqueHop(hopSet):
    uniqueHop = []
    nodeMap_hop = []
    hopLabelDict = {}

    for node in hopSet:
        if hopSet[node] not in uniqueHop:
            uniqueHop.append(hopSet[node])
            nodeMap_hop.append(node)

    for node in hopSet:
        label = getLabel(hopSet[node], uniqueHop)
        hopLabelDict[node] = label

    return (hopLabelDict, nodeMap_hop)
