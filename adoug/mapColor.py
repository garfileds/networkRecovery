#! /usr/bin/python
# -*- coding: utf-8 -*-


def colorMap(labelList, numOfNode, numOfMeasureNode):
    colorRender = []
    for label in labelList:
        if label == 0:
            colorRender.append('b')
        elif label == 1:
            colorRender.append('g')
        elif label == 2:
            colorRender.append('r')
        elif label == 3:
            colorRender.append('c')
        elif label == 4:
            colorRender.append('y')
        elif label == 5:
            colorRender.append('m')
        elif label == 6:
            colorRender.append('#ff8c00')

    colorRender.extend(['w' for i in range(numOfMeasureNode)])
    colorRender.extend(['#ff1493' for i in range(numOfNode-len(labelList)-numOfMeasureNode)])

    return colorRender
