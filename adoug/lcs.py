# -*- coding: utf-8 -*-

'''Longest substring question'''


def getcomlen(firststr, secondstr):
    comlen = 0
    while firststr and secondstr:
        if firststr[0] == secondstr[0]:
            comlen += 1
            firststr = firststr[1:]
            secondstr = secondstr[1:]
        else:
            break
    return comlen


def lcs_base(input_x, input_y):
    max_common_len = 0
    common_index = 0
    for xtemp in range(0, len(input_x)):
        for ytemp in range(0, len(input_y)):
            com_temp = getcomlen(input_x[xtemp: len(input_x)], input_y[ytemp: len(input_y)])
            if com_temp > max_common_len:
                max_common_len = com_temp
                common_index = xtemp

    return (max_common_len - 1, input_x[common_index:common_index + max_common_len])


def lcs_dp(input_x, input_y):
    # input_y as column, input_x as row
    dp = [([0] * len(input_y)) for i in range(len(input_x))]
    maxlen = maxindex = 0
    for i in range(0, len(input_x)):
        for j in range(0, len(input_y)):
            if input_x[i] == input_y[j]:
                if i != 0 and j != 0:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if i == 0 or j == 0:
                    dp[i][j] = 1
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    maxindex = i + 1 - maxlen
    return (maxlen - 1, input_x[maxindex:maxindex + maxlen])
