# -*- coding: utf-8 -*-
from User_Turn import empty_cells
import math

math.inf = float('inf')

def evaluate(com, user, value):

    if(check_game_win(com, value)):
        score = 1
    elif(check_game_win(user, value)):
        score = -1
    else:
        score = 0
    return score

def check_game_win(check,value):

    if(value[0] == check and value[1] == check and value[2] == check
    or value[3] == check and value[4] == check and value[5] == check
    or value[6] == check and value[7] == check and value[8] == check
    or value[0] == check and value[3] == check and value[6] == check
    or value[1] == check and value[4] == check and value[7] == check
    or value[2] == check and value[5] == check and value[8] == check
    or value[0] == check and value[4] == check and value[8] == check
    or value[2] == check and value[4] == check and value[6] == check):
        return True
    else:
        return False

def minmax(state, depth, player, com, value, user):

    if player == "Com":
        best = [-1, -math.inf]
        mark = com
    else:
        best = [-1, +math.inf]
        mark = user

    if depth == 0 or check_game_win(mark, value):
        score = evaluate(com, user, value)
        return [-1, score]

    for cell in empty_cells(value):
        location = cell

        value[location] = mark
        if(player == "Com"):
            player = "User"
        else:
            player = "Com"

        score = minmax(state, depth-1, player, com, value, user)

        value[location] = "!"
        score[0] = location

        if player == "Com":
            if(score[1] > best[1]):
                best = score #max
        else:
            if(score[1] < best[1]):
                best = score #min
    return best