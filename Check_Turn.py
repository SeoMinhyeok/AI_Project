# -*- coding: utf-8 -*-
from random import randrange

def Mark():

    while(True):
        mark_u = raw_input("o 나 x 중 하나를 입력하세요 : ")

        if mark_u == 'o':
            mark_c = 'x'
            print("당신은 o 입니다.")
            break
        elif mark_u == 'x':
            mark_c = 'o'
            print("당신은 x 입니다.")
            break
        else:
            print("잘못 입력하셨습니다. 다시 입력하세요.\n")
    return mark_u, mark_c

def Order():
    order = randrange(0, 2)

    if(order == 0):
        f_turn = "User"
        return f_turn
    else:
        f_turn = "Com"
        return f_turn

