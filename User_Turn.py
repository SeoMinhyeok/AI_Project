# -*- coding: utf-8 -*-

def enter_value(value):

    while (True):
        position = input("\n말의 위치를 입력하세요(0 ~8) : ")
        if (check_convert(position)):
            position = int(position)
        else:
            print("문자는 입력할 수 없습니다.\n")
            continue

        if (position < 0 or position > 8):
            print("말을 놓을 수 있는 위치가 아닙니다. 다시 입력하세요.\n")

        if(value[position] != '!'):
            print("말을 놓을 수 있는 위치가 아닙니다. 다시 입력하세요")
            continue

        return position

def print_game(value):
    for i in range(0, 8, 3):
        print(value[i] + " " +  value[i + 1] + " " + value[i+2])

def check_convert(num):

    try:
        int(num)
        return True
    except ValueError:
        return False

def empty_cells(value):
    cells = []
    for i, y in enumerate(value):
        if(y=='!'):
            cells.append(i)
    return cells