# -*- coding: utf-8 -*-
from Check_Turn import Mark, Order
from User_Turn import enter_value, print_game, empty_cells
from Check_Winner import check_game_win, minmax
from Draw_Part import Draw
import pygame

# 원래 틱택토에 쓰이던 변수
matrix = []
value = []
user, com = Mark()
Turn = Order()

# 그림 그리는 것에 쓰이는 변수
window_width = 800
window_height = 500
board_width = 500
bg_color = (128, 128, 128)
fps = 90
fps_clock = pygame.time.Clock()

for i in range(15):
    matrix.append([])

    for j in range(15):
        matrix[i].append(0)

for i in range(9):
    value.append('!')

game_end = True

# Draw_Part 관련 코드
pygame.init()
surface = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Omok game")
surface.fill(bg_color)
draw_game = Draw(surface)
draw_game.init_game()

while(game_end != False):

    if(Turn =="User"):
        position = enter_value(value)
        value[position] = user
        print_game(value)
        if(check_game_win(user,value)):
            break
        Turn = "Com"

    else:
        depth = len(empty_cells(value))
        if(depth == 9):
            location = 3
        else:
            print("computer")
            move = minmax(value, depth, Turn, com, value, user)
            print(move)
            value[move[0]] = com
            print_game(value)
            if(check_game_win(com, value)):
                print("com win")
                break
        Turn = 'User'