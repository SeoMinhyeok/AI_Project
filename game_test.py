from math import inf as infinity
import random as R

def Mark():

    while(True):
        mark_u = str(input("o 나 x 중 하나를 입력하세요 : "))

        if mark_u == 'o':
            mark_c = 'x'
            print("당신은 o 입니다.")
            return mark_u, mark_c
        elif mark_u == 'x':
            mark_c = 'o'
            print("당신은 x 입니다.")
            return mark_u, mark_c
        else:
            print("잘못 입력하셨습니다. 다시 입력하세요.\n")

def Order():
    order = R.randrange(0, 2)

    if(order == 0):
        f_turn = "User"
        return f_turn
    else:
        f_turn = "Com"
        return f_turn

user, com = Mark()
Turn = Order()
value = []

for i in range(9):
    value.append('!')

def print_game():
    for i in range(0, 8, 3):
        print(value[i] + " " +  value[i + 1] + " " + value[i+2])

game_end = True

def check_convert(num):

    try:
        int(num)
        return True
    except ValueError:
        return False

def check_game_win(check):

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

def empty_cells():
    cells = []
    for i, y in enumerate(value):
        if(y=='!'):
            cells.append(i)
    return cells

def minmax(state, depth, player):

    if player == "Com":
        best = [-1, -infinity]
        mark = com
    else:
        best = [-1, +infinity]
        mark = user

    if depth == 0 or check_game_win(mark):
        score = evaluate()
        return [-1, score]

    for cell in empty_cells():
        location = cell

        value[location] = mark
        if(player == "Com"):
            player = "User"
        else:
            player = "Com"

        score = minmax(state, depth-1, player)

        value[location] = "!"
        score[0] = location

        if player == "Com":
            if(score[1] > best[1]):
                best = score #max
        else:
            if(score[1] < best[1]):
                best = score #min
    return best

def evaluate():

    if(check_game_win(com)):
        score = 1
    elif(check_game_win(user)):
        score = -1
    else:
        score = 0
    return score

def enter_value():

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
            print("말을 놓을 수 있는 위치가 아닙니다. 다시 엽력하세요")
            continue

        return position

while(game_end != False):

    if(Turn =="User"):
        position = enter_value()
        value[position] = user
        print_game()
        if(check_game_win(user)):
            break
        Turn = "Com"

    else:
        depth = len(empty_cells())
        if(depth == 9):
            location = 3
        else:
            print("computer")
            move = minmax(value, depth, Turn)
            print(move)
            value[move[0]] = com
            print_game()
            if(check_game_win(com)):
                print("com win")
                break
        Turn = 'User'