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

def Value():

    for i in range(0, 8, 3):
        print(value[i] + " " +  value[i + 1] + " " + value[i+2])
game_end = True

def check_convert(num):

    try:
        int(num)
        return True
    except ValueError:
        return False

while(game_end != False):

    if(Turn =="User"):

        while(True):

            position = input("\n말의 위치를 입력하세요(0 ~8) : ")

            if(check_convert(position)):
                position = int(position)
            else:
                print("문자는 입력할 수 없습니다.\n")
                continue

            if(position < 0 or position > 8):
                print("말을 놓을 수 있는 위치가 아닙니다. 다시 입력하세요.\n")
            else:
                break

        if(value[position] == '!'):
            value[position] = user
            print(Value())

            if(value[0] == user and value[1] == user and value[2] == user
            or value[3] == user and value[4] == user and value[5] == user
            or value[6] == user and value[7] == user and value[8] == user
            or value[0] == user and value[3] == user and value[6] == user
            or value[1] == user and value[4] == user and value[7] == user
            or value[2] == user and value[5] == user and value[8] == user
            or value[0] == user and value[4] == user and value[8] == user
            or value[2] == user and value[4] == user and value[6] == user):
                print("당신이 이겼습니다!")
                break

            Turn = "Com"
        else:
            print("이미 다른 말이 있습니다. 다시 입력하세요.\n")
    else:

        while(True):
            position_c = R.randrange(0, 9)
            break

        if(value[position_c] == '!'):
            print("\nComputer")
            value[position_c] = com
            Value()
            if(value[0] == com and value[1] == com and value[2] == com
            or value[3] == com and value[4] == com and value[5] == com
            or value[6] == com and value[7] == com and value[8] == com
            or value[0] == com and value[3] == com and value[6] == com
            or value[1] == com and value[4] == com and value[7] == com
            or value[2] == com and value[5] == com and value[8] == com
            or value[0] == com and value[4] == com and value[8] == com
            or value[2] == com and value[4] == com and value[6] == com):
                print("컴퓨터가 이겼습니다!")
                break

            Turn = "User"
        else:
            continue