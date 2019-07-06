import random as R
def Mark():
    while(1):
        mark_u = str(input("o 나 x 중 하나를 입력하시오 : "))

        if mark_u == 'o':
            mark_c = 'x'
            print("당신은 o 입니다.")
            return mark_u, mark_c

        elif mark_u == 'x':
            mark_c = 'o'
            print("당신은 x 입니다.")
            return mark_u, mark_c

        else:
            print("잘못 입력하셨습니다. 다시 입력하세요.")

def Order():
    order = R.randrange(0, 2)

    if(order == 0):
        f_turn = "User"
        return f_turn

    elif(order == 1):
        f_turn = "Com"
        return f_turn

user, com = Mark()
F_turn = Order()
value = []

for i in range(9):
    value.append('!')

for i in range(3):
    print(value[i*3]+value[i*3+1]+value[i*3+2])
