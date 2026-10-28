import random

a = random.randint(1,100)
x = 0

while x != a:
    user_input = input("Угадайте число от 1 до 100: ")

    if user_input.isdigit():
        x = int(user_input)
        
        if x == a:
            print("вы угадали число!")
        elif x < a:
            print("введите число побльше")
        elif x > a:
            print("введите число поменьше")
    else:
        print("введите именно число")
