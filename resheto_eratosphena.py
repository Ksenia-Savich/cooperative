#просим пользователя ввести число n
x = input("введите число больше 2: ")

#проверяем, является ли числом
if not x.isdigit():
    print("ошибка, введите положительное число больше 2!")
else:
    n = int(x)

    if n < 2:
        print("простых чисел от 2 до", n, "не существует")
    else:
        #создаём список от 2 до n
        a = [True] * (n + 1)

        #решето
        i = 2
        while i * i <= n:
            if a[i] == True:
                j = i * i
                while j <= n:
                    a[j] = False
                    j = j + i
            i = i + 1


        #возвращаем: все числа i, для которых A[i] = true.
        for num in range(2, n + 1):
            if a[num] == True:
                print(num)
