while True:
    #просим пользователя ввести значение в цельсиях
    inpt = input("введите температуру, указывая единицы измерения (например, 40C или 100F): ")

    #проверяем корректность ввода
    if len(inpt) < 2:
        print("ошибка, введите число в формате 40C или 100F")
    else:
        a = inpt[:-1] #оставляем только число, без буквы
        #проверяем, верна ли единица измерения
        letter = inpt[-1]
        if letter not in ("C", "F"):
            print("ошибочка, единица измерения должна быть C или F")
        else:
            try:
                value = float(a)
            except ValueError:
                print("ошибка, число не распознано")
            else:
                if letter == "C":
                    result = (value * 1.8) + 32
                    res_unit = "F"
                else: #letter = F
                    result = (value - 32) / 1.8
                    res_unit = "C"

                #округляем до одного знака после запятой или убираем ноль после запятой
                if result == int(result):  #104.0 == 104
                    result = int(result)
                else:
                    result = round(result, 1)

                print(f"{result}{res_unit}")

