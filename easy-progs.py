count = int(input("введите кол-во товара: "))
price = 158.0
if count > 5:
    price = 150.0
name = 'картошка'
print(f'{name}, стоит {price} за килограмм, вы набрали {count}кг, общая сумма {count*price} рублей')
