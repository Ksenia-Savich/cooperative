import math
a = float(input("введите первую сторону треугольника"))
b = float(input("введите вторую сторону треугольника"))
ugol = float(input("введите угол между сторонами"))
c = math.sqrt(a**2 + b**2 - 2 * a * b * math.cos(math.radians(ugol)))
print(c)
