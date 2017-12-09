#print(*objects, sep='', end='', file=Sys.stdout, flush=False)
a = 5
print("The value of a is ", a)
print(1, 2, 3, 4,5, sep='#', end="$")

#output formatting
#formatting can be done using str.format() method

x  = 5
y = 10

print('\nThe value of x is {} and y is {}'.format(x, y))
print('hello {name}, {greeting}'.format(greeting="Good Morning", name="John"))

fValue = 2.4583942849
print("The value of x is: %3.2f", fValue)

#taking input from console imput([optional prompt])

num = input('Enter an expression: ')
print(num)
print(eval(num))

#Python import statement
from math import pi
print(pi)