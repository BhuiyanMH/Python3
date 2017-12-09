# def function_name(parameters):
#     """dicstring"""
#     statements

def greet(name):
    '''this function greet someone with the given name'''
    print("Hello, "+ name+ ". Good Morning!")

greet("Kamal")

#if no return statement, the function will return a 'None' object

# variable declared outside the function can be read inside the function
#if we want to change the value, the varable needed to be declared as 'global'

#Default arg
#if we have a defult arg, all the args right of it must be default

def greet2(name, msg="Good morning"):
    print("Hello "+name+", "+msg)

greet2("Sazzad")
greet2("Sazzad", "Good Night")

#Python allows function  to be called using keyword arguments
#in that case order of arguments doen't matters

greet2(msg="Good Evening", name="Mejbah")

#Arbitary arguments: * is used before the parameter name,
# arguments gets wrapped into a tuple

def greet(*names):
    for name in names:
        print("hello ", name)

greet("Afnan", "Joyeta")

#Python Anonymous or Lambda function: can have any number of arguments
#and simgle expresson, the expression is evaluated and returned
#lamda arguments : expression
#it cal be use solely or filter() or map() function

#Python filter(): takes a function and list of argements, the function is
# called with all the items in the list and a new list is returned contained
# for which the function evaluates to true

myList = [1, 5, 4, 3, 6]
newList = list(filter(lambda x:(x%2==0), myList))
print(newList)

#map() is the same except that, returned list contains the elements returned by the func

my_list = [1, 5, 4, 6]
new_list = list(map(lambda x: x**2, my_list))
print(new_list)