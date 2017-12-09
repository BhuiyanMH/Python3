import math as m
print("The value of Pi is: ",m.pi)

#reload(module_name) is used for this purpose
#if the module changes during course of program, we need to reload it
print(dir(m))

#module is considered as a package if it contains _init_.py
#import Game.level.start
#with out package prefix, from Game.levle import start

