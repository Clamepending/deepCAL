from ctypes import *


# so_file = "C:/Users/Mark/Desktop/savio/deepCAL/deepcallib_helper.so"
so_file = "./deepcallib_helper.so"

my_functions = CDLL(so_file)

print(my_functions.square(10))