import numpy as np
import re

with open("test.txt", "w+") as file:
    array = []
    file.write("Test 1: {}\n".format(array))
    file.write("Test 2: {}".format(array))

with open("test.txt", "r+") as file:
    data = file.readlines()
    array1 = data[0].rstrip().split("Test 1: ")[1][1]
    array2 = list(filter(None, re.split("(:\W)", data[1].rstrip())))
    print("This is array 1: {}".format(array1))
    print("This is array 2: {}".format(array2))

# def writeListOfPaintings(lop, f="demo/list_of_paintings.txt"):
#     with open(f, "w+") as file:
#         print("Writing list of paintings to {}".format(f))
#         file.write("{}".format(lop))

# writeListOfPaintings([1, 2, 3])