import numpy as np
import re

# with open("test.txt", "w+") as file:
#     array = []
#     file.write("Test 1: {}\n".format(array))
#     file.write("Test 2: {}".format(array))

# with open("test.txt", "r+") as file:
#     data = file.readlines()
#     array1 = data[0].rstrip().split("Test 1: ")[1][1]
#     array2 = list(filter(None, re.split("(:\W)", data[1].rstrip())))
#     print("This is array 1: {}".format(array1))
#     print("This is array 2: {}".format(array2))

# def writeListOfPaintings(lop, f="demo/list_of_paintings.txt"):
#     with open(f, "w+") as file:
#         print("Writing list of paintings to {}".format(f))
#         file.write("{}".format(lop))

# writeListOfPaintings([1, 2, 3])

# emotions_labeled_len = 500
# emotions_unlabeled_len = 2500
# iteration_counter = 0
# while(emotions_labeled_len < 3000):
#     iteration_counter += 1
#     emotions_labeled_len += 1 if int((0.05*emotions_unlabeled_len)) < 1 else int((0.05*emotions_unlabeled_len))
#     emotions_unlabeled_len -= 1 if int((0.05*emotions_unlabeled_len)) < 1 else int((0.05*emotions_unlabeled_len))
#     # print(iteration_counter)
#     print(emotions_labeled_len)
#     if(emotions_labeled_len >= 1000):
#         print(emotions_labeled_len)

# print(iteration_counter)
param_grid = {'n_estimators' : [100, 200, 500],
              'criterion' : ['gini', 'entropy'],
              'max_depth' : [1, 2],
              'min_samples_leaf' : [1, 2, 3],
              'max_features' : ["auto", "sqrt", "log2", 0.9, 0.2],
              'oob_score' : [True],
              'n_jobs' : [-1],
              'random_state' : [42]}
num_models = 1
model_settings = param_grid.values()
for k, v in param_grid.items():
    num_models *= len(v)
print(num_models)