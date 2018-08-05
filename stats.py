# python script to count number of lines inside error.log
"""
    REFERENCES USED:
    1. https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
"""
import os, os.path


num_lines = sum(1 for line in open('demo/error.log'))
print("Number of errors: {}".format(num_lines))
DIR = "demo/output"
segmented = (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - num_lines) / 3
print("Number of paintings segmented: {}".format(segmented))