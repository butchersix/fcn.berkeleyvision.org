# python script to count number of lines inside error.log
num_lines = sum(1 for line in open('demo/error.log'))
print(num_lines)