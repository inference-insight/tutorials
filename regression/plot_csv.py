import matplotlib.pyplot as plt 
import numpy as np 
import os, sys, random

file = open('data_train.csv', 'r')

xs = []
ys = []
for line in file.readlines():
    l = line.rstrip().split(',')
    xs.append(float(l[0]))
    ys.append(float(l[1]))

xs = (xs - np.mean(xs)) / np.std(xs)
ys = (ys - np.mean(ys)) / np.std(ys)

plt.scatter(xs, ys)
plt.title("Train Data")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.clf()