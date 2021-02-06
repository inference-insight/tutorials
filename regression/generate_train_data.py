import random, os, sys
import numpy as np

num = 750
f = open("data_train.csv", 'w')
#f = open("data_test.csv", 'w')
pts = []

for i in range(num):

    r = np.random.normal()
    x = random.uniform(-2,2)
    y = np.cos(x)*np.sin(x**2) + 0.5*x + 3.5
    noise = noise = np.random.normal(scale=0.1)
    y = y + noise
    pts.append((x,y))

random.shuffle(pts)
for pt in pts:
    f.write(str(pt[0]) + ',' + str(pt[1]) + '\n')
f.close()

