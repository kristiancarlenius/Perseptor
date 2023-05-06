# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:15:28 2022

@author: Kristian
"""
import matplotlib.pyplot as plt
import numpy as np

liste = [[], []]
def unit_step(v):
	if v >= 0:
		return 1
	else:
		return 0
	
def perceptron(x, w, b):
    v= np.dot(w, x) + b
    #print("db:", v)
    liste[0].append(np.sum(x))
    liste[1].append(v)
    return unit_step(v)

def OR_percep(x):
    w = np.array([1, 1])
    b = -0.5
    return perceptron(x, w, b)

def LOR_percep(x, y, a, z):
    w = np.array([1+a, 1+z])
    b = y
    return perceptron(x, w, b)
"""
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("OR({}, {}) = {}".format(0, 0, OR_percep(example4)))
print("OR({}, {}) = {}".format(0, 1, OR_percep(example3)))
print("OR({}, {}) = {}".format(1, 0, OR_percep(example2)))
print("OR({}, {}) = {}".format(1, 1, OR_percep(example1)))
print("liste:", liste)
np_a = np.array(liste)
plt.plot(np_a[0], np_a[1])
"""
import csv

with open('E:\\assign4data.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

liste2 = []
for i in range(1, len(data)):
    liste2.append(data[i][2])

altstemmer = True
y = -1
z = 1
a = 1
feil2 = 0
while altstemmer:
    liste3 = []
    feil = 0
    for i in range(1, len(data)):
        #print(data[i])
        svar = LOR_percep(np.array([float(data[i][0]), float(data[i][1])]), y, a, z)
        print("OR({}, {}) = {}".format(data[i][0], data[i][1], svar), "fasit:", data[i][2], "y:", y)
        print("")
        if(svar != data[i][2]):
            feil += 1
            if(svar == 0):
                a += 0.2
                z -= 0.1
            else:
                z += 0.2
                a -= 0.1
        liste3.append(svar)
    #if(feil > feil2)
    if(liste3 == liste2):
        altstemmer=False
    else:
        y += 0.01
    feil2  = feil
print("ferdig", y)
    #plt.show()