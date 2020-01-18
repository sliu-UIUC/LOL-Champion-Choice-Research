import os 
import numpy as np
import pandas as pd 
import csv
import glob
import sys
import time
from championList import championList_6_23
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

#read input 

A = np.loadtxt(open("linearResult.csv", "rb"), delimiter=",", skiprows=0) 
directory = os.path.join(r"c:\Users\james\Desktop\College Courses\Summer Research 2018\Code\Test set")
champList = championList_6_23

print(A.shape)
m = 0  #number of test cases
X = np.zeros((134,1)) # 134*1 matrx that stores champion choice for player1
Y = np.zeros((134,1)) # 134*1 matrix that stores champion choice for player2
U = []
correctPred = 0

for ds in os.listdir(directory):
        for root,dirs,files in os.walk(directory+"\\"+ds):
                for file in files:
                        if file.endswith(".csv"):
                                with open(directory+"\\"+ds+"\\"+file,encoding="utf8") as f:
                                        reader = csv.reader(f)
                                        U = 0                   #reinitialize variables
                                        X = np.zeros((134,1))
                                        Y = np.zeros((134,1))
                                        for row in reader:
                                                if row[0]:
                                                        if U==0:
                                                                if row[3].lower()=="false":
                                                                        U = -1
                                                                else:
                                                                        U = 1
                                                        if int(row[0])<5:
                                                                X[champList.index(row[2])] = 1
                                                        else:
                                                                Y[champList.index(row[2])] = 1
                                m += 1
                                prediction = np.dot(np.dot(X.T,A), Y)
                                if U == -1 and prediction < 0 :
                                	correctPred +=1
                                elif U == 1 and prediction > 0 :
                                	correctPred +=1

print('{:.1%}'.format(correctPred/m)) 

