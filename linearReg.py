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
directory = os.path.join(r"c:\Users\james\Desktop\College Courses\Summer Research 2018\LOL Raw data\Challenger_Patch6_23")
champList = championList_6_23
m = 0  # number of test cases 
X = np.zeros((134,1)) # 134*m matrx that stores champion choices for player1
Y = np.zeros((134,1)) # 134*m matrix that stores champion choices for player2
U = [] # 1*m matrix for winning data for each test cases

#********************GET DATA*************************#
#print(os.listdir(directory))
for ds in os.listdir(directory):
        for root,dirs,files in os.walk(directory+"\\"+ds):
                for file in files:
                        if file.endswith(".csv"):
                                currX = np.zeros((134,1)) # initializing x and y to be 134x1 zero vector
                                currY = np.zeros((134,1))
                                u_actual = 0 
                                with open(directory+"\\"+ds+"\\"+file,encoding="utf8") as f:
                                        reader = csv.reader(f)
                                        for row in reader:
                                                if row[0]:
                                                        if u_actual==0:
                                                                if row[3].lower()=="false":
                                                                        u_actual = -1
                                                                else:
                                                                        u_actual = 1
                                                        if int(row[0])<5:
                                                                currX[champList.index(row[2])] = 1
                                                        else:
                                                                currY[champList.index(row[2])] = 1
                                X = np.c_[X,currX] 
                                Y = np.c_[Y,currY] 
                                if U!=[]:
                                        U = np.c_[U,u_actual]
                                else:
                                        U = u_actual
                                m +=1

X = np.delete(X, 0, axis=1)
Y = np.delete(Y, 0, axis=1)
#print(U.T)

#****************************************************#
#Initialize data matrix B
start_time = time.time()

B = np.zeros((m, 134*134))
for k in range(0,m):
        for i in range(0, 134):
                for j in range(0,134):
                        tmp=X[i][k]*Y[j][k]
                        B[k][i*134+j] = tmp
#print(np.count_nonzero(B))

#*************Normal approach************************#
C = 1                        # hyperparameter term for regulization
I = np.identity(17956)          # 17956*17956 identity matrix
#rhs = np.dot(B.T, U.T)          # (17956*m) * (m*1)     = 17956*1
#print(rhs)
#lhs = np.dot(B.T, B)+ C*I       # (17956*m) * (m*17956) = 17956*17956
#print(lhs)
#a = np.linalg.solve(lhs, rhs)   # B.T u = B.T B a, solve for a (17956*1)
#print(a)
#A = a.reshape(134,134)          # reshape to A(134*134)
#print(A)
#print(np.count_nonzero(A))
#print("--- %s seconds ---" % (time.time() - start_time))
#A2 = A
#*************scipy approach*************************#
#works well with sparse matrices
start_time = time.time()
B = np.zeros((m,134*134))
for k in range(0,m):
        for i in range(0, 134):
                for j in range(0,134):
                        tmp=X[i][k]*Y[j][k]
                        B[k][i*134+j] = tmp

B = csc_matrix(B,(m,134*134))
BT = B.T
UT = U.T
rhs = BT.dot(UT)                # (17956*m) * (m*1)  = 17956*1
print(rhs)
rhs = csc_matrix(rhs, rhs.shape)
lhs = BT.dot(B)+ C*I       # (17956*m) * (m*17956) = 17956*17956
print(lhs)
lhs = csc_matrix(lhs, lhs.shape)
a = spsolve(lhs,rhs)
A = a.reshape(134,134)
print("--- %s seconds ---" % (time.time() - start_time))
#**********Ridge regression************#
sz = 134*133/2
E = np.zeros((int(sz),134,134))

#***********Regulization***************#
A = np.maximum(A, A.T)     #make A symmetric along the diagonal
for m in range(0,134):
        for n in range(0,134):
                if n>m:
                        tmp = A[m][n]
                        A[m][n] = -tmp

#**************Write result to file***************#
with open('linearResult.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in A]
