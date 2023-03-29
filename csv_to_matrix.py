import numpy as np

# carga data

data = np.genfromtxt('ShaAng_W50.csv', delimiter=',')[1:]
matrix = np.zeros((2100, 2), dtype=float) 

# Iteración 
row = 0
col = 0
for i in range(2100):
    matrix[i,0] = int(row + 1)
    matrix[i,1] = float(data[row, col])
    
    row += 1
    if row % 100 == 0:
        col += 1
        row = 0

# seteo decimales para revisar
#np.set_printoptions(precision=15)

#np.savetxt('matrix.txt', matrix, fmt='%.15f')

#print(matrix[:200])
