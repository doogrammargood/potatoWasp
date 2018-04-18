import numpy as np
import cv2
from matplotlib import pyplot as plt

X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y))

# convert to np.float32
Z = np.float32(Z)
#print Z
# define criteria and apply kmeans()
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#print ret
#print label
#print center
# Now separate the data, Note the flatten()
#A = Z[label.ravel()==0]
#B = Z[label.ravel()==1]

A = np.ndarray(shape=(2,4,3), dtype=float, order='F')
#col = np.array([])
print A
print "---"
print range(2)
#x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
print A[...,0]
#p = np.ndarray([ [[1,2,1],[4,4,4],[4,5,5]], [[1,2,1],[4,4,4],[4,5,5]] ])
# Plot the data

#plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1],c = 'r')
# plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()
