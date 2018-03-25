import cv2
import numpy as np
import math
from scipy.linalg import block_diag
timeStepSize=2
def createFilter(balls):
    kalman = cv2.KalmanFilter(4*balls,2*balls) # state is (x,y,v_x,v_y, )^t, sensors are (x,y, )
    H = [[1,0,0,0],
         [0,1,0,0]]
    hargs = [H] * balls
    kalman.measurementMatrix = np.array(block_diag(*hargs), np.float32)
    #kalman.measurementMatrix = np.array(map(lambda x: __g(x), range(2*balls)), np.float32)
    A = [[1,0,timeStepSize,0],
         [0,1,0,timeStepSize],
         [0,0,1,0],
         [0,0,0,1]]
    arguments = [A] * balls #n copies of the matrix amove
    kalman.transitionMatrix = np.array(block_diag(*arguments), np.float32)

    kalman.processNoiseCov = np.array(np.identity(4*balls) * .2, np.float32)
    #kalman.measurementNoiseCov = np.identity(2*balls) * .4

    return kalman

kalman1 = createFilter(1)
#kalman.correct(np.array([0,0,0,0], np.float32))
#print kalman.predict()
print kalman1.transitionMatrix
print kalman1.measurementMatrix
print kalman1.processNoiseCov

print 'now the working filter'
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * .2
print kalman.transitionMatrix
print kalman.measurementMatrix
print kalman.processNoiseCov
print kalman.predict()
print kalman1.predict()
