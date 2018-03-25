import itertools
import numpy as np
measurementArray = range(10)
def addZeros(measurementArray):
    length = len(measurementArray)
    measurementArray = np.array([np.array(range(length))])
    measurementArray=measurementArray.reshape(2,length/2, order ='F')
    measurementArray = np.insert(measurementArray, 2, np.zeros(length/2),0)
    measurementArray = np.insert(measurementArray, 3, np.zeros(length/2),0)
    measurementArray=measurementArray.reshape(2*length, order = 'F')
    return measurementArray
measurementArray = addZeros(measurementArray)
print measurementArray
