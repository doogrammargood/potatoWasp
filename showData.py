from csv import reader
from matplotlib import pyplot
import numpy as np
from longestIncSubsequence import subsequence
from copy import deepcopy

errorSize = 2# This is the error of points away from the parabolas

def closeEnough(pt, polyFrame, polyX):
    #Calculates the residual of the point and polynomial.
    diffFrame = abs(polyFrame(pt[2]) - pt[1])
    diffX = abs(polyX(pt[0]) - pt[1])
    return diffFrame < errorSize and diffX < errorSize

tossList = []
with open('testData.csv', 'r') as f:
    data = list(reader(f))
    #assume data has the form [x, y, frame, radius]
data = data[1:] #drops the first array, which has the names of the attributes.
dataCopy = deepcopy(data)
data = [list(map(lambda x: float(x), d)) for d in data] # converts data to float
#pyplot.plot(frame, y_vals, '.')
#pyplot.show()

def addToss(tossList, data, xvalues = "increasing", yvalues = "increasing"):
    #Finds a toss and adds it to the list. Returns a new tossList and data without those points.
    minframe = 50
    maxframe = 60

    #if xvalues == "increasing":
    #    subs = subsequence(data, lambda x,y: x[0] < y[0]) #The longest subsequence of points that is all increasing in X,
    #else:
    #    subs = subsequence(data, lambda x,y: x[0] > y[0])
    #fsubs = subsequence(subs, lambda x,y: x[2]<y[2]) #And increasing in frame number
    #if yvalues == "increasing":
    #    subs2 = subsequence(fsubs, lambda x,y: x[1] < y[1]) #And increasing in Y.
    #else:
    #    subs2 = subsequence(fsubs, lambda x,y: x[1] > y[1]) #Or decreasing in Y.
    subs2 = [d for d in data if d[2] > minframe and d[2] < maxframe]
    potentialThrow = np.array(subs2)
    #potentialThrow = np.array(data)
    condition = True
    counter = 0
    while condition:
        rand = potentialThrow[np.random.choice(potentialThrow.shape[0], 5, replace=False)] #Randomly choose 5 points
        frameArc = np.polyfit([d[2] for d in rand],[d[1] for d in rand], 2, full=True) #polynomial for frames vs y values.
        xArc = np.polyfit([d[0] for d in rand],[d[1] for d in rand], 2, full=True) #polynomial for x vs y values.
        condition = (frameArc[1][0] > 10) or (xArc[1][0] > 10) # choose new points if either the frame or x parabolas dont fit.
        #if counter > 50:
        #    print 'broken'
        #    break
        #else: counter+=1
    polyFrame = np.poly1d(frameArc[0])
    polyX = np.poly1d(xArc[0])
    tossPoints = [x for x in data if closeEnough( x, polyFrame, polyX )] #gathers the points which are on the parabola.

    def recalculate(): #takes the tosspoints, fits them with a parabola, and recalculates toss points until termination.
        print 'tosspoints'
        print tossPoints
        nextFrameArc = np.polyfit([d[2] for d in tossPoints], [d[1] for d in tossPoints], 2, full = True)
        nextXArc = np.polyfit([d[0] for d in tossPoints], [d[1] for d in tossPoints], 2, full = True)
        nextTossPoints = [x for x in data if closeEnough( x, np.poly1d(nextArc[0]), np.poly1d(nextXArc[0]) )]
        return nextArc, nextTossPoints

    frameArc, tossPoints = recalculate() #
    frames = [t[2] for t in tossPoints]
    minFrame = min(frames)
    maxFrame = max(frames)
    tossList.append((toss, minFrame, maxFrame, len(tossPoints))) # adds the polynomial to the list, along with the smallest and largest frame where it occurs.

    data = [x for x in data if x not in tossPoints]
    return tossList, data

addToss(tossList, data, xvalues = 'decreasing', yvalues = 'decreasing')
addToss(tossList, data)
addToss(tossList, data)
for toss in tossList:
    xitems = np.linspace(toss[1],toss[2], 500)
    print toss[0]
    pyplot.plot(xitems, toss[0](xitems))
pyplot.plot([d[2] for d in dataCopy], [d[1] for d in dataCopy], '.')
pyplot.show()
print tossList
