import numpy as np
import cv2
import csv
cap = cv2.VideoCapture('haavardTrick.avi')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
output_data = []
frameNumber = 0
output_path = './haavard_data.csv'
currentClusters = [] #An array of arrays of centers of circles from the past 5 frames. We use None to designate the loss of a circle there.
foundClusters = [] #An array of tuples (x, y, frame) where the cluster is found
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
outputVideo = cv2.VideoWriter('haavard_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width,frame_height))
def addFrameNumber(x):
    #print frameNumber
    x.append(frameNumber)
    return x

def clusterStatus(run): #Determines whether the cluster is complete or a false alarm.
    nonZero = len(filter(lambda x: x is not None, run))
    if nonZero >= 3: #If there are at least 3 non-None frames in the return
        maxY = max( map(lambda x: x[1] if x else None, run) ) #The largest y value.
        #print run
        if run[2] and maxY == run[2][1]: #And frame 2 has the largest y value,
            return 1
    elif len(run) >= 4 and run[-1]==run[-2]==None:
        return 2 #False alarm, the cluster should be removed.
    return 0 #This means the cluster is still indeterminate.

def examineClusters(currentClusters, foundClusters):
    newClusters = [c for c in currentClusters if clusterStatus(c) == 1]
    newClusters = map(lambda x: (x[2][0], x[2][1], frameNumber -2), newClusters) #puts the newclusters in the form(x, y, frame)
    currentClusters = [c for c in currentClusters if clusterStatus(c) == 0] #gets rid of unused clusters.
    foundClusters.extend(newClusters)
    return currentClusters, foundClusters

def moveClusters(currentClusters, newCircles):
    #Having read the next frame, and found the newCircles,
    #this function needs to update the currentClusters.
    scale = 1
    for run in currentClusters:
        lastCircle = reduce(lambda x,y: y if y else x, run) #The last non-None value of run.
        if lastCircle is None:
            print run
        nextCandidates = filter(lambda x: np.linalg.norm(np.array(x[0:2] - np.array(lastCircle[0:2])) ) < scale*lastCircle[3], newCircles )
        if len (nextCandidates) == 0:
            run.append(None) #no continuation for this run.
        if len(nextCandidates) > 1:
            #print 'multiple choices.'
            run.append(nextCandidates.pop(0))
        if len(nextCandidates) == 1:
            #print 'found one'
            run.append(nextCandidates.pop(0))
        if len(run) > 5:
            run.pop(0) # removes the data from the first frame in the window
    currentClusters.extend([ [c] for c in newCircles]) #The unused newCircles form the starts of new runs.
    return currentClusters


with open('haavard_data.csv', 'r') as f:
    peaks = list(csv.reader(f))
peaks = peaks[1:]
print peaks
peaks = [list(map(lambda x: int(float(x)), d)) for d in peaks]
while(True):
    frameNumber += 1
    ret, frame = cap.read()
    if frame is None:
        print('no frame')
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print frame.shape
    #edges = cv2.Canny(frame,100,200)
    fgmask = fgbg.apply(frame)
    foreground = cv2.bitwise_and(frame, frame, mask = fgmask)
    circles = cv2.HoughCircles(foreground,cv2.HOUGH_GRADIENT,1,20,
                            param1=20,param2=7,minRadius=2,maxRadius=5)
    if not circles is None:

        circleList = list(map(lambda x: addFrameNumber(x), circles.tolist()[0]))
        currentClusters = moveClusters(currentClusters, circleList)
        currentClusters, foundClusters = examineClusters(currentClusters, foundClusters)

        #output_data.extend(circleList)

        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(foreground,(i[0],i[1]),i[2],(255,255,255),2)
            # draw the center of the circle
            #cv2.circle(foreground,(i[0],i[1]),2,(0,0,255),3)
    currentPeaks = filter(lambda x: x[2] == frameNumber, peaks)
    if not currentPeaks is None:
        for p in currentPeaks:
            cv2.circle(foreground, (p[0], p[1]), 2, (255,255,255), 2)
    outputVideo.write(frame)
    cv2.imshow('frame',foreground)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


#print currentClusters
#print foundClusters
#write data
with open(output_path, 'w') as csvfile:
     fieldnames = ['x_position', 'y_position', 'frame']
     writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
     writer.writeheader()
     for data in foundClusters:
         x, y, f = data[0], data[1], data[2]
         writer.writerow({'x_position': x, 'y_position': y, 'frame': f})
