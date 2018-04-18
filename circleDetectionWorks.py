import numpy as np
import cv2
cap = cv2.VideoCapture('juggling.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
output_data = []
frameNumber = 0
output_path = './site_swap_data.csv'
def addFrameNumber(x):
    #print frameNumber
    x.append(frameNumber)
    return x
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
                            param1=20,param2=30,minRadius=5,maxRadius=40)
    if not circles is None:
        circleList = list(map(lambda x: addFrameNumber(x), circles.tolist()[0]))
        output_data.extend(circleList)
        #print circleList
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(foreground,(i[0],i[1]),i[2],(255,255,255),2)
            # draw the center of the circle
            cv2.circle(foreground,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('frame',foreground)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

#write data
import csv
with open(output_path, 'w') as csvfile:
    fieldnames = ['x_position', 'y_position', 'frame', 'radius']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for data in output_data:
        x, y, r, f = data[0], data[1], data[2], data[3]
        writer.writerow({'x_position': x, 'y_position': y, 'frame': f, 'radius': r})
