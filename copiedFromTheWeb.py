import numpy as np
import cv2
cap = cv2.VideoCapture('crazy5b.avi')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
tracker = cv2.Tracker_create("MIL")
boolTracked = False #controls noticing objects
while(True):
    ret, frame = cap.read()
    if not ret:
        print('no frame')
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Consider shinking/dilating this mask.
    #print frame.shape
    #edges = cv2.Canny(frame,100,200)
    fgmask = fgbg.apply(frame)
    foreground = cv2.bitwise_and(frame, frame, mask = fgmask)
    circles = cv2.HoughCircles(foreground,cv2.HOUGH_GRADIENT,1,20,
                            param1=20,param2=20,minRadius=5,maxRadius=10)
    if boolTracked:
        ok, bbox = tracker.update(foreground)
        if ok:
            print('tracking')
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            print p1
            print p2
            cv2.rectangle(frame, p1, p2, (255,255,255), 2, 1)
        else:
            print("tracking failure detected")


    if not circles is None:
        newBoxes = []
        #print(circles)
        for i in circles[0,:]:
            newBoxes.append([(i[0]-i[2],i[1]-i[2]),(i[0]+i[2],i[1]+i[2])])
            # draw the outer circle
            #cv2.circle(foreground,(i[0],i[1]),i[2],(255,255,255),2)
            #cv2.rectangle(foreground,newBoxes[0][0],newBoxes[0][1],(255,255,255))
            # draw the center of the circle
            # cv2.circle(foreground,(i[0],i[1]),2,(255,255,255),3)
        if not boolTracked:
            boolTracked = True
            tracker.init(frame, cv2.boundingRect(np.array([newBoxes[0][0], newBoxes[0][1]])))
            print newBoxes

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
