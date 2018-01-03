#From code by Stephen Meschke: https://sites.google.com/view/jugglingdataset/home

import numpy as np
import math
import cv2
cap = cv2.VideoCapture('crazy5b.avi')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#set parameters for tracker failure
object_size = 8 #must be even
delta = 5

#stuff for the mouse callback function
global click_list
click_list = []
positions = []
for i in range(100000): positions.append((0,0))
def callback(event, x, y, flags, param):
    if event == 1: click_list.append((x,y))
cv2.namedWindow('img')
cv2.setMouseCallback('img', callback)

#function to get the roi_hist
def get_roi_hist(frame, c,r,w,h):
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

#get initial tracking window from user input
ret,frame = cap.read()
cv2.imshow('img', frame)
cv2.waitKey(0)
a = click_list[-1][0]-object_size/2, click_list[-1][1]-object_size/2
track_window = (a[0], a[1], object_size, object_size)
roi_hist = get_roi_hist(frame, a[0], a[1], object_size, object_size)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 17, 1 )
frame_number = 0

def distance(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

#kalman filter parameters
meas=[]
pred=[]
frame = np.zeros((400,400,3), np.uint8) # drawing canvas
mp = np.array((2,1), np.float32) # measurement
tp = np.zeros((2,1), np.float32) # tracked / prediction
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * .2


while True:
    #read frame and apply tracking
    if not ret:
        break
    ret, frame = cap.read()
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print np.any(frame2)
    fgmask = fgbg.apply(frame2)
    open_fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((8,8),np.uint8)) #Gets rid of noise.
    foreground = cv2.bitwise_and(frame, frame, mask = fgmask)
    frame = foreground
    #circles = cv2.HoughCircles(foreground,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=20,minRadius=5,maxRadius=10)


    try: lll = frame.shape
    except: break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    # apply meanshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    # Draw it on image
    pts = np.int0(cv2.boxPoints(ret))
    img2 = frame.copy()
    img2 = cv2.polylines(img2,[pts],True, (0,0,255),2)

    #add the center point of the box to the list of tracked positions
    positions[frame_number] = int(((pts[0][0] + pts[1][0] + pts[2][0] +pts[3][0])/4.0)), int((pts[0][1] + pts[1][1] + pts[2][1] +pts[3][1])/4.0)

    #kalman filtering
    mp = np.array([[np.float32(positions[frame_number][0])],[np.float32(positions[frame_number][1])]])
    meas.append((positions[frame_number]))
    kalman.correct(mp)
    tp = kalman.predict()
    pred.append((int(tp[0]),int(tp[1])))

    #draw the tails on the left image
    try:
        for i in range(50): cv2.line(img2,pred[len(pred)-(i+1)],pred[len(pred)-(2+i)],(255,0,0),4)
        for i in range(50): cv2.line(img2,meas[len(meas)-(i+1)],meas[len(meas)-(2+i)],(0,255,0),2)
    except: pass

    cv2.imshow('img',img2)
    cv2.imshow('track_window', frame[track_window[1]:track_window[1]+object_size, track_window[0]:object_size+track_window[0]])
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    #tracker has failed, get user input
    if distance(positions[frame_number], pred[frame_number-1])>delta:
        print 'tracker failed', distance(positions[frame_number], pred[frame_number-1])
        k=cv2.waitKey(0)

        if k==115: #if 's' is pressed
            positions[frame_number] = click_list[-1]
            a = click_list[-1][0]-object_size/2, click_list[-1][1]-object_size/2
            track_window = (a[0], a[1], object_size, object_size)
            b = a[0]+object_size, a[1]+object_size
            img2 = cv2.rectangle(img2,a,b,(0,255,0),2)
            roi_hist = get_roi_hist(frame, a[0], a[1],object_size, object_size)
            cv2.imshow('roi', frame[a[1]:a[1]+object_size, a[0]:a[0]+object_size])


    #increment frame_number
    frame_number  +=1
cv2.destroyAllWindows()
cap.release()

#reduce length of the positions list to include only the data collected
positions = positions[:frame_number]

print 'finished tracking'
