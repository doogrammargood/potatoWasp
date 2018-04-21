#Modified from Nate Guy's Ball Tracking
#Using https://github.com/victordibia/handtracking for hand tracking
from __future__ import division
import numpy as np
import cv2
import itertools
import math
import time
import csv
import copy
from scipy.linalg import block_diag
import scipy.optimize
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy import stats


numBalls = 3 #This needs to be automated

ballPositionMarkerColors = ((200,0,0), (255,200,200), (0,200,0), (0,0,200))
ballTrajectoryMarkerColors = ((200,55,55), (255,200,200), (55,255,55), (55,55,255))

videoFilename = 'example_videos/Juggling4.mov'


# Get a frame from the current video source
def getFrame(cap):
    _, frame = cap.read()
    # frame = cv2.imread('greens.png')
    return frame

# Applies a median blur to an image to smooth out noise
def blur(image):
    blurredImage = cv2.medianBlur(image, 5)
    return blurredImage



def rejectOutlierPoints(points, m=2):
    if len(points[0]) == 0:
        return []
    else:
        # Get means and SDs
        meanX = np.mean([x for (x, y) in points[0]], axis=0)
        stdX = np.std([x for (x, y) in points[0]], axis=0)
        meanY = np.mean([y for (x, y) in points[0]], axis=0)
        stdY = np.std([y for (x, y) in points[0]], axis=0)

        nonOutliers = [(x,y) for x, y in points[0] if (abs(x - meanX) < stdX*m) and (abs(y - meanY) < stdY*m)]

        return np.array([nonOutliers])


# Performs all necessary pre-processing steps before the color thresholding
def processForThresholding(frame):
    #blurredFrame = blur(frame)

    #Subtract background (makes isolation of balls more effective, in combination with thresholding)
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    #fgmask = fgbg.apply(frame)
    #frame = cv2.bitwise_and(frame,frame, mask = fgmask)

    # Convert to HSV color space
    hsvBlurredFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #return frame
    return hsvBlurredFrame

def smoothNoise(frame):
    kernel = np.ones((6,6)).astype(np.uint8)
    frame = cv2.erode(frame, kernel)
    frame = cv2.dilate(frame, kernel)

    return frame

def distance2D(p, q):
    dist = math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
    return dist

def getPredictedStateFromFilter(kalman): #predicts the state in the next frame.
    predicted_state = kalman.predict()
    predicted_state = predicted_state.transpose()[0]
    ball_x_positions = [a[1] for a in enumerate(predicted_state) if a[0]%4==0]
    ball_y_positions = [a[1] for a in enumerate(predicted_state) if a[0]%4==1]
    ball_x_velocities = [a[1] for a in enumerate(predicted_state) if a[0]%4==2]
    ball_y_velocities = [a[1] for a in enumerate(predicted_state) if a[0]%4==3]
    predicted_ball_centers = zip(ball_x_positions, ball_y_positions)
    predicted_ball_velocities = zip(ball_x_velocities, ball_y_velocities)
    return predicted_ball_centers, predicted_ball_velocities

def getStateFromFilter(kalman): #gets the current state from the filter
    current_state = kalman.statePost
    current_state = current_state.transpose()[0]
    ball_x_positions = [a[1] for a in enumerate(current_state) if a[0]%4==0]
    ball_y_positions = [a[1] for a in enumerate(current_state) if a[0]%4==1]
    ball_x_velocities = [a[1] for a in enumerate(current_state) if a[0]%4==2]
    ball_y_velocities = [a[1] for a in enumerate(current_state) if a[0]%4==3]
    predicted_ball_centers = zip(ball_x_positions, ball_y_positions)
    predicted_ball_velocities = zip(ball_x_velocities, ball_y_velocities)
    return predicted_ball_centers, predicted_ball_velocities

def addZeros(measurementArray):
    length = len(measurementArray)
    measurementArray = np.array(measurementArray)
    measurementArray = measurementArray.reshape(2,int(length/2), order ='F')
    measurementArray = np.insert(measurementArray, 2, np.zeros(int(length/2)),0)
    measurementArray = np.insert(measurementArray, 3, np.zeros(int(length/2)),0)
    measurementArray=measurementArray.reshape(2*length, order = 'F')
    return measurementArray

def findBallsInImage(image, kalman, isFirstFrame=False, frameCount=0):
    numBallsToFind = numBalls
    cutoff = 100

    # Get a list of all of the non-blank points in the image
    points = np.dstack(np.where(image>0)).astype(np.float32)

    # points = rejectOutlierPoints(points) # removes points that are physically far away from the rest. Seems pointless and time consuming.
    if len(points) == 0:
        return []

    if len(points[0]) >= numBallsToFind:

        # Break into clusters using k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(points, numBallsToFind, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


        centers = centers.tolist()

        # Centers come to us in (y, x) order. This is annoying, so we'll switch it to (x, y) order.
        centers = [[x,y] for [y,x] in centers]

        pairings = []
        temp_output_data = []
        predictedBallCenters, predictedBallVelocities = getPredictedStateFromFilter(kalman)

        #The Hungarian Algorithm:
        cost_matrix = [ [ [predictedBallCenters[i], centers[j]] for j in range(len(centers))] for i in range(len(predictedBallCenters)) ]
        cost_matrix = map(lambda row: map(lambda pair: distance2D(pair[0], pair[1]),row), cost_matrix)
        _, column_permutation = scipy.optimize.linear_sum_assignment(np.array(cost_matrix)) #an array whose ith element is the index of centers, which is paired to predictedBallCenters[i] in the optimal paring.
        new_measurements = []
        
        for prediction_index, center_index in enumerate(column_permutation):
            if cost_matrix[prediction_index][center_index] < cutoff or frameCount < 20:
                #print cost_matrix[prediction_index][center_index]
                #print frameCount
                new_measurements.append(centers[center_index])
            else:
                #print 'k means failure'
                #print predictedBallCenters[prediction_index]
                new_measurements.append(list(predictedBallCenters[prediction_index])) #assume the ball stays where it is, if no accurate reading was found.
            current_position, _ = getStateFromFilter(kalman)
            temp_output_data = [i[1] for i in current_position]

        new_measurements = sum(new_measurements, []) #flattens the array.
        if isFirstFrame:
            kalman.statePost = np.array([addZeros(new_measurements)],np.float32).transpose()
        else:
            kalman.correct(np.array([new_measurements],np.float32).transpose())
        output_data.append(temp_output_data)
        #if len(min_matches) != numBalls:
        #    print 'matching failed'
        return []
    else:
        return []

def drawBallsAndTrajectory(frameCopy, kalman):
    # print len(matches)
    ballCenters, ballVelocities = getPredictedStateFromFilter(kalman)

    matchedIndices = []
    for i in range(len(ballCenters)):
        centerX = ballCenters[i][0]
        centerY = ballCenters[i][1]
        velocityX = ballVelocities[i][0]
        velocityY = ballVelocities[i][1]
        cv2.circle(frameCopy, (centerX, centerY), 6, ballPositionMarkerColors[i], thickness=6)

    return frameCopy

global hsvColorBounds
hsvColorBounds={}
hsvColorBounds['red'] = (np.array([166, 84, 131], np.uint8),np.array([186,  239, 244], np.uint8)) #bgr
global output_data
output_data = []
#hsvColorBounds['red'] = (np.array([0, 153, 127],np.uint8), np.array([4, 230, 179],np.uint8))
def createFilter(balls):
    kalman = cv2.KalmanFilter(4*balls,2*balls) # state is (x,y,v_x,v_y, )^t, sensors are (x,y, )
    H = [[1,0,0,0],
         [0,1,0,0]]
    hargs = [H] * balls
    kalman.measurementMatrix = np.array(block_diag(*hargs), np.float32)
    #kalman.measurementMatrix = np.array(map(lambda x: __g(x), range(2*balls)), np.float32)
    A = [[1,0,1,0],
         [0,1,0,1],
         [0,0,1,0],
         [0,0,0,1]]
    arguments = [A] * balls #n copies of the matrix amove
    kalman.transitionMatrix = np.array(block_diag(*arguments), np.float32)

    N = [[1,0,0,0],
         [0,5,0,0],
         [0,0,1,0],
         [0,0,0,5]]
    noiseArgs = [N] * balls
    #kalman.processNoiseCov = np.array(block_diag(*noiseArgs), np.float32)
    kalman.processNoiseCov = np.array(np.identity(4*balls), np.float32)
    kalman.measurementNoiseCov = np.array(np.identity(2*balls), np.float32)
    return kalman

def main():
    kalman = createFilter(numBalls)
    showBallDetectionData = False
    global frameCount
    # Get a camera input source
    cap = cv2.VideoCapture(videoFilename)

    boolFirstFrame=True
    frameCount = 0 
    while(cap.isOpened()):
        frameCount+=1
        frame = getFrame(cap)
        if frame is None:
            break
        # Makes a copy before any changes occur
        frameCopy = frame.copy()

        frame = processForThresholding(frame)

        for color, ballIndices in [('red',[0,1,2,3])]:
            # Find locations of ball(s)
            colorBounds = hsvColorBounds[color]
            thresholdImage = cv2.inRange(frame, colorBounds[0], colorBounds[1])
            cv2.imshow('thresholdImage', thresholdImage)
            # Open to remove small elements/noise
            thresholdImage = smoothNoise(thresholdImage)
            # We'll use ballIndices to only select from a subset of the balls to pair
            ballCenters, ballVelocities = getPredictedStateFromFilter(kalman)

            # Find the points in the image where this is true, and get the matches that pair
            # these points to the balls that we're already tracking
            if boolFirstFrame:
                findBallsInImage(thresholdImage, kalman, isFirstFrame = boolFirstFrame, frameCount = frameCount)
                boolFirstFrame = False
            else:
                findBallsInImage(thresholdImage, kalman, isFirstFrame = False, frameCount = frameCount)

            frameCopy = drawBallsAndTrajectory(frameCopy, kalman)

        cv2.imshow('Image with Estimated Ball Center', frameCopy)

        k = cv2.waitKey(int(1000.0 / 30)) & 0xFF

        if k == 27:
            # User hit ESC
            break
    output_path = './dataWithHands.csv'

    # with open(output_path, 'w') as csvfile:
    #     fieldnames = ['frame', 'ball1Error', 'ball2Error', 'ball3Error', 'ball4Error']
    #     writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    #     writer.writeheader()
    #     global output_data
    #     output_data = map(lambda x: [x[0]] + x[1], zip(range(len(output_data)),output_data))
    #     for data in output_data:
    #         #x, y, r, f = data[0], data[1], data[2], data[3]
    #         print data
    #         writer.writerow({'frame': data[0], 'ball1Error': data[1], 'ball2Error': data[2], 'ball3Error': data[3], 'ball4Error': data[4]})
    #
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
