from __future__ import division
import numpy as np
import cv2
import csv
import time
import itertools
import math
import copy
from scipy.linalg import block_diag
import scipy.optimize
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy import stats
global frameCount
global hsvColorBounds
global output_data
videoFilename = 'example_videos/423Modified.avi'
cap = cv2.VideoCapture(videoFilename)
fgbg = cv2.createBackgroundSubtractorMOG2()
output_data = []
frameNumber = 0
#output_path = './circle_data.csv'
def hist_pts(im, mask = None): #good explanation here: http://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/#3dhistogram
    hist = cv2.calcHist([im], [0, 1, 2], mask, [32] * 3, [0, 256] * 3)
    l1norm = np.sum(hist)
    hist = hist/l1norm
    return hist

def probBallGivenPixel(circle_hist, frame_hist): #circle_hist is the prob. of pixel color given that the pixel is in the circle.
                                                #frame_hist is the prob. of pixel color.
                                                 #This is Bayes Rule.
    #We use additive smoothing to avoid division by zero.
    normalized_color = np.divide(circle_hist + 1, frame_hist + 32**3)
    norm = sum(normalized_color)
    normalized_color = (1/norm)*normalized_color
    return normalized_color

def filterBallHistograms(f, ball_histograms, m = 1.0): #See Benjamin Bannier's answer here: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    radii = map(f, ball_histograms) #returns a list of the radii which occur.
    med = np.median(np.array(radii))
    d = np.abs(radii - np.median(radii))
    mdev = np.median(d)
    filtered_histograms = [b for b in ball_histograms if abs(f(b) - med) <  m*mdev]
    radius = np.median(map(f ,filtered_histograms))
    return radius, filtered_histograms

def removeHeterogeneousHistograms(ball_histograms, threshold = 1): #Find elements with high heterogenaity (variance) and remove them.
    variances = [np.var(b) for b in ball_histograms]
    average_variance = np.mean(variances)
    toReturn = [b for b in ball_histograms if np.var(b) < average_variance*threshold]
    return (average_variance*threshold)**0.5, toReturn
def addFrameNumber(x):
    x.append(frameNumber)
    return x

def dist(a,b): #returns the square of the 3-dimensional euclidean distance.
    total = 0
    for i in range(3):
        total += (a[i]-b[i])**2
    return total

def truncDist(a,b,delta): #returns the truncated distance between a and b.
    return min(delta,dist(a,b))

def displayColor(color): #Converts from the 'bins' 8 bit representation back into 256.
    new_color = list(color)
    return map(lambda c: 8*c+4, new_color)


frame_histograms = []
ball_histograms = []
while(True):
    frameNumber += 1
    ret, frame = cap.read()
    if frame is None:

        print('no frame')
        break
    color_copy = frame
    color_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_copy = color_copy[...,0] #returns just the hue.
    color_copy = frame
    fgmask = fgbg.apply(frame_copy)
    #fgmask = smoothNoise(fgmask)
    foreground = cv2.bitwise_and(frame_copy, frame_copy, mask = fgmask)
    circles = cv2.HoughCircles(foreground,cv2.HOUGH_GRADIENT,1,20,
                            param1=2,param2=25,minRadius=0,maxRadius=20) #param2 determines sensitivity to circles.
    points = hist_pts(color_copy)
    if frameNumber % 20 == 0:
        frame_histograms.append(points)
    #cv2.imshow('histogram', curve)
    cv2.imshow('color', color_copy)
    #print len(circles[0])
    if (not circles is None) and len(circles[0]) < 5:
        circleList = list(map(lambda x: addFrameNumber(x), circles.tolist()[0]))
        #output_data.extend(circleList)

        for circ in circles[0,:]:
            # draw the outer circle
            cv2.circle(foreground,(circ[0],circ[1]),circ[2],(255,255,255),2)
            # draw the center of the circle
            cv2.circle(foreground,(circ[0],circ[1]),2,(0,0,255),3)

            black = np.zeros(frame_copy.shape, dtype=np.uint8)
            circle_mask = cv2.circle(black, (circ[0],circ[1]), int(circ[2]*0.9), 255, -1) #we shrink the radius to 0.9 of its original size to avoid weird stuff on the edges
            #isolated_circle = cv2.bitwise_and(color_copy, color_copy, mask = circle_mask)
            #print circ
            #cv2.imshow('mask', circle_mask)
            ball_histograms.append( (circ , hist_pts(color_copy, circle_mask) ))
            #cv2.imshow('maskedHistogram', circle_hist)
            #cv2.imshow('maskedCircle', isolated_circle)

    cv2.imshow('frame',foreground)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
frame_average_histogram = np.average(np.array(frame_histograms), axis = 0) #averages the histograms over the frames.
typical_radius, ball_histograms = filterBallHistograms(lambda b: b[0].item(2), ball_histograms, 1.5)#filter histograms to reject radius outliers.

ball_histograms = [(b[0], probBallGivenPixel(b[1],frame_average_histogram)) for b in ball_histograms]
ball_histograms = [b[1] for b in ball_histograms] #drop data about the circles
deviation, ball_histograms = removeHeterogeneousHistograms(ball_histograms)
colors = [np.unravel_index(b.argmax(), b.shape) for b in ball_histograms] #Gets the best predictor for the circle.
colors = map(lambda c: list(c), colors)
colors = np.array(colors)
#heirarchical_clustering = linkage(colors, 'complete', lambda u, v: np.sqrt(((u-v)**2).sum())) 
heirarchical_clustering = linkage(colors, 'complete', lambda u, v: min(np.sqrt(((u-v)**2).sum()), 25)) #uses the truncated distance to perform heirarchical clustering
clusters = fcluster(heirarchical_clustering, 24, criterion = 'distance')
print clusters
max_cluster = stats.mode(clusters)[0][0]
print max_cluster
indicies = [i for i, c in enumerate(clusters) if c==max_cluster]
colors = colors[indicies]
colors = np.array([map(displayColor, colors)]).astype(np.uint8)


colors = cv2.cvtColor(colors, cv2.COLOR_BGR2HSV) #Warning: hue is stored at half its actual value, so that it lies between 0 and 255.
colors = colors[0]

#for c in colors: #The hues are an 'angle of color' and are equivalent modulo 180.
#    if c[0]<25:  #This step depends highly on the current process.
#        c[0] = c[0] + 180
deviationH = max(np.var([c[0] for c in colors])**0.5, 3)
deviationS = np.var([c[1] for c in colors])**0.5
deviationV = np.var([c[2] for c in colors])**0.5

ball_color = np.mean(colors, axis = 0)

K=3 #standard deviation tolerance.
lower_bound = [ball_color[0] - K*deviationH, max(ball_color[1] - K*deviationS, 0), max(ball_color[2] - K*deviationV, 0)]
upper_bound = [ball_color[0] + K*deviationH, min(ball_color[1] + K*deviationS, 255), min(ball_color[2] + K*deviationV, 255)]
lower_bound = map(lambda l: int(l), lower_bound)
upper_bound = map(lambda u: int(u), upper_bound)
print [ball_color[0]*2, ball_color[1]/255.0, ball_color[2]/255.0]
print lower_bound
print upper_bound


cap.release()
cv2.destroyAllWindows()












'''-------'''
numBalls = 3 #This needs to be automated

ballPositionMarkerColors = ((200,0,0), (255,200,200), (0,200,0), (0,0,200))
ballTrajectoryMarkerColors = ((200,55,55), (255,200,200), (55,255,55), (55,55,255))




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
def smoothNoise(frame, kernel = None):
    if kernel is None:
        kernel = np.ones((3,3)).astype(np.uint8) #This value should be set in the preprocessing step
    frame = cv2.erode(frame, kernel)
    frame = cv2.dilate(frame, kernel)

    return frame

# Performs all necessary pre-processing steps before the color thresholding
def processForThresholding(frame):
    blurredframe = blur(frame)

    #Subtract background (makes isolation of balls more effective, in combination with thresholding)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(blurredframe)
    #fgmask = smoothNoise(fgmask, kernel = np.ones((2,2)).astype(np.uint8))
    cv2.imshow('fgmask', fgmask)
    frame = cv2.bitwise_and(blurredframe,blurredframe, mask = fgmask)

    # Convert to HSV color space
    hsvBlurredFrame = cv2.cvtColor(blurredframe, cv2.COLOR_BGR2HSV)
    #return frame
    return hsvBlurredFrame




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

def getStateFromFilter(kalman): #gets the current state from the filter, dispite the names.
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

def findBallsInImage(image, kalman, isFirstFrame=False, frameCount=0, failcount=0):
    numBallsToFind = numBalls
    cutoff = 150

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
        currentBallCenters, _ = getStateFromFilter(kalman)
        predictedBallCenters, predictedBallVelocities = getPredictedStateFromFilter(kalman)

        #The Hungarian Algorithm:
        cost_matrix = [ [ [predictedBallCenters[i], centers[j]] for j in range(len(centers))] for i in range(len(predictedBallCenters)) ]
        cost_matrix = map(lambda row: map(lambda pair: distance2D(pair[0], pair[1]),row), cost_matrix)
        _, column_permutation = scipy.optimize.linear_sum_assignment(np.array(cost_matrix)) #an array whose ith element is the index of centers, which is paired to predictedBallCenters[i] in the optimal paring.
        new_measurements = []
        
        for prediction_index, center_index in enumerate(column_permutation):
            if cost_matrix[prediction_index][center_index] < cutoff or frameCount < 20 or failcount > 0:
                #print cost_matrix[prediction_index][center_index]
                #print frameCount
                new_measurements.append(centers[center_index])
                failcount = 0
                print 'reset fail count'
            else:
                print 'k means failure'
                print currentBallCenters[prediction_index]
                failcount +=1
                #We assume the missing measurement is halfway between where it should be an where it was.
                fabricated_measurement = [(predictedBallCenters[prediction_index][0]+currentBallCenters[prediction_index][0])/2.0, (predictedBallCenters[prediction_index][1]+currentBallCenters[prediction_index][1])/2.0,]
                new_measurements.append(fabricated_measurement) #assume the ball stays where it is, if no accurate reading was found.
            current_position, _ = getStateFromFilter(kalman)
            temp_output_data = [i[1] for i in current_position]
            temp_output_datax = [i[0] for i in current_position]

        new_measurements = sum(new_measurements, []) #flattens the array.
        if isFirstFrame:
            kalman.statePost = np.array([addZeros(new_measurements)],np.float32).transpose()
        else:
            kalman.correct(np.array([new_measurements],np.float32).transpose())
        output_data.append(temp_output_data)
        output_datax.append(temp_output_datax)
        #if len(min_matches) != numBalls:
        #    print 'matching failed'
        return failcount
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
#hsvColorBounds = np.array([15,0,209],np.uint8), np.array([45,173,255],np.uint8)
hsvColorBounds = (np.array(lower_bound, np.uint8),np.array(upper_bound, np.uint8)) #bgr

output_data = []
output_datax = [] #I now realize I should also store the x-values.
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
    kalman.processNoiseCov = 2*np.array(block_diag(*noiseArgs), np.float32)
    #kalman.processNoiseCov = np.array(np.identity(4*balls), np.float32)
    kalman.measurementNoiseCov = 3*np.array(np.identity(2*balls), np.float32)
    return kalman

def main():
    kalman = createFilter(numBalls)
    showBallDetectionData = False
    # Get a camera input source
    cap = cv2.VideoCapture(videoFilename)

    boolFirstFrame=True
    frameCount = 0 
    failcount = 0
    while(cap.isOpened()):
        frameCount+=1
        frame = getFrame(cap)
        if frame is None:
            break
        # Makes a copy before any changes occur
        frameCopy = frame.copy()

        frame = processForThresholding(frame)

        # Find locations of ball(s)
        colorBounds = hsvColorBounds
        thresholdImage = cv2.inRange(frame, colorBounds[0], colorBounds[1])
        thresholdImage = smoothNoise(thresholdImage)
        cv2.imshow('thresholdImage', thresholdImage)
        # Open to remove small elements/noise
        # We'll use ballIndices to only select from a subset of the balls to pair
        ballCenters, ballVelocities = getPredictedStateFromFilter(kalman)

        # Find the points in the image where this is true, and get the matches that pair
        # these points to the balls that we're already tracking
        if boolFirstFrame:
            findBallsInImage(thresholdImage, kalman, isFirstFrame = boolFirstFrame, frameCount = frameCount)
            boolFirstFrame = False
        else:
            failcount = findBallsInImage(thresholdImage, kalman, isFirstFrame = False, frameCount = frameCount, failcount = failcount)

        frameCopy = drawBallsAndTrajectory(frameCopy, kalman)

        cv2.imshow('Image with Estimated Ball Center', frameCopy)

        k = cv2.waitKey(int(1000.0 / 30)) & 0xFF

        if k == 27:
            # User hit ESC
            break
    output_path = './data_from_master.csv'
    global output_data
    with open(output_path, 'w') as csvfile:
        fieldnames = ['frame', 'ball1Position', 'ball2Position', 'ball3Position']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        output_data = map(lambda x: [x[0]] + x[1], zip(range(len(output_data)),output_data)) #adds the frame numbers
        for data in output_data:
            #x, y, r, f = data[0], data[1], data[2], data[3]
            #print data
            writer.writerow({'frame': data[0], 'ball1Position': data[1], 'ball2Position': data[2], 'ball3Position': data[3]})
    output_path = './xdata_from_master.csv'
    global output_datax
    with open(output_path, 'w') as csvfile:
        fieldnames = ['frame', 'ball1Position', 'ball2Position', 'ball3Position']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        output_datax = map(lambda x: [x[0]] + x[1], zip(range(len(output_datax)),output_datax)) #adds the frame numbers
        for data in output_datax:
            #x, y, r, f = data[0], data[1], data[2], data[3]
            #print data
            writer.writerow({'frame': data[0], 'ball1Position': data[1], 'ball2Position': data[2], 'ball3Position': data[3]})
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
