import numpy as np
import cv2
import time

#from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy import stats

cap = cv2.VideoCapture('example_videos/Juggling4.mov')
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
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
    #print 'in probBallGivenPixel'
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

def removeHeterogeneousHistograms(ball_histograms, threshold = 0.3): #Find elements with high heterogenaity (variance) and remove them.
    print np.shape(ball_histograms[0])
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
    foreground = cv2.bitwise_and(frame_copy, frame_copy, mask = fgmask)
    circles = cv2.HoughCircles(foreground,cv2.HOUGH_GRADIENT,1,20,
                            param1=1,param2=25,minRadius=0,maxRadius=30) #param2 determines sensitivity to circles.
    points = hist_pts(color_copy)
    if frameNumber % 20 == 0:
    	frame_histograms.append(points)
    #cv2.imshow('histogram', curve)
    cv2.imshow('color', color_copy)

    if (not circles is None) and len(circles[0]) < 10:
        circleList = list(map(lambda x: addFrameNumber(x), circles.tolist()[0]))
        output_data.extend(circleList)

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
print 'there are this many histograms'
print len(ball_histograms)
print 'removing heterogeneous...'
deviation, ball_histograms = removeHeterogeneousHistograms(ball_histograms)
print 'Now there are this many:'
print len(ball_histograms)
colors = [np.unravel_index(b.argmax(), b.shape) for b in ball_histograms] #Gets the best predictor for the circle.
print map(displayColor, colors)
print 'col'
print colors
colors = map(lambda c: list(c), colors)
print colors
colors = np.array(colors)
print colors
print np.sqrt((colors[0]-colors[1])**2).sum()
#heirarchical_clustering = linkage(colors, 'complete', lambda u, v: np.sqrt(((u-v)**2).sum())) 
heirarchical_clustering = linkage(colors, 'complete', lambda u, v: min(np.sqrt(((u-v)**2).sum()), 25)) #uses the truncated distance to perform heirarchical clustering
clusters = fcluster(heirarchical_clustering, 24, criterion = 'distance')
print clusters
max_cluster = stats.mode(clusters)[0][0]
print 'max cluster'
print max_cluster
indicies = [i for i, c in enumerate(clusters) if c==max_cluster]
colors = colors[indicies]
colors = np.array([map(displayColor, colors)]).astype(np.uint8)
print colors

colors = cv2.cvtColor(colors, cv2.COLOR_BGR2HSV) #Warning: hue is stored at half its actual value, so that it lies between 0 and 255.
print colors
colors = colors[0]

for c in colors: #The hues are an 'angle of color' and are equivalent modulo 180.
	if c[0]<25:  #This step depends highly on the current process.
		c[0] = c[0] + 180
print colors
deviationH = np.var([c[0] for c in colors])**0.5
deviationS = np.var([c[1] for c in colors])**0.5
deviationV = np.var([c[2] for c in colors])**0.5

print 'hsv deviation'
print deviationH
print deviationS
print deviationV
ball_color = np.mean(colors, axis = 0)

K=2.0 #standard deviation tolerance.
lower_bound = [ball_color[0] - K*deviationH, ball_color[1] - K*deviationS, ball_color[2] - K*deviationV]
upper_bound = [ball_color[0] + K*deviationH, ball_color[1] + K*deviationS, ball_color[2] + K*deviationV]
print 'lower bound:'
print lower_bound
print 'upper_bound:'
print upper_bound
cap.release()
cv2.destroyAllWindows()
