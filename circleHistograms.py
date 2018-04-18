import numpy as np
import cv2
import time
cap = cv2.VideoCapture('juggling.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
output_data = []
frameNumber = 0
#output_path = './circle_data.csv'
bins = np.array([i+1 for i in range(255)]) #exclude 0.
bins = bins.reshape(255,1)
frame_histograms = [] #an array containing histograms for each frame.
ball_histograms = [] #an array containing tuples- (circle, histogram in that circle), where circle is the tuple (x,y,r)
#bins = np.arange(256).reshape(256,1)
def hist_pts(im): # from https://github.com/opencv/opencv/blob/master/samples/python/hist.py
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
         color = [ (255,0,0),(0,255,0),(0,0,255) ]
    toReturn = []
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
        hist_item = np.delete(hist_item,(0,0)) #remove 0 element, which should be black
        #cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        #hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist_item)))
        l1norm = sum([p[1] for p in pts])
        pts =[x[1]*1000.0/l1norm for x in pts] #normalizes pts in the L1 norm. There should be a way to do this with opencv2 but idk.
        #print pts
        toReturn.append(pts)
    return toReturn
    #    cv2.polylines(h,[pts],False,col)
    #y=np.flipud(h)
    #return y

def probBallGivenPixel(circle_hist, frame_hist): #circle_hist is the prob. of pixel color given that the pixel is in the circle.
                                                #frame_hist is the prob. of pixel color.
                                                 #This is Bayes Rule.
    #We use additive smoothing to avoid division by zero.
    toReturn = []
    #print circle_hist
    for c in range(3):
        normalized_color = [ (circle_hist[c][i] + 1) / (frame_hist[c][i] + 1) for i in range(len(frame_hist[0]))] #We leave out the factor of Pr(pixel in circle). It is a constant.
        norm = sum(normalized_color)
        normalized_color = [ p/norm for p in normalized_color] #normalizes the resulting distribution.
        toReturn.append(normalized_color)
    return toReturn

def filterBallHistograms(f, ball_histograms, m = 1.0): #See Benjamin Bannier's answer here: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    radii = map(f, ball_histograms) #returns a list of the radii which occur.
    med = np.median(np.array(radii))
    d = np.abs(radii - np.median(radii))
    mdev = np.median(d)
    filtered_histograms = [b for b in ball_histograms if abs(f(b) - med) <  m*mdev]
    return filtered_histograms

def removeHeterogeneousHistograms(ball_histograms, cutoff): #Find elements with high heterogenaity and remove them.
    variance_array = []
    #print ball_histograms[0]
    for col in range(3):
        color_distributions = [np.array(hist[1][col]) for hist in ball_histograms]
        variance_array.append([np.var(dist) for dist in color_distributions])
    total_variance = variance_array[0]+variance_array[1]+variance_array[2]
    average_variance = sum(total_variance) / len(ball_histograms)
    print average_variance
    print "avg var"
    b = ball_histograms[0]
    print np.var(b[1][0]) + np.var(b[1][1]) + np.var(b[1][2])
    toReturn = [b for b in ball_histograms if np.var(b[1][0]) + np.var(b[1][1]) + np.var(b[1][2]) < average_variance*.75]
    #print toReturn
    variance = average_variance*.5
    return variance, toReturn
def addFrameNumber(x):
    #print frameNumber
    x.append(frameNumber)
    return x

def dist(a,b): #returns the square of the 3-dimensional euclidean distance.
    total = 0
    for i in range(3):
        total += (a[i]-b[i])**2
    return total

while(True):
    frameNumber += 1
    ret, frame = cap.read()
    if frame is None:

        print('no frame')
        break
    color_copy = frame
    #color_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame_copy)
    foreground = cv2.bitwise_and(frame_copy, frame_copy, mask = fgmask)
    circles = cv2.HoughCircles(foreground,cv2.HOUGH_GRADIENT,1,20,
                            param1=20,param2=30,minRadius=5,maxRadius=40)
    #hist = cv2.calcHist([color_copy],[0],None,[256],[0,256])
    points = hist_pts(color_copy)
    #print points
    frame_histograms.append(points)
    #cv2.imshow('histogram', curve)
    #cv2.imshow('color', color_copy)
    #print color_copy
    if not circles is None:
        circleList = list(map(lambda x: addFrameNumber(x), circles.tolist()[0]))
        output_data.extend(circleList)
        #print circleList
        for circ in circles[0,:]:
            # draw the outer circle
            cv2.circle(foreground,(circ[0],circ[1]),circ[2],(255,255,255),2)
            # draw the center of the circle
            cv2.circle(foreground,(circ[0],circ[1]),2,(0,0,255),3)

            black = np.zeros(frame_copy.shape, dtype=np.uint8)
            circle_mask = cv2.circle(black, (circ[0],circ[1]), int(circ[2]*0.9), 255, -1) #we shrink the radius to 0.9 of its original size to avoid weird stuff on the edges
            isolated_circle = cv2.bitwise_and(color_copy, color_copy, mask = circle_mask)
            ball_histograms.append( (circ , hist_pts(isolated_circle) ))
            #cv2.imshow('maskedHistogram', circle_hist)
            #cv2.imshow('maskedCircle', isolated_circle)

    #cv2.imshow('frame',foreground)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
frame_average_histogram = np.average(np.array(frame_histograms), axis = 0) #averages the histograms over the frames.
ball_histograms = filterBallHistograms(lambda b: b[0].item(2), ball_histograms, 1.5)#filter histograms to reject radius outliers.
#ball_histograms = filterBallHistograms(lambda b: b[0].item(2), ball_histograms, 1.5)#remove histograms with high variances.
ball_histograms = [(b[0], probBallGivenPixel(b[1],frame_average_histogram)) for b in ball_histograms]
#print ball_histograms[0]
variance, ball_histograms = removeHeterogeneousHistograms(ball_histograms, 0)
ball_histograms = [b[1] for b in ball_histograms] #drop data about the circles
colors = []
for i in range(len(ball_histograms)):
    colors.append([np.argmax(ball_histograms[i][0]), np.argmax(ball_histograms[i][1]), np.argmax(ball_histograms[i][2])])
colors = np.array(colors, dtype = np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print len(colors)
numClusters = 3
ret,label,center=cv2.kmeans(colors,numClusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
label = np.transpose(label)[0].tolist() #Now label is an array for all the points.
dev_array = [0] * numClusters
for index, colorPoint in enumerate(colors):
    dev_array[label[index]] += dist(colorPoint, center[label[index]])
dev_array[0]= dev_array[0]**(0.5)/label.count(0)
dev_array[1]= dev_array[1]**(0.5)/label.count(1)
dev_array[2]= dev_array[2]**(0.5)/label.count(2)
print ret
print label
print center
print dev_array

#TODO: cluster with varying numbers until all clusters have a similar spread.
#Drop the last cluster (it's skin tone)
#use var
#print ball_histograms
#Perform k-means clustering on these histograms. See https://arxiv.org/pdf/1303.6001.pdf

#cv2.imshow('histogram', frame_average_histogram)

cap.release()
cv2.destroyAllWindows()

#write data
# import csv
# with open(output_path, 'w') as csvfile:
#     fieldnames = ['x_position', 'y_position', 'frame', 'radius']
#     writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
#     writer.writeheader()
#     for data in output_data:
#         x, y, r, f = data[0], data[1], data[2], data[3]
#         writer.writerow({'x_position': x, 'y_position': y, 'frame': f, 'radius': r})
