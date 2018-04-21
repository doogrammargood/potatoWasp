import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from itertools import combinations
with open('./data_from_master.csv', 'r') as f:
    peaks = list(csv.reader(f))
#with open('./xdata_from_master.csv') as f:
#    xpeaks = list(csv.reader(f))

peaks = peaks[10:]
#xpeaks = xpeaks[10:]
def plotBalls(peaks):
    frameNumbers = np.array([int(i[0]) for i in peaks])
    ballOne = np.array([float(i[1]) for i in peaks])
    ballTwo = np.array([float(i[2]) for i in peaks])
    ballThree = np.array([float(i[3]) for i in peaks])

    ballOne = savgol_filter(ballOne, 5, 2)
    ballTwo = savgol_filter(ballTwo, 5, 2)
    ballThree = savgol_filter(ballThree, 5, 2)

    balls = [ballOne, ballTwo, ballThree]

    plt.plot(frameNumbers, ballOne, 'red')
    plt.plot(frameNumbers, ballTwo, 'blue')
    plt.plot(frameNumbers, ballThree, 'green')

    extreme_frames = argrelextrema(ballOne, np.greater, order = 2)[0].tolist()
    extreme_frames += argrelextrema(ballOne, np.less, order = 2)[0].tolist()
    extreme_heights = [ballOne[i] for i in extreme_frames]
    extreme_frames = map(lambda x: x+9, extreme_frames)
    plt.plot(extreme_frames, extreme_heights, 'ro')
    #
    extreme_frames1 = argrelextrema(ballTwo, np.greater, order = 2)[0].tolist()

    extreme_frames1 += argrelextrema(ballTwo, np.less, order = 2)[0].tolist()
    extreme_heights1 = [ballTwo[i] for i in extreme_frames1]
    extreme_frames1 = map(lambda x: x+9, extreme_frames1)
    plt.plot(extreme_frames1, extreme_heights1, 'ro')

    extreme_frames1 = argrelextrema(ballThree, np.greater, order = 2)[0].tolist()

    extreme_frames1 += argrelextrema(ballThree, np.less, order = 2)[0].tolist()
    extreme_heights1 = [ballThree[i] for i in extreme_frames1]
    extreme_frames1 = map(lambda x: x+9, extreme_frames1)
    plt.plot(extreme_frames1, extreme_heights1, 'ro')

    return balls

def getCrossingEvents(peaks): #Since we have locally smoothed the ball trajectories, it doesn't really make sense to try to get the siteswap from local behavior, like local minima.
                              #We should use as much global behavior as possible.
    frameNumbers = np.array([int(i[0]) for i in peaks])
    for i,j in combinations([1,2,3],2):
        print i
        print j


getCrossingEvents(peaks)
#balls = plotBalls(peaks)
#balls = plotBalls(xpeaks)

extrema = []
counter = [0] * 3
siteswap = []
for ball in balls:
    extrema.append(argrelextrema(ball, np.greater, order=7)[0].tolist())
index_of_current_catch = 0
while(True):
     #which ball is currently being considered
    next_frame_seen = 9999
    for current_index, ball in enumerate(extrema): #loop through all the balls to find the one which lands next
        #print current_index
        if len(ball) > 0 and ball[0] < next_frame_seen:
            index_of_current_catch = current_index
            next_frame_seen = ball[0]

    extrema[index_of_current_catch].pop(0)
    siteswap.append(counter[index_of_current_catch]) #add the counter to the siteswap
    counter[index_of_current_catch] = 0 #reset the counter
    counter = map(lambda c: c+1,counter) #increment each counter
    if all( len(ball)==0 for ball in extrema):
        break

siteswap.reverse()
print siteswap

plt.show()
