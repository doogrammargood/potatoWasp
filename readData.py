import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
with open('./dataWithHands.csv', 'r') as f:
    peaks = list(csv.reader(f))


peaks = peaks[30:]
frameNumbers = [int(i[0]) for i in peaks]
ballOne = [float(i[1]) for i in peaks]
ballTwo = [float(i[2]) for i in peaks]
ballThree = [float(i[3]) for i in peaks]
ballFour = [float(i[4]) for i in peaks]

ballOne = savgol_filter(ballOne, 21, 2)
ballTwo = savgol_filter(ballTwo, 21, 2)
ballThree = savgol_filter(ballThree, 21, 2)
ballFour = savgol_filter(ballFour, 21, 2)

balls = [ballOne, ballTwo, ballThree, ballFour]

plt.plot(frameNumbers, ballOne, 'red')
plt.plot(frameNumbers, ballTwo, 'blue')
plt.plot(frameNumbers, ballThree, 'green')
plt.plot(frameNumbers, ballFour, 'purple')

extreme_frames = argrelextrema(ballOne, np.greater, order = 7)[0].tolist()
extreme_heights = [ballOne[i] for i in extreme_frames]
extreme_frames = map(lambda x: x+29, extreme_frames)
plt.plot(extreme_frames, extreme_heights, 'ro')
#
extreme_frames1 = argrelextrema(ballTwo, np.greater, order = 7)[0].tolist()
extreme_heights1 = [ballTwo[i] for i in extreme_frames1]
extreme_frames1 = map(lambda x: x+29, extreme_frames1)
plt.plot(extreme_frames1, extreme_heights1, 'ro')

extreme_frames1 = argrelextrema(ballThree, np.greater, order = 7)[0].tolist()
extreme_heights1 = [ballThree[i] for i in extreme_frames1]
extreme_frames1 = map(lambda x: x+29, extreme_frames1)
plt.plot(extreme_frames1, extreme_heights1, 'ro')

extreme_frames1 = argrelextrema(ballFour, np.greater, order = 7)[0].tolist()
extreme_heights1 = [ballFour[i] for i in extreme_frames1]
extreme_frames1 = map(lambda x: x+29, extreme_frames1)
plt.plot(extreme_frames1, extreme_heights1, 'ro')

extrema = []
counter = [0] * 4
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
