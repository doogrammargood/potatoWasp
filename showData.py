from csv import reader
from matplotlib import pyplot
import numpy as np
from longestIncSubsequence import subsequence

with open('site_swap_data.csv', 'r') as f:
    data = list(reader(f))
x_vals = [d[0] for d in data]
y_vals = [d[1] for d in data]
radii = [d[2] for d in data]
frame = [d[3] for d in data]
#pyplot.plot(frame, y_vals, '.')
#pyplot.show()

subs = subsequence(data, lambda x,y: x[0]<y[0])
subs2 = subsequence(subs, lambda x,y: x[1]<y[1])
pyplot.plot([d[1] for d in subs2], [d[3] for d in subs2], '.')
pyplot.show()
#arc,  = np.polyfit([float(d[1]) for d in subs],[float(d[3]) for d in subs], 2)
#print arc
