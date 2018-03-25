import csv
import matplotlib.pyplot as plt
import numpy as np
with open('./throw_catch_data2.csv', 'r') as f:
    peaks = list(csv.reader(f))


print peaks[0]
peaks = peaks[1:]
frameNumbers = [i[0] for i in peaks[:100]]
ballOne = [i[1] for i in peaks[:100]]
plt.plot(frameNumbers, ballOne)
plt.show()