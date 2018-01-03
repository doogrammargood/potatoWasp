import numpy as np
import skvideo.io

vid = skvideo.io.vread('thomasDietz28.avi')
for frame in vid:
    print(frame.shape)
