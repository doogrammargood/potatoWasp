import numpy as np

z = [1,3,4]
x = [3,5,3]

def clusterStatus(run): #Determines whether the cluster is complete or a false alarm.
    nonZero = len(filter(lambda x: x is not None, run))
    if nonZero >= 3: #If there are at least 3 non-None frames in the return
        maxY = max( map(lambda x: x[1] if x else None, run) ) #The largest y value.
        if maxY == run[2][1]: #And frame 2 has the largest y value,
            return 1
    elif len(run) >= 4 and run[-1]==run[-2]==None:
        return 2 #False alarm, the cluster should be removed.
    return 0 #This means the cluster is still indeterminate.
print clusterStatus([None,None,None,None])

def examineClusters(currentClusters, foundClusters):
    newClusters = [c for c in currentClusters if clusterStatus(c) == 1]
    newClusters = map(lambda x: (x[2][0], x[2][1], frameNumber -2), newClusters) #puts the newclusters in the form(x, y, frame)
    currentClusters = [c for c in currentClusters if clusterStatus(c) == 0] #gets rid of unused clusters
    foundClusters.extend(newClusters)
run = None
if run and 7 = run[2][1]
# currentClusters =[[None, None, None, None]]
# foundClusters = []
# examineClusters(currentClusters, foundClusters)
# print currentClusters
# print foundClusters
