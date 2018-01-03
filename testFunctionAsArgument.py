def evalFunction(list, comparison = None):
    if comparison is None:
        def comparison(x,y):
            return (x < y)
    if comparison(list[0],list[1]):
        return True
    else: return False

print evalFunction([[1,3],[7,1]], lambda x, y: x[0] < y[0])
