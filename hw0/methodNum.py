import time as t
import numpy as np

class methodNum:
    
    def __init__(self):
        self.c = 0

    def count(self, *argv):
        self.c += 1
        return self.c

def functimer(func, *argv):
    tick = t.time()
    c = func(*argv)
    print('c: ' +str(c))
    tock = t.time()
    print('elapsed time: ' + str(tock - tick) + ' seconds')
    
counter = methodNum()
counter.count()
counter.count()
counter.count()
functimer(counter.count, 1, 2, 3)

l = [(1,2),(3,4),(5,6),(7,8)]
d = { a:b for a,b in l}
listoftuples = [(key,d[key]) for key in d.keys()]

z = np.zeros((3,3))