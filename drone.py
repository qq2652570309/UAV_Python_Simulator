import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq


class Drone:
    def __init__(self, id=-1, launch_row=-1, launch_col=-1, landing_row=-1, landing_col=-1, trajectors=None, noflyzone=None):
        self.id = id
        self.current_row = launch_row
        self.current_col = launch_col
        self.landing_row = landing_row
        self.landing_col = landing_col
        self.trajectors = trajectors
        self.nfz = noflyzone
        self.path = []
        
    def move(self, currentTime):
        if self.current_row == self.landing_row and self.current_col == self.landing_col:
            return
        
        dist1, available1 = self.location(currentTime, self.current_row, self.current_col-1)
        dist2, available2 = self.location(currentTime, self.current_row, self.current_col+1)
        dist3, available3 = self.location(currentTime, self.current_row-1, self.current_col)
        dist4, available4 = self.location(currentTime, self.current_row+1, self.current_col)
        
        heap = []
        heapq.heappush(heap, (dist1, available1, 0, -1))    # up
        heapq.heappush(heap, (dist2, available2, 0, 1))     # down
        heapq.heappush(heap, (dist3, available3, -1, 0))    # left
        heapq.heappush(heap, (dist4, available4, 1, 0))     # right
        
        directCol = 0
        directRow = 0
        for _, available, row, col in heap:
            if available:
                directCol = col
                directRow = row
                break
        
        self.current_col += directCol
        self.current_row += directRow
        
        if directCol == directRow == 0:
            self.path = []
        
        self.trajectors[currentTime, self.current_row, self.current_col] += 1
        self.path.append((self.current_row, self.current_col))
        # if self.id == 1:
        #     print(currentTime, self.current_row, self.current_col)
        return self.trajectors
    
    
    def location(self, time, row, col):
        if row < 0 or row > 99:
            return 200, False
        if col < 0 or col > 99:
            return 200, False
        
        uav = self.trajectors[time, row, col] # uav conflict
        if uav:
            return 200, False
        
        r1, c1 = self.nfz[0] # no fly zone
        r3, c3 = self.nfz[2]
        if r1 <= row <= r3 and c1 <= col <= c3:
            return 200, False
        
        if (row, col) in self.path: # old path
            return 200, False
        
        dist = self.distance(row, col, self.landing_row, self.landing_col)
        return dist, True 


    def distance(self, row1, col1, row2, col2):
        return abs(row1 - row2) + abs(col1 - col2)
    
    def isArrived(self):
        if self.current_col == self.landing_col and self.current_row == self.landing_row:
            return True
        return False

    