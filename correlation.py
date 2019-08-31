import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bn(a):
    for i in range(len(a)):
        a[i] = (a[i] - np.min(a[i])) / (np.max(a[i]) - np.min(a[i]))
    return a

def correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sx = np.std(x)
    sy = np.std(y)

    s = 0
    for x1, y1 in zip(x, y):
        s += (x1 - x_mean) * (y1 - y_mean)

    r = s /  (sx * sy) / (len(x) - 1)

    return r

lstm = np.load('../uav_test/tmpdata/lstmdata.npy')
cnn = np.load('../uav_test/tmpdata/evaluate_cnn.npy')
gtr = np.load('../uav_test/tmpdata/y_test.npy')
# bn(lstm)
bn(cnn)
bn(gtr)
# lstm = lstm[0]
# cnn = cnn[0]
# gtr = gtr[0]

# lstm = lstm.reshape((-1))
# cnn = cnn.reshape((-1))
# gtr = gtr.reshape((-1))

# print(lstm.shape)
print(cnn.shape)
print(gtr.shape)


meanx = []
meany = []
for cn, gt in zip(cnn, gtr):
    cn = cn.reshape((-1))
    gt = gt.reshape((-1))
    x_p = []
    y_p = []
    for c, g in zip(cn, gt):
        x_p.append(c)
        y_p.append(g)
    meanx.append(x_p)
    meany.append(y_p)

meanx = np.mean(meanx, axis=0)
meany = np.mean(meany, axis=0)

meanx = (meanx - np.min(meanx)) / (np.max(meanx) - np.min(meanx))
meany = (meany - np.min(meany)) / (np.max(meany) - np.min(meany))

r = np.corrcoef(meanx, meany)
r = r[0,1]
y = meanx * r
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# print(stats.mode(meanx)[0][0], stats.mode(meanx)[1])
# print(stats.mode(meany)[0][0], stats.mode(meany)[0][1], stats.mode(meany)[0][2])
# np.argmax

print('correlation coefficient : \n', r)

import collections
import heapq
 
nums = meanx
k = 5
count = collections.Counter(nums)  
out = heapq.nlargest(k, count.keys(), key=count.get)
print(out)

 
nums = meany
k = 5
count = collections.Counter(nums)  
out = heapq.nlargest(k, count.keys(), key=count.get)
print(out)


plt.figure()
palette = plt.get_cmap('Set1')
plt.plot(meanx, y, color=palette(2),label="correlation", linewidth=1, zorder=-1)
plt.scatter(meanx, meany, color=palette(0), label="actual", s=1, zorder=1)
plt.xlabel("prediction")
plt.ylabel("groundTrue")
plt.title('Correlation Coefficient = 0.99')
plt.legend()
plt.savefig("img/relation.png")

# print(a/b)'''
