import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bn(a):
    for i in range(len(a)):
        a[i] = (a[i] - np.min(a[i])) / (np.max(a[i]) - np.min(a[i]))
    return a


prediction = np.load('../prediction.npy')
label = np.load('../label.npy')
# v1 model data
# prediction = np.load('tmpdata/evaluate_cnn.npy')
# label = np.load('tmpdata/y_test.npy')
# prediction = prediction[:40]
# label = label[:40]


print(prediction.shape)
print(label.shape)


x = prediction.reshape((-1))
y_true = label.reshape((-1))
r = np.corrcoef(x, y_true)
r = r[0,1]
print('correlation coefficient : \n', r)
y_pre = x*r



plt.figure()
palette = plt.get_cmap('Set1')
plt.plot(x, y_pre, color='blue',label="correlation", linewidth=1, zorder=1)
plt.scatter(x, y_true, color=palette(0), label="actual", s=0.1, zorder=-1)
plt.xlabel("prediction")
plt.ylabel("label")
plt.title('Correlation Coefficient = {0}'.format(np.round(r,2)))
plt.legend()
plt.savefig("img/relation_15_30epics.png")
# plt.savefig("img/v1_relation.png")

# print(a/b)'''
