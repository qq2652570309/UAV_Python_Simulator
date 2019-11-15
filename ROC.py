import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
# from sklearn import metrics

# y = np.array([1, 1, 2, 2])
# scores = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

# y = np.array([1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2])
# scores = np.array([0.98, 0.98, 0.96, 0.94, 0.93, 0.92, 0.91, 0.89, 0.89, 0.89, 0.77, 0.72])
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)


# print("fpr: ", fpr)
# print("tpr: ", tpr)
# print("thresholds: ", thresholds)

# for fp, tp in zip(fpr, tpr):
#     print("({0}, {1})".format(fp, tp))



# attact is true,  Positive 'po'
# normal is false, negative 'ne'

x = []
y = []
l = ['po', 'po', 'ne', 'po', 'po', 'po', 'ne', 'po', 'po', 'po', 'ne', 'ne']
f = [0.98, 0.97, 0.96, 0.96, 0.93, 0.92, 0.91, 0.89, 0.89, 0.89, 0.77, 0.72]
p = 8
n = 4

fp = 0
tp = 0
r = []
f_prev = -11000
i = 0
while i < len(l):
    if f[i] != f_prev:
        r.append((fp/n, tp/p))
        x.append(fp/n)
        y.append(tp/p)
        f_prev = f[i]
    if l[i] == 'po':
        tp += 1
    else:
        fp += 1
    i += 1
r.append((fp/n, tp/p))
x.append(fp/n)
y.append(tp/p)

print(r)

plt.plot(x, y)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()
plt.savefig("ROC.png")