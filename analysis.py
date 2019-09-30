import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random


# time = [0,1,2,3,4,5,6]
# mae = np.round(np.random.rand(7), 2)


# p = np.percentile(mae, 90)

# print(mae)

mae = np.load('data/mae.npy')
mae = mae[:,:-2]
print(mae.shape)
# print(mae[3])



fig, ax = plt.subplots()


num = 0
for m in mae:
    r = ax.plot(range(mae.shape[1]), m)
    num += 1
    print(num)

#     num+=1
#     plt.plot(range(mae.shape[1]), m, marker=">", color=palette(num), linewidth=1, alpha=0.9, label=num)
 
# Add legend
# plt.legend(loc=2, ncol=2)



ax.set(xlabel='time', ylabel='mae')
ax.grid()
# ax.set_xlim(0, 6)
# ax.set_ylim(0, 1.2)

fig.savefig("img/test.png")





