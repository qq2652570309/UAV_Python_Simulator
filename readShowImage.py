from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

im = Image.open("123.png")
np_im = np.array(im)

np_im = np.sum(np_im, axis=-1)

print(np_im.shape)
np_im[np_im>0] = 1

plt.gray()
plt.imshow(np_im)
plt.savefig("img/test.png")

