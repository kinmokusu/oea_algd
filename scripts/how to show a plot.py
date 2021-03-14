import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection


# split the data into two parts
t1 = [0, 8.69, 7.14, 6.66, 12, 23, 20, 41, 28, 32, 37, 47, 35]#acc
t2 = [0.0968, 0.08851, 0.2467, 0.0839, 0.08148, 0.07072, 0.08303, 0.0961, 0.13593, 0.106, 0.0717]#loss

# sort the data so it makes clean curves
t1.sort()
t2.sort(reverse= True)

# create some y data points
y = range(1,301)

# normalize
t1 = np.array(t1)
t2 = np.array(t2)

y = np.array(y)
y = np.array(y)

fig, ax = plt.subplots()
ax.plot(y, t1)

ax.set(xlabel='Epoch', ylabel='Accuracy',
       title='Accuracy rate')
ax.grid()

fig.savefig("test.png")
plt.show()
