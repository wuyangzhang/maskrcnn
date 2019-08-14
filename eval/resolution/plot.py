import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load(file):

    vals = defaultdict(list)
    with open(file, 'r') as f:
        for line in f.readlines():
            if 'time' not in line:
                continue
            # time*0.12182331085205078*resolutiuon with *(640, 480)
            line = line.split('*')
            time = float(line[1]) * 1000
            reso = line[-1]
            vals[reso].append(time)

    return vals

vals = load('maskrcnn.log')



font = {'family': 'normal',

        'weight': 'bold',

        'size': 16}

plt.rc('font', **font)

resolution = [(224, 224), (640, 480), (1280, 720), (2048, 1080), (3840, 2160)]
reso_ratio = [1] * len(resolution)
for i in range(1, len(resolution)):
    pre_x, pre_y = resolution[0]
    x, y = resolution[i]
    reso_ratio[i] = x * y / pre_x / pre_y

'''
infero
sns.color_palette("inferno")
[(0.15585, 0.044559, 0.325338),
 (0.397674, 0.083257, 0.433183),
 (0.621685, 0.164184, 0.388781),
 (0.832299, 0.283913, 0.257383),
 (0.961293, 0.488716, 0.084289),
 (0.981173, 0.759135, 0.156863)]
'''

xlabel = vals.keys()
mean = [np.mean(vals[res]) for res in vals]
y_pos = np.arange(len(resolution))
plt.plot(mean, color=(0.15585, 0.044559, 0.325338), marker='x', markersize=10)
plt.xticks(y_pos, resolution)


ax = plt.gca()
#
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', rotation=15)
plt.xlabel('frame resolution')
plt.ylabel('computing latency (ms)', color=(0.15585, 0.044559, 0.325338))
plt.grid(linestyle='dotted')
#
ratio = 5
#plt.gcf().set_size_inches(0.618 *ratio, 1 * ratio)
ax.spines['left'].set_color('#91989F')
ax.spines['bottom'].set_color('#91989F')

ax2 = ax.twinx()
ax2.set_ylabel('resolution increased ratio', color=(0.621685, 0.164184, 0.388781))  # we already handled the x-label with ax1
ax2.plot(reso_ratio, color=(0.621685, 0.164184, 0.388781), marker='o', markersize=12)

plt.tight_layout()
plt.savefig('resolution_latency.pdf')
plt.show()