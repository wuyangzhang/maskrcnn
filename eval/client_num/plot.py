import glob
from collections import defaultdict
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

client = defaultdict(list)

for log in glob.glob('resnet_*.log'):
    with open(log, 'r') as f:
        num = int(log.split('.')[0].split('_')[-1])
        print(num)
        vals = []
        for line in f.readlines():
            if 'resolution' not in line:
                continue
            val = float(line.split(',')[-1]) * 1000
            vals.append(val)
    client[num] = vals

nums = sorted(list(client.keys()))

mean = list()
std = list()
for num in nums:
    m = np.mean(client[num])
    mean.append(m)
    s = np.std(client[num])
    std.append(s)

font = {'family': 'normal',

        'weight': 'bold',

        'size': 16}

plt.rc('font', **font)


xlabel = [1, 2, 4, 6]
y_pos = np.arange(len(xlabel))

plt.plot(y_pos, mean, color=(0.621685, 0.164184, 0.388781), marker='x', markersize=10)
plt.xticks(y_pos, xlabel)
plt.ylabel('Usage')
ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', rotation=0)
plt.xlabel('concurrent client num')
plt.ylabel('computing latency (ms)' )
plt.grid(linestyle='dotted')

ratio = 5
plt.gcf().set_size_inches(0.618 *ratio, 1 * ratio)
ax.spines['left'].set_color('#91989F')
ax.spines['bottom'].set_color('#91989F')
plt.tight_layout()
plt.savefig('clientNum_latency.pdf')
plt.show()

