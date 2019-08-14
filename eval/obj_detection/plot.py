import matplotlib.pyplot as plt
import numpy as np


font = {'family': 'normal',

        'weight': 'bold',

        'size': 20}

plt.rc('font', **font)

def plot_overall():
    ratio = 10
    # non partition: 62.8, 403.6 KB, 0.169 + 0.18
    # avg offload size 403.6290533472803 KB, avg server time 0.1696596707592782 seconds

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

    ax = plt.subplot(1, 3, 1)

    plt.bar(np.arange(2), [62.8, 62.3], color=('#FFC408', '#3A8FB7'))
    plt.xticks(np.arange(2), ['non-par', 'MobiDist'], rotation=30)
    plt.ylabel('Performance (IoU)')
    ax.grid(linestyle='dotted')
    #plt.title('Programming language usage')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#91989F')
    ax.spines['bottom'].set_color('#91989F')
    plt.gcf().set_size_inches(1 * ratio, 0.618 * ratio)


    ax = plt.subplot(1, 3, 2)
    plt.bar(np.arange(2), [233, 140], color=('#FFC408', '#3A8FB7'))
    plt.xticks(np.arange(2), ['non-par', 'MobiDist'], rotation=30)
    plt.ylabel('End-to-end latency (ms)')
    ax.grid(linestyle='dotted')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#91989F')
    ax.spines['bottom'].set_color('#91989F')
    plt.gcf().set_size_inches(1 * ratio, 0.618 * ratio)

    ax = plt.subplot(1, 3, 3)
    plt.bar(np.arange(2), [0.42, 0.21], color=('#FFC408', '#3A8FB7'))
    plt.xticks(np.arange(2), ['non-par', 'MobiDist'], rotation=30)
    plt.ylabel('Average offloading size (MB)')
    ax.grid(linestyle='dotted')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#91989F')
    ax.spines['bottom'].set_color('#91989F')
    plt.gcf().set_size_inches(1 * ratio, 0.618 * ratio)

    plt.tight_layout()

    plt.savefig('mask_rcnn_overall.pdf')
    plt.show()


if __name__ == '__main__':
    plot_overall()
