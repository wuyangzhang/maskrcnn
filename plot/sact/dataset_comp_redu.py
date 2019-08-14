import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import pandas as pd


font = {'family': 'normal',
        'weight': 'bold',
        'size': 20}

plt.rc('font', **font)


def plot():

    coco_trace_path = '/home/wuyang/maskrcnn-benchmark/logs/sact/dataset/coco.log'
    kitti_trace_path = '/home/wuyang/maskrcnn-benchmark/logs/sact/dataset/coco.log'
    human36_trace_path = '/home/wuyang/maskrcnn-benchmark/logs/sact/dataset/coco.log'

    trace_path = [coco_trace_path, kitti_trace_path, human36_trace_path]
    positions = ['block1', 'block2', 'block3', 'block4', 'overall']
    headers = ['val', 'pos', 'dataset']
    dataset_name = ['COCO', 'KITTI', 'HUMAN3.6M']

    def preprocess(path):

        block = [[], [], [], []]
        with open(path, 'r') as f:
            for line in f.readlines():
                # 2019-07-21 18:35:54.986 WARNING:	format:block[x] shows y complexity,3,0.9036841988563538
                if 'format:block[x]' not in line:
                    break
                line = line.split(',')
                block_id, complexity = int(line[1]), float(line[2])
                block[block_id].append(complexity)
        return np.array(block)

    all_data = []
    all_position = []
    all_dataset = []

    for path_c, path in enumerate(trace_path):
        data = preprocess(path)
        overall = (data[0] * 1 * 3 + data[1] * 0.5 * 4 + data[2] * 0.25 * 6 + data[3] * 0.125 * 3) / (3 * 1 + 4 * 0.5 + 6 * 0.25 + 3 * 0.125)
        for i, block in enumerate(data):
            data[i] = 1 - block
            #print(block.mean(), block.std())
        overall = (1 - overall).reshape(1, -1)
        data = np.concatenate([data, overall])
        data = data.reshape(-1)
        all_data.append(data)
        for k in range(len(positions)):
            for _ in range(len(overall[0])):
                all_position.append(positions[k])
        dataset = [dataset_name[path_c] for _ in range(len(data))]
        all_dataset += dataset
    all_data = np.concatenate(all_data)

    # make dict : header + data
    data_dict = dict(zip(headers, [all_data, all_position, all_dataset]))
    df = pd.DataFrame(data_dict)

    # create sns
    ax = sns.boxplot(x='pos', y='val', hue='dataset', data=df,
                     palette='inferno')

    # edit the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=30)
    plt.xlabel('')
    plt.ylabel('GFLOPS reduction (%)' )
    plt.grid(linestyle='dotted')
        #ax.tick_params(axis='x', colors='#91989F')
    ax.spines['left'].set_color('#91989F')
    ax.spines['bottom'].set_color('#91989F')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.legend(loc=0, title='dataset', prop={'size': 12} )

    # auto layout
    plt.tight_layout()

    # cut white space

    plt.savefig('flops_red.png', bbox_inches='tight')

    plt.show()
    # convert numpy data into panda data frames.
    #     data_dict = dict(zip(headers, data))
    #     df = pd.DataFrame(data_dict)
        # path = path.split('/')[-1].split('.')[0] + '.csv'
        # df.to_csv(path)


    #print(overall.mean(), overall.std())

    # ax = plt.gca()
    # ax.spines['left'].set_linestyle('-.')
    # ax.spines['left'].set_color('#91989F')
    # ax.spines['bottom'].set_linestyle('-.')
    # ax.spines['bottom'].set_color('#91989F')
    #
    # ax.spines['right'].set_linestyle('--')
    # ax.spines['right'].set_color('#91989F')
    # ax.spines['top'].set_linestyle('--')
    # ax.spines['top'].set_color('#91989F')
    #
    # ax.tick_params(axis='x', colors='#91989F')
    # ax.tick_params(axis='y', colors='#91989F')
    #
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    #
    # if xticks:
    #     plt.xticks(np.arange(6), xticks)
    # plt.grid(linestyle='dotted')
    #
    # for policy in policyList:
    #     print(policy, vals[policy])
    #     plt.plot(x, vals[policy], label=policy)
    #
    # plt.legend(loc='upper left')
    #
    # if xlabel:
    #     plt.xlabel(xlabel)
    # if ylabel:
    #     plt.ylabel(ylabel)
    #
    # plt.show()


def plotSampleAge(path, xticks=None, xlabel=None, ylabel=None):
    with open(path) as f:
        aoi_list = json.load(f)

    ax = plt.gca()
    ax.spines['left'].set_linestyle('-.')
    ax.spines['left'].set_color('#91989F')
    ax.spines['bottom'].set_linestyle('-.')
    ax.spines['bottom'].set_color('#91989F')

    ax.spines['right'].set_linestyle('--')
    ax.spines['right'].set_color('#91989F')
    ax.spines['top'].set_linestyle('--')
    ax.spines['top'].set_color('#91989F')

    ax.tick_params(axis='x', colors='#91989F')
    ax.tick_params(axis='y', colors='#91989F')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(linestyle='dotted')

    for i, user in enumerate(aoi_list):
        user = user[:500]
        x = [_ for _ in range(len(user))]
        plt.plot(x, user, label='user #' + str(i))

    plt.legend(loc='upper left')

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.show()


def plotBox(path, xlabel=None, ylabel=None):
    with open(path) as f:
        aoi_list = json.load(f)

    ax = plt.gca()
    ax.spines['left'].set_linestyle('-.')
    ax.spines['left'].set_color('#91989F')
    ax.spines['bottom'].set_linestyle('-.')
    ax.spines['bottom'].set_color('#91989F')

    ax.spines['right'].set_linestyle('--')
    ax.spines['right'].set_color('#91989F')
    ax.spines['top'].set_linestyle('--')
    ax.spines['top'].set_color('#91989F')

    ax.tick_params(axis='x', colors='#91989F')
    ax.tick_params(axis='y', colors='#91989F')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.grid(linestyle='dotted')

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(aoi_list, showfliers=False)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.show()


if __name__ == '__main__':
    plot()