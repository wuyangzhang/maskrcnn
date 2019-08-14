import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

font = {'family': 'normal',
        'weight': 'bold',
        'size': 20}

plt.rc('font', **font)

ratio = 12

def plot():

    weight_002 = '/home/wuyang/maskrcnn-benchmark/trace_002.log'
    weight_001 = '/home/wuyang/maskrcnn-benchmark/trace_001.log'
    weight_003 = '/home/wuyang/maskrcnn-benchmark/trace_003.log'
    weight_005 = '/home/wuyang/maskrcnn-benchmark/trace_005.log'
    weight_01 = '/home/wuyang/maskrcnn-benchmark/trace_01.log'

    trace_path = [weight_001, weight_002, weight_005]

    positions = ['block1', 'block2', 'block3', 'block4', 'overall']
    headers = ['val', 'pos', 'dataset']
    weight_name = ['0.01', '0.02', '0.05']

    def preprocess(path):

        block = [[], [], [], []]
        with open(path, 'r') as f:
            start_sig = False
            for line in f.readlines():
                # 2019-07-21 18:35:54.986 WARNING:	format:block[x] shows y complexity,3,0.9036841988563538
                if 'Start evaluation on coco_2014_miniva' in line:
                    start_sig = True
                    continue
                if not start_sig or 'format:block[x]' not in line:
                    continue
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
        print(path, overall.mean())
        data = np.concatenate([data, overall])
        data = data.reshape(-1)
        all_data.append(data)
        for k in range(len(positions)):
            for _ in range(len(overall[0])):
                all_position.append(positions[k])
        dataset = [weight_name[path_c] for _ in range(len(data))]
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

    plt.gcf().set_size_inches(1*ratio, 0.618 * ratio)
        #ax.tick_params(axis='x', colors='#91989F')
    ax.spines['left'].set_color('#91989F')
    ax.spines['bottom'].set_color('#91989F')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.legend(loc=0,  prop={'size': 14}, title='cost weight')

    # auto layout
    plt.tight_layout()

    # cut white space

    plt.savefig('sact_weights.pdf', bbox_inches='tight')

    plt.show()




if __name__ == '__main__':
    plot()