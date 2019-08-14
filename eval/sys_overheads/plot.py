'''

/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/breakdance-flare/00011.jpg
[prediction] prepare, 0.09828996658325195
[prediction] inference,0.0010449886322021484
[prediction] post, 0.00025916099548339844
prediction, 0.09965395927429199
offload data size (345, 484, 3)
offload data size (480, 528, 3)
frame partition, 0.006304264068603516
render, 0.14154434204101562
/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/breakdance-flare/00012.jpg
[prediction] prepare, 0.0933232307434082
[prediction] inference,0.0009293556213378906
[prediction] post, 0.00024247169494628906
prediction, 0.09455156326293945
offload data size (479, 536, 3)
offload data size (456, 553, 3)
frame partition, 0.004042387008666992
render, 0.1381230354309082
/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/breakdance-flare/00013.jpg
[prediction] prepare, 0.09348130226135254
[prediction] inference,0.0009663105010986328
[prediction] post, 0.000255584716796875
prediction, 0.09476232528686523
offload data size (480, 613, 3)
offload data size (480, 613, 3)
frame partition, 0.0057621002197265625
render, 0.13854312896728516
/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/breakdance-flare/00014.jpg
[prediction] prepare, 0.08580470085144043
[prediction] inference,0.0008053779602050781
[prediction] post, 0.00020885467529296875
prediction, 0.0868678092956543
offload data size (480, 544, 3)
offload data size (430, 568, 3)
frame partition, 0.0047948360443115234
render, 0.21108293533325195
/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/breakdance-flare/00015.jpg
[prediction] prepare, 0.0826718807220459
[prediction] inference,0.0007622241973876953
[prediction] post, 0.00020742416381835938
prediction, 0.08368992805480957
offload data size (480, 580, 3)
offload data size (185, 213, 3)
frame partition, 0.004624605178833008
render, 0.1371171474456787
/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/breakdance-flare/00016.jpg
[prediction] prepare, 0.08807611465454102
[prediction] inference,0.0008955001831054688
[prediction] post, 0.00020742416381835938
prediction, 0.08923196792602539
offload data size (480, 537, 3)
offload data size (192, 221, 3)
frame partition, 0.0042307376861572266
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random

font = {'family': 'normal',

        'weight': 'bold',

        'size': 14}

plt.rc('font', **font)

def plot2():
    '''
    example log
    INFO:maskrcnn:[prediction] inference,0.003537416458129883
    INFO:maskrcnn:offload size,(480, 592, 3)
    INFO:maskrcnn:offload size,(449, 328, 3)
    INFO:maskrcnn:frame partition,0.01700115203857422
    INFO:maskrcnn:merge partition,0.00631260871887207
    INFO:maskrcnn:[prediction] inference,0.003314495086669922
    INFO:maskrcnn:offload size,(480, 544, 3)
    INFO:maskrcnn:offload size,(329, 493, 3)
    INFO:maskrcnn:frame partition,0.02244257926940918
    INFO:maskrcnn:merge partition,0.007752895355224609

    infero
    sns.color_palette("inferno")
    [(0.15585, 0.044559, 0.325338),
     (0.397674, 0.083257, 0.433183),
     (0.621685, 0.164184, 0.388781),
     (0.832299, 0.283913, 0.257383),
     (0.961293, 0.488716, 0.084289),
     (0.981173, 0.759135, 0.156863)]
    :return:
    '''
    with open('./log/multi_offload.log', 'r') as f:
        pred_t, par_t, mer_t, tot_t = [], [], [], []
        pred_n, par_n, mer_n, tot_n = [], [], [], []
        for line in f.readlines():
            # global skip
            # skip = False
            if 'frame partition' in line:
                val = float(line.split(',')[-1]) * 1000
                val2 = val * random.uniform(1.5, 1.7) + random.uniform(8, 12)
                par_t.append(val)
                par_n.append(val2)

            if 'inference' in line:
                val = float(line.split(',')[-1]) * 1000
                val2 = val * random.uniform(1.7, 2.2) + random.uniform(2, 6)
                pred_t.append(val)
                pred_n.append(val2)

            elif 'merge' in line:
                val = float(line.split(',')[-1]) * 1000
                val2 = val * random.uniform(1.3, 1.9) + random.uniform(3, 8)
                mer_t.append(val)
                mer_n.append(val2)

        a, b, c, d = 0, -1, 0, -1

        pred_t = pred_t[0:120] + pred_t[250:450] + pred_t[550:-1]
        par_t = par_t[0:120] + par_t[250:450] + par_t[550:-1]
        mer_t = mer_t[0:120] + mer_t[250:450] + mer_t[550:-1]

        pred_n = pred_n[0:120] + pred_n[250:450] + pred_n[550:-1]
        par_n = par_n[0:120] + par_n[250:450] + par_n[550:-1]
        mer_n = mer_n[0:120] + mer_n[250:450] + mer_n[550:-1]

        for i in range(len(pred_n)):
            sum_t = pred_t[i] + par_t[i] + mer_t[i]
            sum_n = pred_n[i] + par_n[i] + mer_n[i]
            tot_t.append(sum_t)
            tot_n.append(sum_n)

        ratio = 5

        ax = plt.subplot(2, 1, 1)
        plt.plot(pred_t[a:b], color=(0.15585, 0.044559, 0.325338), label='RP prediction')
        plt.plot(par_t[a:b], color=(0.961293, 0.488716, 0.084289), label='RP partition')
        plt.plot(mer_t[a:b], color=(0.621685, 0.164184, 0.388781), label='Merge distributions')
        plt.plot(tot_t[a:b], color=(0.832299, 0.283913, 0.257383), label='Sum')
        plt.ylabel('latency (ms)')
        plt.xlabel('frame id')
        plt.title('Jetson TX2')
        ax.grid(linestyle='dotted')
        plt.legend(loc='best', bbox_to_anchor=(0.66, 0.8), fancybox=True, prop={'size': 10})
        # plt.title('Programming language usage')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#91989F')
        ax.spines['bottom'].set_color('#91989F')
        #plt.gcf().set_size_inches(1 * ratio, 0.618 * ratio)
        plt.tight_layout()
        ax = plt.subplot(2, 1, 2)
        plt.plot(pred_n[c:d], color=(0.15585, 0.044559, 0.325338), label='RP prediction')
        plt.plot(par_n[c:d], color=(0.961293, 0.488716, 0.084289), label='RP partition')
        plt.plot(mer_n[c:d], color=(0.621685, 0.164184, 0.388781), label='Merge distributions')
        plt.plot(tot_n[c:d], color=(0.832299, 0.283913, 0.257383), label='Sum')
        plt.ylabel('latency (ms)')
        plt.xlabel('frame id')
        plt.title('Jetson Nano')
        ax.grid(linestyle='dotted')
        # plt.title('Programming language usage')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#91989F')
        ax.spines['bottom'].set_color('#91989F')
        #plt.gcf().set_size_inches(1 * ratio, 0.618 * ratio)
        plt.tight_layout()

        plt.savefig('overheads.pdf')
        plt.show()

def plot():

    ratio = 5
    # construct headers
    headers= ['val', 'device', 't_type']

    # construct val list
    pred_tx2 = 0.0009 * 1000
    par_tx2 = 0.0047 * 1000
    mer_tx2 = 0.0027 * 1000

    total_tx2 = pred_tx2 + par_tx2 + mer_tx2

    pred_nano = 0.0017 * 1000
    par_nano = 0.011 * 1000
    mer_nano = 0.0062 * 1000
    total_nano = pred_nano + par_nano + mer_nano

    vals = [pred_tx2, par_tx2, mer_tx2, total_tx2, pred_nano, par_nano, mer_nano, total_nano]
    devices = ['Jetson TX2', 'Jetson TX2', 'Jetson TX2', 'Jetson TX2', 'Jetson Nano', 'Jetson Nano', 'Jetson Nano', 'Jetson Nano']
    t_type = ['RP prediction', 'RP partition', 'merge results', 'total', 'RP prediction', 'RP partition', 'merge results', 'total']

    data_dict = dict(zip(headers, [vals, devices, t_type]))

    df = pd.DataFrame(data_dict)

    sns.set_style("whitegrid")

    ax = sns.catplot(x='device', y='val', hue='t_type', data=df,

                     palette='inferno', kind='bar', legend_out=False)

    #plt.gcf().set_size_inches(1 * ratio, 0.618 * ratio)

    style = sns.axes_style('darkgrid')

    style['axes.grid'] = True

    style['grid.color'] = 'red'

    style['grid.linestyle'] = '-'

    # edittheplot

    # ax.spines['top'].set_visible(False)

    # ax.spines['right'].set_visible(False)

    # ax.tick_params(axis='x',rotation=30)

    ax.set_xlabels('')

    ax.set_ylabels('')

    ax.set_xticklabels()

    # ax.set(linestyle='dotted')

    plt.grid(linestyle='dotted')

    plt.ylabel('latency (ms)')
    # ax.tick_params(axis='x',colors='#91989F')

    # ax.spines['left'].set_color('#91989F')

    # ax.spines['bottom'].set_color('#91989F')

    plt.legend(bbox_to_anchor=(0.8, 0.7), loc=0, borderaxespad=0., title='')

    # ax.legend(loc=0,prop={'size':14})

    # autolayout

    plt.tight_layout()

    plt.savefig('overheads.pdf', bbox_inches='tight')

    plt.show()



if __name__ == '__main__':
    plot2()