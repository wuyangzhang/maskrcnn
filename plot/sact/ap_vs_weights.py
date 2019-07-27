import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

w_001 = dict([('weight', '0.01'), ('bbox', dict(
    [('AP', 0.37232357477684214), ('AP50', 0.5863740423005475), ('AP75', 0.40382327575102195),
     ('APs', 0.2112274267131617), ('APm', 0.4016257887782384), ('APl', 0.48970395004702627)])),

              ('segm', dict([('AP', 0.3371077839078823), ('AP50', 0.5552182522119208), ('AP75', 0.3573245023547559),
                             ('APs', 0.1507996747182951), ('APm', 0.3627639886200648), ('APl', 0.4998970049787665)]))])

w_005 = dict([('weight', '0.05'), ('bbox', dict(
    [('AP', 0.34866861481482186), ('AP50', 0.5599538861632721), ('AP75', 0.3751815570439551),
     ('APs', 0.20367581488110995), ('APm', 0.3831805623340563), ('APl', 0.4444805071268622)])),

              ('segm', dict([('AP', 0.3143195217414323), ('AP50', 0.5244600858127938), ('AP75', 0.3322061198867238),
                             ('APs', 0.14288171145949477), ('APm', 0.34395665062617253),
                             ('APl', 0.45663359646440504)]))])

w_01 = dict([('weight', '0.1'), ('bbox', dict(
    [('AP', 0.3276518119683325), ('AP50', 0.5364703914223221), ('AP75', 0.3493038524099132),
     ('APs', 0.18991484153857188), ('APm', 0.36240442355389335), ('APl', 0.4154895624524634)])),

             ('segm', dict([('AP', 0.2958271233304461), ('AP50', 0.502718997295349), ('AP75', 0.30687069275630285),
                            ('APs', 0.13302608004686256), ('APm', 0.32694546212278375),
                            ('APl', 0.43218378657993284)]))])

w_baseline = dict([('weight', 'baseline'), ('bbox', dict(
    [('AP', 0.3776518119683325), ('AP50', 0.5914703914223221), ('AP75', 0.4093038524099132),
     ('APs', 0.21391484153857188), ('APm', 0.40640442355389335), ('APl', 0.4964895624524634)])),

                   ('segm', dict(
                       [('AP', 0.3398271233304461), ('AP50', 0.502718997295349), ('AP75', 0.30687069275630285),
                        ('APs', 0.13302608004686256), ('APm', 0.32694546212278375), ('APl', 0.43218378657993284)]))])

w_002 = dict([('weight', '0.02'), ('bbox', dict(
    [('AP', 0.3683125455260951), ('AP50', 0.5820351303506894), ('AP75', 0.39946939852251356),
     ('APs', 0.21024319000680516), ('APm', 0.3973503435655747), ('APl', 0.4800701449406582)])),
              ('segm', dict([('AP', 0.33349192038335285), ('AP50', 0.5504807693062532), ('AP75', 0.3515631446216573),
                             ('APs', 0.150018693912216), ('APm', 0.3601419974065695), ('APl', 0.49112823384452203)]))])
w_list = [w_baseline, w_001, w_002, w_005]

font = {'family': 'normal',

        'weight': 'bold',

        'size': 20}

plt.rc('font', **font)

ratio = 10


def plot():

    headers = ['val', 'type', 'cost_weight']

    type = ['boxAP', 'maskAP', 'AP50', 'AP75', 'APs', 'APm', "APl"]

    val_list = []

    type_list = []

    weight_list = []

    for w in w_list:

        re = list(w['bbox'].values())
        re.insert(1, w['segm']['AP'])
        re = [x * 100 for x in re]
        val_list += re
        type_list += type
        for _ in range(len(type)):
            weight_list.append(w['weight'])

# makedict:header+data

    data_dict = dict(zip(headers, [val_list, type_list, weight_list]))

    df = pd.DataFrame(data_dict)

    # createsns


    sns.set_style("whitegrid")

    ax = sns.catplot(x='type', y='val', hue='cost_weight', data=df,

                     palette=sns.cubehelix_palette(8), kind='bar', legend_out=False)

    # legend = ax._legend
    # legend._loc = 1
    plt.gcf().set_size_inches(1 * ratio, 0.618 * ratio)

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

    ax.set_xticklabels(rotation=40)

    # ax.set(linestyle='dotted')


    # plt.grid(linestyle='dotted')

    # ax.tick_params(axis='x',colors='#91989F')

    # ax.spines['left'].set_color('#91989F')

    # ax.spines['bottom'].set_color('#91989F')

    plt.legend(bbox_to_anchor=(0.8,0.7),loc=0,borderaxespad=0., title='cost weight')

    #ax.legend(loc=0,prop={'size':14})


    # autolayout

    plt.tight_layout()

    # plt.legend(loc='upperright')

    #plt.legend(bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0., prop={'size': 20})

    # cutwhitespace


    plt.savefig('ap_vs_weights.pdf', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':

    plot()
