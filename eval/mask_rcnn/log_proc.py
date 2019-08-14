

def single_offload(log_addr):
    '''
    INFO:maskrcnn:path,breakdance-flare
    INFO:maskrcnn:offload size,(480, 854, 3)
    INFO:maskrcnn:server processing time,0.15617942810058594
    INFO:maskrcnn:/home/nvidia/datasets/davis/DAVIS/JPEGImages/480p/breakdance-flare/00021.jpg
    INFO:maskrcnn:path,breakdance-flare
    INFO:maskrcnn:offload size,(480, 854, 3)
    INFO:maskrcnn:server processing time,0.1607205867767334
    :return:
    '''

    with open(log_addr, 'r') as f:
        size = []
        time = []
        for line in f.readlines():
            if 'size' in line:
                line = line.split('(')[1].split(')')[0]
                line = line.split(',')
                h, w = int(line[0]), int(line[1])
                size.append(h*w*3)
            if 'time' in line:
                line = line.split(',')[1]
                time.append(float(line))

        print('avg offload size {} KB, avg server time {} seconds'.format(
            sum(size) / len(size) / 1024, #kb
            sum(time) / len(time)
        ))

if __name__ == '__main__':
    single_offload('./log/single_offload.log')

