import glob, os, pickle

gt_list = list()
for gt in glob.glob('davis_gt_*.pkl'):
    gt_name = gt.split('.')[0].split('_')[-1]
    gt_list.append(gt_name)


# get all
root_path = '/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p'
paths = list()
for gt in gt_list:
    path = root_path + '/' + gt
    imgs = os.listdir(path)
    imgs.sort()

    paths += [path + '/' + img for img in imgs]


with open('davis_videos.txt', 'w+') as f:
    for path in paths:
        f.write(path + '\n')
