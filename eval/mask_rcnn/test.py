paths = []
with open('davis_videos.txt', 'r') as f:
    for line in f.readlines():
        paths.append(line.split()[0])

cur_path = None

masks = []
for img_path in paths:
    # remove cache if a new video is playing
    #print(img_path)
    path = img_path.split('/')[-2]
    if cur_path is not None and path != cur_path:
        print(cur_path)
    if cur_path is None or path != cur_path:
        cur_path = path