import os
import glob
import tqdm

# 这里设置TT100K_root_path为数据集的根目录
TT100K_root_path = 'F:\dataset\TT100k\data'
classes_id = []
for file in ['train', 'test']:
    txt_root = os.path.join(TT100K_root_path, os.path.join('labels', file))
    txt_paths = os.listdir(txt_root)
    for txt in tqdm.tqdm(txt_paths):
        with open(os.path.join(TT100K_root_path + '/labels/' + file, txt), 'r') as f:
            objects = f.readlines()
            if objects:
                for object in objects:
                    object_class = object.split(' ')[0]
                    if object_class not in classes_id:
                        classes_id.append(object_class)
print(len(classes_id))
