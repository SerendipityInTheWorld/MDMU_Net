import os
import shutil
from tqdm import tqdm

result_path = r''
save_result_path = r''

file_name = os.listdir(result_path)

all_result_path = []

for i in tqdm(file_name):
    name = os.path.join(result_path,i)
    mid_name = os.listdir(name)
    for j in mid_name:
        old_name = os.path.join(name,j)
    name_list = mid_name[0].split('_')
    name_list2 = name_list[2].split('.')
    new_name = name_list[0] + '_' +name_list[1] + '.' + name_list2[1]+'.'+name_list2[2]
    new_name_path = os.path.join(save_result_path,new_name)

    if os.path.exists(save_result_path):
        pass
    else:
        os.mkdir(save_result_path)
    
    shutil.copyfile(old_name,new_name_path)
