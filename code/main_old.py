import os
import time
import shutil

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

if os.path.exists('/tcdata/'):
    raw_data_path = '/tcdata/'
    output_path = '/data/'
    feature_path = '/data/train_test_data/'
elif os.path.exists('/data/final/'):
    raw_data_path = '/data/final/'
    output_path = '/data/mid_feature/'
    feature_path = '/data/mid_feature/train_test_data/'
else:
    raw_data_path = '/mnt/aicas_2023/'
    output_path = '/mnt/test_aicas_2023/'
    feature_path = '/mnt/test_aicas_2023/train_test_data_v2/'
    for file in ['user_last_time.feather', 'user_item_last_time.feather', 'user_item_last5_cnt.feather',
                 'user_item_first_time.feather', 'user_item_cnt.feather', 'user_cnt.feather',
                 'user_category_last_time.feather','user_category_first_time.feather','user_category_cnt.feather'
                 ,'item_cnt.feather','itemcategory_cnt.feather','all_item.feather'
                 ,'user_category_last_time.feather','user_category_last_time.feather','user_category_last_time.feather'
                 ]:
        if os.path.exists(output_path + file):
            os.remove(output_path + file)
    shutil.rmtree(feature_path)


all_path=' '.join([raw_data_path, output_path, feature_path])
two_path=' '.join([output_path, feature_path])

t0 = time.time()
if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(feature_path):
    os.mkdir(feature_path)
print('-' * 20, '5个part并行开始', '-' * 20)
# os.system('python ./dataset/dataloader.py 0 '+raw_data_path+' '+output_path+' & python ./dataset/dataloader.py 1 '+raw_data_path+' '+output_path
#           +' & python ./dataset/dataloader.py 4 '+raw_data_path+' '+output_path+' & wait')
# os.system('python ./dataset/dataloader.py 3 '+raw_data_path+' '+output_path
#           +' & python ./dataset/dataloader.py 2 '+raw_data_path+' '+output_path+' & wait')
os.system('python ./dataset/dataloader.py 0 '+raw_data_path+' '+output_path+' & python ./dataset/dataloader.py 1 '+raw_data_path+' '+output_path
          +' & python ./dataset/dataloader.py 4 '+raw_data_path+' '+output_path+' & python ./dataset/dataloader.py 3 '+raw_data_path+' '+output_path
          +' & python ./dataset/dataloader.py 2 '+raw_data_path+' '+output_path+' & wait')
print('-' * 20, '5个part并行完成', '-' * 20)


print('-' * 20, 'test_label开始', '-' * 20)
# 距离用户最后一次登录特征
os.system('python ./code/test_label_category.py '+all_path+' & python ./code/merge_user_last_time.py '+two_path+' & python ./code/test_label.py '+two_path+' & python ./code/feature_user_item_last5.py '+all_path+' & wait')
# os.system('python ./code/test_label.py '+all_path)
print('-' * 20, 'test_label完成', '-' * 20)

t1 = time.time()
print(round((t1 - t0) / 60, 4), '分钟')
print('-' * 20, '特征计算并行开始', '-' * 20)
# 距离用户最后一次登录特征
# os.system('python ./code/feature_user.py ' + all_path + ' & python ./code/feature_category.py ' + all_path + ' & (python ./code/merge_user_item_cnt.py '+two_path+' && (python ./code/feature_item.py '+all_path+' & python ./code/feature_user_item_part2.py '+all_path+' & wait)) & (python ./code/merge_user_item_time.py ' + two_path +' && (python ./code/feature_user_item_part1.py '+all_path+' & python ./code/feature_user_item_part3.py '+all_path+' & wait)) & (python ./code/merge_user_category_time.py ' + output_path + ' && (python ./code/feature_user_category_part1.py '+all_path+' & python ./code/feature_user_category_part3.py '+all_path+' & wait)) & (python ./code/merge_user_category_cnt.py ' + output_path + ' && python ./code/feature_user_category_part2.py '+all_path+') & wait')
# 只有part1和part2并行
# os.system('python ./code/feature_user.py ' + all_path + ' & python ./code/feature_category.py ' + all_path + ' & (python ./code/merge_user_item_cnt.py '+two_path+' && (python ./code/feature_item.py '+all_path+' & python ./code/feature_user_item_part2.py '+all_path+' & wait)) & (python ./code/merge_user_item_time.py ' + two_path +' && python ./code/feature_user_item_part1.py '+all_path+') & (python ./code/merge_user_category_time.py ' + output_path + ' && python ./code/feature_user_category_part1.py '+all_path+') & (python ./code/merge_user_category_cnt.py ' + output_path + ' && python ./code/feature_user_category_part2.py '+all_path+') & wait')
os.system('python ./code/feature_user.py ' + all_path + ' & python ./code/feature_category.py ' + all_path + ' & (python ./code/merge_user_item_cnt.py '+two_path+' && (python ./code/feature_item.py '+all_path+' & python ./code/feature_user_item_part2.py '+all_path+' & wait)) & (python ./code/merge_user_item_time.py ' + two_path +' && python ./code/feature_user_item_part1.py '+all_path+') & (python ./code/merge_user_category_time.py ' + output_path + ' && python ./code/feature_user_category_part1.py '+all_path+') & (python ./code/merge_user_category_cnt.py ' + output_path + ' && python ./code/feature_user_category_part2.py '+all_path+') & wait')
print('-' * 20, '特征计算并行完成', '-' * 20)
t2 = time.time()
print(round((t2 - t1) / 60, 4), '分钟')


t3 = time.time()
print(round((t3 - t2) / 60, 4), '分钟')

print('-' * 20, '推理开始', '-' * 20)
os.system('python ./code/infer_part1.py ' + feature_path)
print('-' * 20, '推理完成', '-' * 20)
t4 = time.time()
print(round((t4 - t3) / 60, 4), '分钟')

print('总共', round((t4 - t0) / 60, 4), '分钟')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

