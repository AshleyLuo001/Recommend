import pandas as pd
import sys
import polars as pl
import os
import joblib
from tqdm import tqdm
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
path=sys.argv[1]
test_df_path=path+'test_df_v40/'
infer_df_path=path+'infer_df_v40/'
if not os.path.exists(test_df_path):
    os.mkdir(test_df_path)
if not os.path.exists(infer_df_path):
    os.mkdir(infer_df_path)


def create_train_test_df2(path, train_or_test, input_file, feat_cols):
    train_df = pd.read_feather(input_file)
    train_df=pl.DataFrame(train_df)
    # if train_or_test=='train':
    #     train_df=train_df.sample(frac=1,random_state=2023)
    for file_name in tqdm(
            [train_or_test + '_df_01_01.feather',train_or_test + '_df_01_07.feather']):
        tmp_df = pd.read_feather(path + file_name)
        print(file_name,tmp_df.shape)
        # if '01_01' in file_name :
        tmp_cols=sorted(list(set(tmp_df)&set(feat_cols)))
        tmp_df=tmp_df[['User_ID']+tmp_cols]
        tmp_df=pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on='User_ID')
    for file_name in tqdm([train_or_test + '_df_02_01.feather', train_or_test + '_df_02_05.feather']):
        tmp_df = pd.read_feather(path + file_name)
        print(file_name, tmp_df.shape)
        tmp_cols = sorted(list(set(tmp_df) & set(feat_cols)))
        tmp_df = tmp_df[['Item_ID'] + tmp_cols]
        tmp_df = pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on='Item_ID')
    for i in tqdm([1, 2, 3, 4, 5,  11,41]):# 38 距离最后一次登录,40 6, 7,
        file_name = train_or_test + '_df_03_' + str(i).zfill(2) + '.feather'
        tmp_df = pd.read_feather(path + file_name)
        print(file_name, tmp_df.shape)
        # if i!=11:
        tmp_cols=sorted(list(set(tmp_df)&set(feat_cols)))
        tmp_df=tmp_df[['User_ID','Item_ID']+tmp_cols]
        tmp_df = pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on=['User_ID', 'Item_ID'])
    for file_name in tqdm([train_or_test + '_df_04_01.feather', train_or_test + '_df_04_06.feather']):
        tmp_df = pd.read_feather(path + file_name)
        print(file_name, tmp_df.shape)
        tmp_cols = sorted(list(set(tmp_df) & set(feat_cols)))
        tmp_df = tmp_df[['Item_Category'] + tmp_cols]
        tmp_df = pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on=['Item_Category'])
    for i in tqdm([1, 2, 3, 4, 5, 6, 7, 11]):# 21 距离最后一次登录
        file_name = train_or_test + '_df_05_' + str(i).zfill(2) + '.feather'
        tmp_df = pd.read_feather(path + file_name)
        print(file_name, tmp_df.shape)
        # if i!=11:
        tmp_cols=sorted(list(set(tmp_df)&set(feat_cols)))
        tmp_df=tmp_df[['User_ID','Item_Category']+tmp_cols]
        tmp_df = pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on=['User_ID', 'Item_Category'])
    train_df = train_df.to_pandas()

    # for tuple_tmp in [('衰减后10天用户商品浏览次数', '衰减后10天用户浏览次数'),
    #                   ('衰减后10天用户商品收藏次数', '衰减后10天用户收藏次数'),
    #                   ('衰减后10天用户商品加购次数', '衰减后10天用户加购次数'),
    #                   ('衰减后10天用户商品购买次数', '衰减后10天用户购买次数'),
    #                   ('衰减后10天用户商品总交互次数', '衰减后10天用户总交互次数')]:
    #     col1 = tuple_tmp[0]
    #     col2 = tuple_tmp[1]
    #     train_df[col1 + '_占比1'] = train_df[col1] / train_df[col2]
    #
    # for tuple_tmp in [('衰减后10天用户商品浏览次数', '衰减后10天用户品类浏览次数'),
    #                   ('衰减后10天用户商品收藏次数', '衰减后10天用户品类收藏次数'),
    #                   ('衰减后10天用户商品加购次数', '衰减后10天用户品类加购次数'),
    #                   ('衰减后10天用户商品购买次数', '衰减后10天用户品类购买次数'),
    #                   ('衰减后10天用户商品总交互次数', '衰减后10天用户品类总交互次数')]:
    #     col1 = tuple_tmp[0]
    #     col2 = tuple_tmp[1]
    #     train_df[col1 + '_占比2'] = train_df[col1] / train_df[col2]

    # cols=['User_ID_Item_ID_'+str(i)+'_last_look_to_now' for i in range(1,5)]
    # train_df['User_ID_Item_ID_last_look_to_now']=train_df[cols].min(axis=1)
    # cols = ['User_ID_Item_Category_' + str(i) + '_last_look_to_now' for i in range(1, 5)]
    # train_df['User_ID_Item_Category_last_look_to_now']=train_df[cols].min(axis=1)
    # for col in ['User_ID_Item_ID_1_last_look_to_now',
    #             'User_ID_Item_ID_2_last_look_to_now',
    #             'User_ID_Item_ID_3_last_look_to_now',
    #             'User_ID_Item_ID_4_last_look_to_now',
    #             'User_ID_Item_ID_last_look_to_now',
    #             'User_ID_Item_Category_1_last_look_to_now',
    #             'User_ID_Item_Category_2_last_look_to_now',
    #             'User_ID_Item_Category_3_last_look_to_now',
    #             'User_ID_Item_Category_4_last_look_to_now',
    #             'User_ID_Item_Category_last_look_to_now']:
    #     col2zh = {'1': '浏览', '2': '收藏', '3': '加购', '4': '购买'}
    #     tmp_col='用户最后'+col2zh.get(col.split('_')[4],'登录')+'距今'
    #     train_df[col+'_距离用户最后登录']=train_df[col]-train_df[tmp_col]
        
    if train_or_test == 'train':
        train_df['id'] = train_df.apply(
            lambda x: train_or_test + '_' + str(int(x['User_ID'])) + '_' + str(int(x['Item_ID'])), axis=1)
        del train_df['User_ID'], train_df['Item_ID']
        train_df = train_df.sample(frac=1, random_state=2023)
        train_df.reset_index(drop=True, inplace=True)
        train_df.to_feather(input_file.replace('label', 'df'))
    # train_df.reset_index(drop=True, inplace=True)
    return train_df

seeds=5
feat_cols=joblib.load('./weight/feat_cols.pkl')
test_df_tmp=create_train_test_df2(path,'test',path+'test_label.feather',feat_cols)
xgb_test = xgb.DMatrix(test_df_tmp[feat_cols])# 这里比较慢
for seed in tqdm(range(seeds)):
    model_xgb = joblib.load('./weight/model_'+str(seed)+'.pkl')
    test_df_tmp['pred'+str(seed)] = model_xgb.predict(xgb_test)

test_df_tmp['pred']=test_df_tmp[['pred'+str(seed) for seed in range(seeds)]].mean(axis=1)
print(test_df_tmp[['User_ID','Item_ID','pred']+['pred'+str(seed) for seed in range(seeds)]].head())
test_df_tmp = test_df_tmp[test_df_tmp['pred'] > 2].copy()
# xgb_test = xgb.DMatrix(test_df_tmp[feat_cols])
# joblib.dump(xgb_test,test_df_path+'xgb_test.pkl')

del test_df_tmp['pred']
for seed in range(seeds):
    del test_df_tmp['pred'+str(seed)]
test_df_tmp=test_df_tmp.reset_index(drop=True)
test_df_tmp.to_feather(test_df_path+'test_df.feather')