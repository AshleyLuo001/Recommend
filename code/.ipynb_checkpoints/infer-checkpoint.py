
import pandas as pd
import sys
import polars as pl
import os
import xgboost as xgb
import joblib
from tqdm import tqdm
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
    # if train_or_test=='train':
    #     train_df=train_df.sample(frac=1,random_state=2023)
    for file_name in tqdm(
            [train_or_test + '_df_01_01.feather', train_or_test + '_df_01_06.feather',
             train_or_test + '_df_01_07.feather']):
        tmp_df = pd.read_feather(path + file_name)
        tmp_cols = sorted(list(set(tmp_df) & set(feat_cols)))
        tmp_df = tmp_df[['User_ID'] + tmp_cols]
        train_df = train_df.merge(tmp_df, how='left', on='User_ID')
    train_df = pl.DataFrame(train_df)
    # for file_name in tqdm(['train_df_02_01.feather',train_or_test+'_df_02_07.feather',train_or_test+'_df_02_05.feather']):
    for file_name in tqdm([train_or_test + '_df_02_01.feather', train_or_test + '_df_02_07.feather',
                           train_or_test + '_df_02_05.feather']):
        tmp_df = pd.read_feather(path + file_name)
        tmp_cols = sorted(list(set(tmp_df) & set(feat_cols)))
        tmp_df = tmp_df[['Item_ID'] + tmp_cols]
        tmp_df = pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on='Item_ID')
    for i in tqdm([1, 2, 3, 4, 5, 6, 7, 9, 30, 11, 12, 13, 14, 15]):
        file_name = train_or_test + '_df_03_' + str(i).zfill(2) + '.feather'
        tmp_df = pd.read_feather(path + file_name)
        tmp_cols = sorted(list(set(tmp_df) & set(feat_cols)))
        tmp_df = tmp_df[['User_ID', 'Item_ID'] + tmp_cols]
        tmp_df = pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on=['User_ID', 'Item_ID'])
    # for file_name in tqdm(['train_df_04_01.feather',train_or_test+'_df_04_05.feather',train_or_test+'_df_04_06.feather']):
    for file_name in tqdm([train_or_test + '_df_04_01.feather', train_or_test + '_df_04_05.feather',
                           train_or_test + '_df_04_06.feather']):
        tmp_df = pd.read_feather(path + file_name)
        tmp_cols = sorted(list(set(tmp_df) & set(feat_cols)))
        tmp_df = tmp_df[['Item_Category'] + tmp_cols]
        tmp_df = pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on=['Item_Category'])
    for i in tqdm([1, 2, 3, 4, 5, 6, 7, 9, 10, 11]):
        file_name = train_or_test + '_df_05_' + str(i).zfill(2) + '.feather'
        tmp_df = pd.read_feather(path + file_name)
        tmp_cols = sorted(list(set(tmp_df) & set(feat_cols)))
        tmp_df = tmp_df[['User_ID', 'Item_Category'] + tmp_cols]
        tmp_df = pl.DataFrame(tmp_df)
        train_df = train_df.join(tmp_df, how='left', on=['User_ID', 'Item_Category'])
    train_df = train_df.to_pandas()
    
    if train_or_test == 'train':
        train_df['id'] = train_df.apply(
            lambda x: train_or_test + '_' + str(int(x['User_ID'])) + '_' + str(int(x['Item_ID'])), axis=1)
        del train_df['User_ID'], train_df['Item_ID']
        train_df = train_df.sample(frac=1, random_state=2023)
        train_df.reset_index(drop=True, inplace=True)
        train_df.to_feather(input_file.replace('label', 'df'))
    train_df.reset_index(drop=True, inplace=True)
    return train_df


model_xgb=joblib.load('./weight/model.pkl')
feat_cols=joblib.load('./weight/feat_cols.pkl')
test_df_tmp=create_train_test_df2(path,'test',path+'test_label.feather',feat_cols)
xgb_test = xgb.DMatrix(test_df_tmp[feat_cols])
test_df_tmp['pred']=model_xgb.predict(xgb_test)
test_df_tmp=test_df_tmp[test_df_tmp['pred']>1.3][['User_ID','Item_ID','pred']].copy()
test_df_tmp=test_df_tmp.reset_index(drop=True)
print(test_df_tmp.shape)
infer_df=test_df_tmp[['User_ID','Item_ID','pred']]

yuzhi_cnt=38000
l1=[]
for i in tqdm(range(1000)):
    cnt=len(infer_df[infer_df['pred']>i/100])
    l1.append([i/100,abs(yuzhi_cnt-cnt)])
tmp_df_01=pd.DataFrame(l1)
tmp_df_01.columns=['阈值','计数差']
tmp_df_01=tmp_df_01.sort_values(by=['计数差'],ascending=[True])
tmp_df_01=tmp_df_01.reset_index(drop=True)
yuzhi=tmp_df_01['阈值'][0]
infer_df=infer_df[infer_df['pred']>yuzhi]
print(yuzhi)
print(infer_df.shape)
assert len(infer_df)>0
infer_df[['User_ID','Item_ID']].to_csv('./submit.txt',index=False,sep='\t',header=None)
infer_df[['User_ID','Item_ID']].to_csv('./code/submit.txt',index=False,sep='\t',header=None)
infer_df[['User_ID','Item_ID']].to_csv('./result.txt',index=False,sep='\t',header=None)
infer_df[['User_ID','Item_ID']].to_csv('./code/result.txt',index=False,sep='\t',header=None)