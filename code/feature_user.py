# coding: utf-8
import pandas as pd
from tqdm import tqdm
import os
from itertools import combinations
import numpy as np
import polars as pl
import sys
import arrow
import joblib
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas(desc='pandas bar')
raw_data_path=sys.argv[1]
input_path=sys.argv[2]
output_path=sys.argv[3]
config_dict={'train':'2014-12-18','test':'2014-12-19'}
train_or_test='test'
pred_date=config_dict.get(train_or_test)
if not os.path.exists(output_path):
    os.mkdir(output_path)

print('*'*20,'合并user_cnt计算','*'*20)
res=[]
for part in tqdm(range(5)):
    tmp_df_00=pd.read_feather(input_path+str(part)+'_user_cnt.feather')
    tmp_df_00=pl.DataFrame(tmp_df_00)
    res.append(tmp_df_00)
tmp_df_00=pl.concat(res)
tmp_df_00=tmp_df_00.groupby(['User_ID','日期','Behavior_Type']).sum()
tmp_df_00=tmp_df_00.rename({'count':'cnt'})
df_cnt=tmp_df_00.select(['User_ID','日期','Behavior_Type','cnt'])
print('*'*20,'合并user_cnt完成','*'*20)
del res

feat_cols=joblib.load('./weight/feat_cols.pkl')
# # 函数
def create_feat_func(df_cnt,out_file_name,pk_list,col_name):
    train_df_01_01 = df_cnt.filter(pl.col('日期') < '2014-12-18')
    train_df_01_01=train_df_01_01.pivot(values="cnt", index=pk_list, columns="Behavior_Type", aggregate_function="sum")
    train_df_01_01=train_df_01_01.fill_null(strategy="zero")
    train_df_01_01=train_df_01_01.to_pandas()
    col2zh={'1':'浏览次数','2':'收藏次数','3':'加购次数','4':'购买次数'}
    train_df_01_01.columns=[(i if i in pk_list else col_name+col2zh[i])for i in list(train_df_01_01)]
    l2=[col_name+'浏览次数',col_name+'收藏次数',col_name+'加购次数',col_name+'购买次数']
    combs= combinations(range(4),2)
    for tmp_comb in combs:
        col1=l2[tmp_comb[0]]
        col2=l2[tmp_comb[1]]
        if col1+'+'+col2 in feat_cols:
            train_df_01_01[col1+'+'+col2]=train_df_01_01[col1]+train_df_01_01[col2]
    train_df_01_01[col_name+'总交互次数']=train_df_01_01[col_name+'浏览次数']+train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数']+train_df_01_01[col_name+'购买次数']
    if col_name + '转化率1' in feat_cols:
        train_df_01_01[col_name+'转化率1']=train_df_01_01[col_name+'购买次数']/train_df_01_01[col_name+'浏览次数']
        train_df_01_01[col_name + '转化率1'] = train_df_01_01[col_name + '转化率1'].apply(
            lambda x: 1 if x == np.inf else x)
    if col_name + '转化率2' in feat_cols:
        train_df_01_01[col_name+'转化率2']=(train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数'])/train_df_01_01[col_name+'浏览次数']
        train_df_01_01[col_name + '转化率2'] = train_df_01_01[col_name + '转化率2'].apply(
            lambda x: 1 if x == np.inf else x)
    if col_name + '转化率3' in feat_cols:
        train_df_01_01[col_name+'转化率3']=train_df_01_01[col_name+'购买次数']/(train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数'])
        train_df_01_01[col_name + '转化率3'] = train_df_01_01[col_name + '转化率3'].apply(
            lambda x: 1 if x == np.inf else x)
    if col_name + '浏览占比' in feat_cols:
        train_df_01_01[col_name+'浏览占比']=train_df_01_01[col_name+'浏览次数']/train_df_01_01[col_name+'总交互次数']
    if col_name + '收藏占比' in feat_cols:
        train_df_01_01[col_name+'收藏占比']=train_df_01_01[col_name+'收藏次数']/train_df_01_01[col_name+'总交互次数']
    if col_name + '加购占比' in feat_cols:
        train_df_01_01[col_name+'加购占比']=train_df_01_01[col_name+'加购次数']/train_df_01_01[col_name+'总交互次数']
    if col_name + '收藏加购占比' in feat_cols:
        train_df_01_01[col_name+'收藏加购占比']=(train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数'])/train_df_01_01[col_name+'总交互次数']
    if col_name + '购买占比' in feat_cols:
        train_df_01_01[col_name+'购买占比']=train_df_01_01[col_name+'购买次数']/train_df_01_01[col_name+'总交互次数']
    cols = [
        col_name+'浏览次数', col_name+'收藏次数', col_name+'加购次数', col_name+'购买次数', col_name+'浏览次数+'+col_name+'收藏次数',
        col_name+'浏览次数+'+col_name+'加购次数', col_name+'浏览次数+'+col_name+'购买次数', col_name+'收藏次数+'+col_name+'加购次数', col_name+'收藏次数+'+col_name+'购买次数',
        col_name+'加购次数+'+col_name+'购买次数', col_name+'总交互次数', col_name+'转化率1', col_name+'转化率2', col_name+'转化率3', col_name+'浏览占比',
        col_name+'收藏占比',col_name+ '加购占比', col_name+'收藏加购占比', col_name+'购买占比'
    ]
    cols=pk_list +sorted(list(set(feat_cols)&set(cols)))
    train_df_01_01[cols].to_feather(output_path+out_file_name)
    print(out_file_name,'已完成')
# 衰减求和后N天的计数
def create_last_n_days_feat1(df_cnt,out_file_name,pk_list,col_name,n_days=3,xishu=1):
    # df_cnt=pd.read_feather(input_path+in_file_name)
    # df_cnt=pl.DataFrame(df_cnt)
    start_day0=arrow.get(pred_date).shift(days=-1*n_days).format('YYYY-MM-DD')
    df_cnt=df_cnt.filter(pl.col('日期')>=start_day0)
    df_cnt=df_cnt.with_columns(pl.Series((pd.to_datetime(pred_date)-pd.to_datetime(df_cnt['日期'].to_numpy())).days.tolist()).alias("cnt2"))
    df_cnt=df_cnt.with_columns(df_cnt['cnt2'].apply(lambda x:1/(1+xishu*x)).alias("cnt2"))
    df_cnt=df_cnt.with_columns((pl.col('cnt')*pl.col('cnt2')).alias("cnt"))
    train_df_01_01_res=[]
    for n_day in range(1,n_days+1):
        start_day=arrow.get(pred_date).shift(days=-1*n_day).format('YYYY-MM-DD')
        train_df_01_01=df_cnt.filter(pl.col('日期')>=start_day)
        train_df_01_01=train_df_01_01.pivot(values="cnt", index=pk_list, columns="Behavior_Type", aggregate_function="sum")
        train_df_01_01=train_df_01_01.fill_null(strategy="zero")
        train_df_01_01=train_df_01_01.to_pandas()
        col2zh={'1':'浏览次数','2':'收藏次数','3':'加购次数','4':'购买次数'}
        train_df_01_01.columns=[(i if i in pk_list else col_name+col2zh[i])for i in list(train_df_01_01)]
        l2=[col_name+'浏览次数',col_name+'收藏次数',col_name+'加购次数',col_name+'购买次数']
        combs= combinations(range(4),2)
        for tmp_comb in combs:
            col1=l2[tmp_comb[0]]
            col2=l2[tmp_comb[1]]
            train_df_01_01[col1+'+'+col2]=train_df_01_01[col1]+train_df_01_01[col2]
        train_df_01_01[col_name+'总交互次数']=train_df_01_01[col_name+'浏览次数']+train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数']+train_df_01_01[col_name+'购买次数']
        train_df_01_01[col_name+'转化率1']=train_df_01_01[col_name+'购买次数']/train_df_01_01[col_name+'浏览次数']
        train_df_01_01[col_name+'转化率2']=(train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数'])/train_df_01_01[col_name+'浏览次数']
        train_df_01_01[col_name+'转化率3']=train_df_01_01[col_name+'购买次数']/(train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数'])
        train_df_01_01[col_name+'浏览占比']=train_df_01_01[col_name+'浏览次数']/train_df_01_01[col_name+'总交互次数']
        train_df_01_01[col_name+'收藏占比']=train_df_01_01[col_name+'收藏次数']/train_df_01_01[col_name+'总交互次数']
        train_df_01_01[col_name+'加购占比']=train_df_01_01[col_name+'加购次数']/train_df_01_01[col_name+'总交互次数']
        train_df_01_01[col_name+'收藏加购占比']=(train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数'])/train_df_01_01[col_name+'总交互次数']
        train_df_01_01[col_name+'购买占比']=train_df_01_01[col_name+'购买次数']/train_df_01_01[col_name+'总交互次数']
        for i in range(1,4):
            train_df_01_01[col_name+'转化率'+str(i)]=train_df_01_01[col_name+'转化率'+str(i)].apply(lambda x:1 if x==np.inf else x)
        cols = pk_list+[
            col_name+'浏览次数', col_name+'收藏次数', col_name+'加购次数', col_name+'购买次数', col_name+'浏览次数+'+col_name+'收藏次数',
            col_name+'浏览次数+'+col_name+'加购次数', col_name+'浏览次数+'+col_name+'购买次数', col_name+'收藏次数+'+col_name+'加购次数', col_name+'收藏次数+'+col_name+'购买次数',
            col_name+'加购次数+'+col_name+'购买次数', col_name+'总交互次数', col_name+'转化率1', col_name+'转化率2', col_name+'转化率3', col_name+'浏览占比',
            col_name+'收藏占比',col_name+ '加购占比', col_name+'收藏加购占比', col_name+'购买占比'
        ]
        train_df_01_01=train_df_01_01[cols].copy()
        new_col_name='衰减后'+str(n_day)+'天'+col_name
        new_cols=pk_list+[
            new_col_name+'浏览次数', new_col_name+'收藏次数', new_col_name+'加购次数', new_col_name+'购买次数', new_col_name+'浏览次数+'+new_col_name+'收藏次数',
            new_col_name+'浏览次数+'+new_col_name+'加购次数', new_col_name+'浏览次数+'+new_col_name+'购买次数', new_col_name+'收藏次数+'+new_col_name+'加购次数', new_col_name+'收藏次数+'+new_col_name+'购买次数',
            new_col_name+'加购次数+'+new_col_name+'购买次数', new_col_name+'总交互次数', new_col_name+'转化率1', new_col_name+'转化率2', new_col_name+'转化率3', new_col_name+'浏览占比',
            new_col_name+'收藏占比',new_col_name+ '加购占比', new_col_name+'收藏加购占比', new_col_name+'购买占比'
        ]
        train_df_01_01.columns=new_cols
        # new_cols = pk_list + sorted(list(set(feat_cols) & set(new_cols)))
        # train_df_01_01 = pl.DataFrame(train_df_01_01[new_cols])
        train_df_01_01 = pl.DataFrame(train_df_01_01)
        train_df_01_01_res.append(train_df_01_01)
    train_df_01_01 = train_df_01_01_res[0]
    for tmp_df in tqdm(train_df_01_01_res[1:]):
        train_df_01_01 = train_df_01_01.join(tmp_df, how='outer', on=pk_list)
    train_df_01_01 = train_df_01_01.fill_null(strategy="zero")
    train_df_01_01 = train_df_01_01.to_pandas(use_pyarrow_extension_array=True)
    train_df_01_01.to_feather(output_path + out_file_name)
    print(out_file_name,'已完成')

# # 用户特征
create_feat_func(df_cnt,train_or_test+'_df_01_01.feather',['User_ID'],'用户')
create_last_n_days_feat1(df_cnt,train_or_test+'_df_01_07.feather',['User_ID'],'用户',10)

print('feature_user 全部完成')