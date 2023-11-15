# coding: utf-8
import pandas as pd
from tqdm import tqdm
import os
import gc
from itertools import combinations
import numpy as np
import polars as pl
import arrow
import sys
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



# 求和后N天的计数
def create_last_n_days_feat_old(in_file_name,out_file_name,pk_list,col_name,n_days=3):
    df_cnt=pd.read_feather(input_path+in_file_name)
    df_cnt=pl.DataFrame(df_cnt)
    start_day0=arrow.get(pred_date).shift(days=-1*n_days).format('YYYY-MM-DD')
    df_cnt=df_cnt.filter(pl.col('日期')>=start_day0)
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
        new_col_name='后'+str(n_day)+'天'+col_name
        new_cols=pk_list+[
            new_col_name+'浏览次数', new_col_name+'收藏次数', new_col_name+'加购次数', new_col_name+'购买次数', new_col_name+'浏览次数+'+new_col_name+'收藏次数',
            new_col_name+'浏览次数+'+new_col_name+'加购次数', new_col_name+'浏览次数+'+new_col_name+'购买次数', new_col_name+'收藏次数+'+new_col_name+'加购次数', new_col_name+'收藏次数+'+new_col_name+'购买次数',
            new_col_name+'加购次数+'+new_col_name+'购买次数', new_col_name+'总交互次数', new_col_name+'转化率1', new_col_name+'转化率2', new_col_name+'转化率3', new_col_name+'浏览占比',
            new_col_name+'收藏占比',new_col_name+ '加购占比', new_col_name+'收藏加购占比', new_col_name+'购买占比'
        ]
        train_df_01_01.columns=new_cols
        train_df_01_01=pl.DataFrame(train_df_01_01)
        train_df_01_01_res.append(train_df_01_01)
    train_df_01_01=train_df_01_01_res[0]
    for tmp_df in train_df_01_01_res[1:]:
        train_df_01_01=train_df_01_01.join(tmp_df,how='outer',on=pk_list)
    train_df_01_01=train_df_01_01.fill_null(strategy="zero")
    train_df_01_01=train_df_01_01.to_pandas()
    train_df_01_01.to_feather(output_path+out_file_name)
    print(out_file_name,'已完成')

# # 用户商品特征
create_last_n_days_feat_old('user_item_last5_cnt.feather',train_or_test+'_df_03_30.feather',['User_ID','Item_ID'],'用户商品',3)

# # 用户下的排序特征
# 用户商品在品类里的排序
import joblib
feat_cols=joblib.load('./weight/feat_cols.pkl')
def create_rank_feat1(input_file,output_file,feat_cols):
    train_df_03_10=pd.read_feather(output_path+input_file)
    all_item=pd.read_feather(input_path+'all_item.feather')
    train_df_03_10=train_df_03_10.merge(all_item,'left','Item_ID')
    cols=[i for i in list(train_df_03_10) if i not in ['User_ID','Item_ID','Item_Category']]
    new_cols = ['User_ID', 'Item_ID']
    for col in tqdm(cols):
        new_col=col+'_品类rk'
        if new_col in feat_cols:
            train_df_03_10[col+'_品类rk']=train_df_03_10.groupby(['User_ID','Item_Category'])[col].rank(ascending=False)
            new_cols.append(new_col)
    for col in list(train_df_03_10):
        if col not in new_cols:
            del train_df_03_10[col]
    train_df_03_10.to_feather(output_path+output_file)
create_rank_feat1(train_or_test+'_df_03_30.feather',train_or_test+'_df_03_12.feather',feat_cols)
# 用户商品在用户里的排序
def create_rank_feat2(input_file,output_file,feat_cols):
    train_df_03_10=pd.read_feather(output_path+input_file)
    cols=[i for i in list(train_df_03_10) if i not in ['User_ID','Item_ID']]
    new_cols = ['User_ID', 'Item_ID']
    for col in tqdm(cols):
        new_col = col + '_用户rk'
        if new_col in feat_cols :
            train_df_03_10[col+'_用户rk']=train_df_03_10.groupby(['User_ID'])[col].rank(ascending=False)
            new_cols.append(new_col)
    for col in list(train_df_03_10):
        if col not in new_cols:
            del train_df_03_10[col]
    train_df_03_10.to_feather(output_path+output_file)
    print(output_file,'已完成')
create_rank_feat2(train_or_test+'_df_03_30.feather',train_or_test+'_df_03_14.feather',feat_cols)


print('feature_user_item 全部完成')
