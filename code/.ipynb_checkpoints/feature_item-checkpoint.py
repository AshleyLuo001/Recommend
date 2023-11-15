# coding: utf-8
import pandas as pd
from tqdm import tqdm
import os
from itertools import combinations
import numpy as np
import polars as pl
import sys
import arrow
import time
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


# # 函数
def create_feat_func(in_file_name,out_file_name,pk_list,col_name):
    train_df_01_01=pd.read_feather(input_path+in_file_name)
    train_df_01_01=pl.DataFrame(train_df_01_01)
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
    train_df_01_01[cols].to_feather(output_path+out_file_name)
    print(out_file_name,'已完成')
# 拆分后十天数据
def split_last_10_days(in_file_name,out_file_name,pk_list,col_name):
    user_cnt_df=pd.read_feather(input_path+in_file_name)
    start_day=arrow.get(pred_date).shift(days=-10).format('YYYY-MM-DD')
    user_cnt_df=user_cnt_df[user_cnt_df['日期']>=start_day]
    user_cnt_df=pd.pivot_table(user_cnt_df,index=pk_list+[ '日期'],columns=['Behavior_Type'],values=['cnt'],aggfunc='sum',fill_value=0).reset_index()
    col2zh={1:'浏览次数',2:'收藏次数',3:'加购次数',4:'购买次数'}
    user_cnt_df.columns=[(i[0] if i[1]=='' else col_name+col2zh[i[1]])for i in list(user_cnt_df)]
    user_cnt_df[col_name+'总交互次数']=user_cnt_df[col_name+'浏览次数']+user_cnt_df[col_name+'收藏次数']+user_cnt_df[col_name+'加购次数']+user_cnt_df[col_name+'购买次数']
    user_cnt_df.sort_values(by=['日期'],ascending=[True],inplace=True)
    from itertools import groupby
    cols=list(user_cnt_df)
    for key,rows in tqdm(groupby(user_cnt_df.values,lambda x:x[len(pk_list)])):
        tmp_df=pd.DataFrame(list(rows))
        tmp_df.columns=cols
        del tmp_df['日期']
        tmp_df.columns=pk_list+[key+'_'+i for i in list(tmp_df)[len(pk_list):]]
        tmp_df.to_feather(output_path+out_file_name+key+'.feather')
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
# 衰减求和后N天的计数
def create_last_n_days_feat2(in_file_name,out_file_name,pk_list,col_name,n_days=3,xishu=1):
    train_df_01_01=pd.read_feather(input_path+in_file_name)
    train_df_01_01=pl.DataFrame(train_df_01_01)
    start_day=arrow.get(pred_date).shift(days=-1*n_days).format('YYYY-MM-DD')
    train_df_01_01=train_df_01_01.filter(pl.col('日期')>=start_day)
    train_df_01_01=train_df_01_01.with_columns(pl.Series((pd.to_datetime(pred_date)-pd.to_datetime(train_df_01_01['日期'].to_numpy())).days.tolist()).alias("cnt2"))
    train_df_01_01=train_df_01_01.with_columns(train_df_01_01['cnt2'].apply(lambda x:1/(1+xishu*x)).alias("cnt2"))
    train_df_01_01=train_df_01_01.with_columns((pl.col('cnt')*pl.col('cnt2')).alias("cnt"))
    train_df_01_01=train_df_01_01.pivot(values="cnt", index=pk_list, columns="Behavior_Type", aggregate_function="sum")
    train_df_01_01=train_df_01_01.fill_null(strategy="zero")
    train_df_01_01=train_df_01_01.to_pandas()
    col2zh={'1':'浏览次数','2':'收藏次数','3':'加购次数','4':'购买次数'}
    train_df_01_01.columns=[(i if i in pk_list else col_name+col2zh[i])for i in list(train_df_01_01)]
    train_df_01_01[col_name+'总交互次数']=train_df_01_01[col_name+'浏览次数']+train_df_01_01[col_name+'收藏次数']+train_df_01_01[col_name+'加购次数']+train_df_01_01[col_name+'购买次数']
    cols = pk_list+[
        col_name+'浏览次数', col_name+'收藏次数', col_name+'加购次数', col_name+'购买次数', col_name+'总交互次数'
    ]
    train_df_01_01=train_df_01_01[cols].copy()
    new_col_name='衰减后'+str(n_days)+'天'+col_name
    new_cols=pk_list+[
        new_col_name+'浏览次数', new_col_name+'收藏次数', new_col_name+'加购次数', new_col_name+'购买次数', new_col_name+'总交互次数']
    train_df_01_01.columns=new_cols
    train_df_01_01.to_feather(output_path+out_file_name)
    print(out_file_name,'已完成')

# # 商品特征
# total_time=0
# while not os.path.exists(input_path+'item_cnt.feather'):
#     time.sleep(5)
#     total_time+=5
#     if total_time%30==0:
#         print('等待item_cnt完成_目前耗时'+str(total_time))
    
create_feat_func('item_cnt.feather',train_or_test+'_df_02_01.feather',['Item_ID'],'商品')
create_last_n_days_feat_old('item_cnt.feather',train_or_test+'_df_02_07.feather',['Item_ID'],'商品',3)
create_last_n_days_feat2('item_cnt.feather',train_or_test+'_df_02_05.feather',['Item_ID'],'商品',10)
