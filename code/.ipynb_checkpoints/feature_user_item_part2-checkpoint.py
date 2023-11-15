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
def create_last_feat(in_file_name,out_file_name,pk_list,in_type):
    train_df_03_02=pd.read_feather(input_path+in_file_name)
    train_df_03_02=train_df_03_02[train_df_03_02['Behavior_Type']==in_type]
    train_df_03_02['last_time']=pd.to_datetime(train_df_03_02['Time'])
    train_df_03_02['_'.join(pk_list)+'_'+str(in_type)+'_last_look_to_now']=(pd.to_datetime(pred_date)-train_df_03_02['last_time']).dt.total_seconds()/3600
    train_df_03_02=train_df_03_02[pk_list+['_'.join(pk_list)+'_'+str(in_type)+'_last_look_to_now']].reset_index(drop=True)
    train_df_03_02.to_feather(output_path+out_file_name)
    print(out_file_name+' 已完成')
def create_first_feat2(in_file_name,out_file_name,pk_list,feat_type):
    train_df_03_07_tmp=pd.read_feather(output_path+train_or_test+'_df_'+feat_type+'_first_time.feather')
    user_item_cnt_df=pd.read_feather(input_path+in_file_name)
    user_item_cnt_df=user_item_cnt_df[user_item_cnt_df['Behavior_Type']==1]
    user_item_cnt_df=user_item_cnt_df.merge(train_df_03_07_tmp[pk_list],'inner',pk_list)
    train_df_03_07=user_item_cnt_df.groupby(pk_list)['cnt'].sum().reset_index()
    train_df_03_07.columns=pk_list+['第一次购买'+feat_type+'前浏览了几次']
    train_df_03_07.to_feather(output_path+out_file_name)
    print(out_file_name,'已完成')
def create_first_feat(in_file_name,out_file_name,pk_list,feat_type):
    train_df_03_06=pd.read_feather(input_path+in_file_name)
    train_df_03_06_tmp1=train_df_03_06[train_df_03_06['Behavior_Type']==1].copy()
    train_df_03_06_tmp1=train_df_03_06_tmp1[pk_list+['Time']]
    train_df_03_06_tmp1.columns=pk_list+['first_look_time']
    train_df_03_06_tmp2=train_df_03_06[train_df_03_06['Behavior_Type']==4].copy()
    train_df_03_06_tmp2=train_df_03_06_tmp2[pk_list+['Time']]
    train_df_03_06_tmp2.columns=pk_list+['first_buy_time']
    train_df_03_06=train_df_03_06_tmp2.merge(train_df_03_06_tmp1,'inner',pk_list)
    train_df_03_06['first_look_time']=pd.to_datetime(train_df_03_06['first_look_time'])
    train_df_03_06['first_buy_time']=pd.to_datetime(train_df_03_06['first_buy_time'])
    train_df_03_06['_'.join(pk_list)+'_first_look_to_buy']=(train_df_03_06['first_buy_time']-train_df_03_06['first_look_time']).dt.total_seconds()/3600
    train_df_03_06=train_df_03_06[train_df_03_06['_'.join(pk_list)+'_first_look_to_buy']>=0]
    train_df_03_06=train_df_03_06.reset_index(drop=True)
    train_df_03_06[pk_list+['_'.join(pk_list)+'_first_look_to_buy']].to_feather(output_path+out_file_name)
    train_df_03_06.to_feather(output_path+train_or_test+'_df_'+feat_type+'_first_time.feather')
    print(out_file_name+' 已完成')

# 衰减求和后N天的计数
def create_last_n_days_feat3(in_file_name,out_file_name,pk_list,col_name,n_days=3,xishu=1):
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
    train_df_01_01=train_df_01_01[cols]
    new_col_name='衰减后'+str(n_days)+'天'+col_name
    new_cols=pk_list+[
        new_col_name+'浏览次数', new_col_name+'收藏次数', new_col_name+'加购次数', new_col_name+'购买次数', new_col_name+'总交互次数']
    train_df_01_01.columns=new_cols
    train_df_01_01.to_feather(output_path+out_file_name)
    print(out_file_name,'已完成')
# 是否用户最后一小时交互的商品
def df_03_09(in_file_name,out_file_name,pk_list,col_name):
    train_df_03_09_tmp1=pd.read_feather(input_path+in_file_name)
    train_df_03_09_tmp1=pl.DataFrame(train_df_03_09_tmp1)
    train_df_03_09_tmp1=train_df_03_09_tmp1.select(pk_list+['Time']).groupby(pk_list).max()
    train_df_03_09_tmp2=pd.read_feather(input_path+'user_last_time.feather')
    train_df_03_09_tmp2=pl.DataFrame(train_df_03_09_tmp2)
    train_df_03_09=train_df_03_09_tmp1.join(train_df_03_09_tmp2,how='left',on=['User_ID'])
    train_df_03_09=train_df_03_09.with_columns(((train_df_03_09['Time']==train_df_03_09['Time_right']).apply(lambda x:1 if x else 0)).alias("用户最后一小时交互"+col_name))
    train_df_03_09=train_df_03_09.to_pandas()
    train_df_03_09=train_df_03_09[pk_list+["用户最后一小时交互"+col_name]]
    train_df_03_09.to_feather(output_path+out_file_name)
    print(out_file_name,'已完成')

# # 用户商品特征
create_feat_func('user_item_cnt.feather',train_or_test+'_df_03_01.feather',['User_ID','Item_ID'],'用户商品')
create_last_feat('user_item_last_time.feather',train_or_test+'_df_03_02.feather',['User_ID','Item_ID'],1)
create_last_feat('user_item_last_time.feather',train_or_test+'_df_03_03.feather',['User_ID','Item_ID'],2)
df_03_09('user_item_last_time.feather',train_or_test+'_df_03_09.feather',['User_ID','Item_ID'],'')
create_last_feat('user_item_last_time.feather',train_or_test+'_df_03_04.feather',['User_ID','Item_ID'],3)
create_last_feat('user_item_last_time.feather',train_or_test+'_df_03_05.feather',['User_ID','Item_ID'],4)
create_first_feat('user_item_first_time.feather',train_or_test+'_df_03_06.feather',['User_ID','Item_ID'],'item')
create_first_feat2('user_item_cnt.feather',train_or_test+'_df_03_07.feather',['User_ID','Item_ID'],'item')
create_last_n_days_feat3('user_item_last5_cnt.feather',train_or_test+'_df_03_11.feather',['User_ID','Item_ID'],'用户商品',10)

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
create_rank_feat1(train_or_test+'_df_03_11.feather',train_or_test+'_df_03_13.feather',feat_cols)
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
create_rank_feat2(train_or_test+'_df_03_11.feather',train_or_test+'_df_03_15.feather',feat_cols)

print('feature_user_item_part2 全部完成')
