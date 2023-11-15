# coding: utf-8
import pandas as pd
from tqdm import tqdm
import os
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

# # 用户品类特征 此处可以结合feat_cols优化
create_last_feat('user_category_last_time.feather',train_or_test+'_df_05_02.feather',['User_ID','Item_Category'],1)
create_last_feat('user_category_last_time.feather',train_or_test+'_df_05_03.feather',['User_ID','Item_Category'],2)
create_last_feat('user_category_last_time.feather',train_or_test+'_df_05_04.feather',['User_ID','Item_Category'],3)
create_last_feat('user_category_last_time.feather',train_or_test+'_df_05_05.feather',['User_ID','Item_Category'],4)
create_first_feat('user_category_first_time.feather',train_or_test+'_df_05_06.feather',['User_ID','Item_Category'],'category')
create_first_feat2('user_category_cnt.feather',train_or_test+'_df_05_07.feather',['User_ID','Item_Category'],'category')

print('feature_user_category_part1 全部完成')
