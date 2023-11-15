import pandas as pd
import polars as pl
from tqdm import tqdm
import sys
output_path=sys.argv[1]
feature_path=sys.argv[2]


print('*'*20,'合并user_last_time开始','*'*20)
res=[]
for part in tqdm(range(5)):
    # tmp_df_01=pd.read_feather(output_path+str(part)+'_user_last_time.feather')
    tmp_df_01=pd.read_feather(output_path+str(part)+'_user_last_time_v2.feather')
    tmp_df_01=pl.DataFrame(tmp_df_01)
    res.append(tmp_df_01)
tmp_df_01=pl.concat(res)
# tmp_df_01=tmp_df_01.groupby(['User_ID']).min()# 这里以前算错了
# tmp_df_01=tmp_df_01.to_pandas()
# tmp_df_01.to_feather(output_path+'user_last_time.feather')
tmp_df_01=tmp_df_01.groupby(['User_ID','Behavior_Type']).max()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01['用户最后登录距今']=(pd.to_datetime('2014-12-18')-pd.to_datetime(tmp_df_01['Time'])).dt.total_seconds()/3600
del tmp_df_01['Time']
tmp_df_01=pl.DataFrame(tmp_df_01)
pk_list=['User_ID']
train_df_01_20=tmp_df_01.pivot(values="用户最后登录距今",index=pk_list,columns="Behavior_Type",aggregate_function="min")
train_df_01_20=train_df_01_20.to_pandas(use_pyarrow_extension_array=True)
col2zh = {'1': '浏览', '2': '收藏', '3': '加购', '4': '购买'}
train_df_01_20.columns = [(i if i in pk_list else '用户最后' + col2zh[i]+'距今') for i in list(train_df_01_20)]
tmp_df_01=tmp_df_01.select(pk_list+['用户最后登录距今']).groupby(pk_list).min().to_pandas(use_pyarrow_extension_array=True)
train_df_01_20=train_df_01_20.merge(tmp_df_01,how='left',on=pk_list)
train_df_01_20.to_feather(feature_path+'test_df_01_20.feather')
print('*'*20,'合并user_last_time完成','*'*20)