import pandas as pd
import polars as pl
from tqdm import tqdm
import datetime
import sys
raw_data_path=sys.argv[1]
input_path=sys.argv[2]
output_path=sys.argv[3]


print('*'*20,'合并user_item_last5_cnt开始','*'*20)
res=[]
for part in tqdm(range(5)):
    df_cnt=pd.read_feather(input_path+str(part)+'_user_item_last5_cnt.feather')
    df_cnt=pl.DataFrame(df_cnt)
    res.append(df_cnt)
df_cnt=pl.concat(res)
df_cnt=df_cnt.groupby(['User_ID','Item_ID','Behavior_Type','日期']).sum()
print('*'*20,'合并user_item_last5_cnt完成','*'*20)
df_cnt_02=df_cnt.filter(pl.col('Behavior_Type')==3)
df_cnt_02=df_cnt_02.select(['User_ID','Item_ID','日期','count'])
df_cnt=df_cnt.select(['User_ID','Item_ID','日期','count']).groupby(['User_ID','Item_ID','日期']).sum()
def create_last_n_days_feat3_last5(df,col_name, xishu=1):
    train_df_01_01 = df.with_columns(
        (datetime.datetime.strptime('2014-12-19', '%Y-%m-%d') - df['日期'].str.to_date("%Y-%m-%d")).dt.days().alias(
            "cnt2"))
    train_df_01_01 = train_df_01_01.with_columns((pl.col('count') / (1 + pl.col('cnt2') * xishu)).alias("cnt"))
    train_df_01_01 = train_df_01_01.select(['User_ID', 'Item_ID', 'cnt']).groupby(['User_ID', 'Item_ID']).sum()
    train_df_01_01 = train_df_01_01.sort(['User_ID', 'cnt'], descending=[False, True])
    train_df_01_01_tmp = train_df_01_01.groupby(['User_ID'], maintain_order=True).agg(
        pl.col("cnt").rank(method='average', descending=True, seed=2023))
    train_df_01_01_tmp = train_df_01_01_tmp.explode("cnt")
    train_df_01_01 = train_df_01_01.with_columns(train_df_01_01_tmp['cnt'].alias(col_name)).select(['User_ID', 'Item_ID', col_name])
    return train_df_01_01
train_df_03_41_tmp1=create_last_n_days_feat3_last5(df_cnt,'衰减后5天用户商品总交互次数_用户rk',2)
train_df_03_41_tmp2=create_last_n_days_feat3_last5(df_cnt_02,'衰减后5天商品加购次数_用户rk',2)
# train_label=pd.read_feather(output_path+'test_label.feather')
# train_label=train_label[['User_ID', 'Item_ID']]
# train_label=pl.DataFrame(train_label)
# train_df_03_41=train_label.join(train_df_03_41_tmp1,how='left',on=['User_ID', 'Item_ID']).join(train_df_03_41_tmp2,how='left',on=['User_ID', 'Item_ID'])
train_df_03_41=train_df_03_41_tmp1.join(train_df_03_41_tmp2,how='left',on=['User_ID', 'Item_ID'])
train_df_03_41=train_df_03_41.to_pandas(use_pyarrow_extension_array=True)
train_df_03_41.to_feather(output_path+'test_df_03_41.feather')
