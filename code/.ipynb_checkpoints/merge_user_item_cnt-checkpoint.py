import pandas as pd
import polars as pl
from tqdm import tqdm
import sys
print('*'*20,'合并user_item_cnt计算','*'*20)
output_path=sys.argv[1]

tmp_df_00=pd.read_feather(output_path+'0_user_item_cnt.feather')
tmp_df_00=tmp_df_00.rename(columns={'count':'cnt_0'})
tmp_df_00=pl.DataFrame(tmp_df_00)
for part in tqdm(range(1,5)):
    tmp_df_01=pd.read_feather(output_path+str(part)+'_user_item_cnt.feather')
    tmp_df_01=tmp_df_01.rename(columns={'count':'cnt_'+str(part)})
    tmp_df_01=pl.DataFrame(tmp_df_01)
    tmp_df_00=tmp_df_00.join(tmp_df_01,how='outer',on=['User_ID','Item_ID','Item_Category','日期','Behavior_Type'])
tmp_df_00=tmp_df_00.fill_null(strategy="zero")
tmp_df_00=tmp_df_00.with_columns((tmp_df_00['cnt_0']+tmp_df_00['cnt_1']+tmp_df_00['cnt_2']+tmp_df_00['cnt_3']+tmp_df_00['cnt_4']).alias("cnt"))
tmp_df_00=tmp_df_00.select(['User_ID','Item_ID','Item_Category','日期','Behavior_Type','cnt'])

# 按照商品聚合，可以只取子集部分
tmp_df_01=tmp_df_00.select(['Item_ID','日期','Behavior_Type','cnt']).groupby(['Item_ID','日期','Behavior_Type']).sum()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+'item_cnt.feather')
# 过滤用户商品，只取子集部分
tmp_df_00=tmp_df_00.to_pandas()
tmp_df_00.to_feather(output_path+'user_item_cnt.feather')
print('*'*20,'合并user_item_cnt完成','*'*20)