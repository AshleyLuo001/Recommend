import pandas as pd
import polars as pl
from tqdm import tqdm
import sys
output_path=sys.argv[1]
feature_path=sys.argv[2]

test_label=pd.read_feather(feature_path+'test_label.feather')
test_label=pl.DataFrame(test_label)
test_label=test_label.select(['User_ID','Item_ID'])


print('*'*20,'合并user_item_cnt计算','*'*20)
res=[]
for part in tqdm(range(5)):
    tmp_df_00=pd.read_feather(output_path+str(part)+'_user_item_cnt.feather')
    tmp_df_00=pl.DataFrame(tmp_df_00)
    res.append(tmp_df_00)
tmp_df_00=pl.concat(res)
tmp_df_00=tmp_df_00.groupby(['User_ID','Item_ID','Item_Category','日期','Behavior_Type']).sum()
tmp_df_00=tmp_df_00.rename({'count':'cnt'})
tmp_df_00=tmp_df_00.select(['User_ID','Item_ID','Item_Category','日期','Behavior_Type','cnt'])

# 按照商品聚合，可以只取子集部分
tmp_df_01=tmp_df_00.select(['Item_ID','日期','Behavior_Type','cnt']).groupby(['Item_ID','日期','Behavior_Type']).sum()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+'item_cnt.feather')
# 过滤用户商品，只取子集部分
tmp_df_00=tmp_df_00.join(test_label,how='inner',on=['User_ID','Item_ID'])
tmp_df_00=tmp_df_00.to_pandas()
tmp_df_00.to_feather(output_path+'user_item_cnt.feather')
print('*'*20,'合并user_item_cnt完成','*'*20)



