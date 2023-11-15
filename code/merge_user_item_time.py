import pandas as pd
import polars as pl
from tqdm import tqdm
import sys
output_path=sys.argv[1]
feature_path=sys.argv[2]

test_label=pd.read_feather(feature_path+'test_label.feather')
test_label=pl.DataFrame(test_label[['User_ID','Item_ID']])

# print('*'*20,'合并用户商品first计算','*'*20)
# res=[]
# for part in tqdm(range(5)):
#     tmp_df_01=pd.read_feather(output_path+str(part)+'_user_item_first_time.feather')
#     tmp_df_01=pl.DataFrame(tmp_df_01)
#     tmp_df_01 = tmp_df_01.join(test_label, how='inner', on=['User_ID', 'Item_ID'])
#     res.append(tmp_df_01)
# tmp_df_01=pl.concat(res)
# tmp_df_01=tmp_df_01.groupby(['User_ID','Item_ID','Item_Category','Behavior_Type']).min()
# tmp_df_01=tmp_df_01.to_pandas()
# tmp_df_01.to_feather(output_path+'user_item_first_time.feather')
# print('*'*20,'合并用户商品first完成','*'*20)

print('*'*20,'合并用户商品last计算','*'*20)
res=[]
for part in tqdm(range(5)):
    tmp_df_01=pd.read_feather(output_path+str(part)+'_user_item_last_time.feather')
    tmp_df_01=pl.DataFrame(tmp_df_01)
    tmp_df_01 = tmp_df_01.join(test_label, how='inner', on=['User_ID', 'Item_ID'])
    res.append(tmp_df_01)
tmp_df_01=pl.concat(res)
tmp_df_01=tmp_df_01.groupby(['User_ID','Item_ID','Item_Category','Behavior_Type']).max()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+'user_item_last_time.feather')
print('*'*20,'合并用户商品last完成','*'*20)
