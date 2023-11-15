import pandas as pd
import polars as pl
from tqdm import tqdm
import sys
output_path=sys.argv[1]

test_label_tmp2=pd.read_feather(output_path+'test_label_category.feather')
test_label_tmp2=pl.DataFrame(test_label_tmp2)

print('*'*20,'合并用户品类first计算','*'*20)
res=[]
for part in tqdm(range(5)):
    tmp_df_01=pd.read_feather(output_path+str(part)+'_user_category_first_time.feather')
    tmp_df_01=pl.DataFrame(tmp_df_01)
    tmp_df_01=tmp_df_01.join(test_label_tmp2,how='inner',on=['User_ID','Item_Category'])
    res.append(tmp_df_01)
tmp_df_01=pl.concat(res)
tmp_df_01=tmp_df_01.groupby(['User_ID','Item_Category','Behavior_Type']).min()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+'user_category_first_time.feather')
print('*'*20,'合并用户品类first完成','*'*20)

print('*'*20,'合并用户品类last计算','*'*20)
res=[]
for part in tqdm(range(5)):
    tmp_df_01=pd.read_feather(output_path+str(part)+'_user_category_last_time.feather')
    tmp_df_01=pl.DataFrame(tmp_df_01)
    tmp_df_01 = tmp_df_01.join(test_label_tmp2, how='inner', on=['User_ID', 'Item_Category'])
    res.append(tmp_df_01)
tmp_df_01=pl.concat(res)
tmp_df_01=tmp_df_01.groupby(['User_ID','Item_Category','Behavior_Type']).max()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+'user_category_last_time.feather')
print('*'*20,'合并用户品类last完成','*'*20)

