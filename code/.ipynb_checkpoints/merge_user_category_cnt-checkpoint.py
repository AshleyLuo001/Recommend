import pandas as pd
import polars as pl
from tqdm import tqdm
import sys
output_path=sys.argv[1]

test_label_category=pd.read_feather(output_path+'test_label_category.feather')
test_label_category=pl.DataFrame(test_label_category)


print('*'*20,'合并user_category_cnt计算','*'*20)
res=[]
for part in tqdm(range(5)):
    tmp_df_01=pd.read_feather(output_path+str(part)+'_user_category_cnt.feather')
    tmp_df_01=pl.DataFrame(tmp_df_01)
    tmp_df_01 = tmp_df_01.join(test_label_category, how='inner', on=['User_ID', 'Item_Category'])
    res.append(tmp_df_01)
tmp_df_01=pl.concat(res)
tmp_df_01=tmp_df_01.groupby(['User_ID','Item_Category','日期','Behavior_Type']).sum()
tmp_df_01=tmp_df_01.rename({'count':'cnt'})
tmp_df_01=tmp_df_01.select(['User_ID','Item_Category','日期','Behavior_Type','cnt'])
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+'user_category_cnt.feather')
print('*'*20,'合并user_category_cnt完成','*'*20)