import pandas as pd
import polars as pl
from tqdm import tqdm
import sys

raw_data_path=sys.argv[1]
input_path=sys.argv[2]
output_path=sys.argv[3]

part2_item=pl.read_csv(raw_data_path+"part2_item.txt",separator='\t',has_header=False)
part2_item=part2_item.rename({'column_1':'Item_ID','column_2':'Item_Geohash','column_3':'Item_Category'}).select(['Item_ID','Item_Category'])
part2_item_item=set(part2_item['Item_ID'])
part2_item_category=set(part2_item['Item_Category'])

res=[]
for part in tqdm(range(5)):
    tmp_df_03=pd.read_feather(input_path+str(part)+'_user_item_last5_cnt_list.feather')
    tmp_df_03=pl.DataFrame(tmp_df_03)
    res.append(tmp_df_03)
tmp_df_03=pl.concat(res)
tmp_df_03=tmp_df_03.groupby(['User_ID','Item_ID','Item_Category']).min().select(['User_ID','Item_ID','Item_Category'])
tmp_df_03=tmp_df_03.to_pandas()
tmp_df_03.to_feather(input_path+'test_label_all.feather')
tmp_df_04=tmp_df_03[tmp_df_03['Item_Category'].isin(part2_item_category)].copy()
tmp_df_04=tmp_df_04[['User_ID','Item_Category']].drop_duplicates(subset=None, keep='first', inplace=False).copy()
tmp_df_04=tmp_df_04.reset_index(drop=True)
tmp_df_04.to_feather(input_path+'test_label_category.feather')
tmp_df_05=tmp_df_03[tmp_df_03['Item_ID'].isin(part2_item_item)].copy()
tmp_df_05=tmp_df_05.reset_index(drop=True)
tmp_df_05.to_feather(output_path+'test_label.feather')

