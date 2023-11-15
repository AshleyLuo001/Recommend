import pandas as pd
import polars as pl
from tqdm import tqdm
import sys

raw_data_path=sys.argv[1]
input_path=sys.argv[2]
output_path=sys.argv[3]

res=[]
for part in tqdm(range(5)):
    tmp_df_03=pd.read_feather(input_path+str(part)+'_test_label_category.feather')
    tmp_df_03=pl.DataFrame(tmp_df_03)
    res.append(tmp_df_03)
tmp_df_03=pl.concat(res)
tmp_df_04=tmp_df_03.groupby(['User_ID','Item_Category']).min().select(['User_ID','Item_Category'])
tmp_df_04=tmp_df_04.to_pandas(use_pyarrow_extension_array=True)
tmp_df_04.to_feather(input_path+'test_label_category.feather')