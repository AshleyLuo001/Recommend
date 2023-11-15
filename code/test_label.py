import pandas as pd
import polars as pl
from tqdm import tqdm
import sys

input_path=sys.argv[1]
output_path=sys.argv[2]

res=[]
for part in tqdm(range(5)):
    tmp_df_03=pd.read_feather(input_path+str(part)+'_test_label.feather')
    tmp_df_03=pl.DataFrame(tmp_df_03)
    res.append(tmp_df_03)
tmp_df_03=pl.concat(res)
tmp_df_03=tmp_df_03.groupby(['User_ID','Item_ID','Item_Category']).min().select(['User_ID','Item_ID','Item_Category'])
tmp_df_03=tmp_df_03.to_pandas(use_pyarrow_extension_array=True)
tmp_df_03.to_feather(output_path+'test_label.feather')