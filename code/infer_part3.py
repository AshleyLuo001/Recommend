import pandas as pd
import sys
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
path = sys.argv[1]
infer_df_path = path + 'infer_df_v40/'


infer_list=[]
for seed in range(50):
    infer_df_tmp=pd.read_feather(infer_df_path+'infer_df_'+str(seed)+'.feather')
    infer_list.append(infer_df_tmp)
infer_df = pd.concat(infer_list)
infer_df=infer_df.groupby(['User_ID', 'Item_ID'])['pred'].sum().reset_index()
infer_df['pred']=infer_df['pred']/50

yuzhi_cnt = 38000
l1 = []
for i in tqdm(range(500,2500)) :
    cnt = len(infer_df[infer_df['pred'] > i / 500])
    l1.append([i / 500, abs(yuzhi_cnt - cnt)])
tmp_df_01 = pd.DataFrame(l1)
tmp_df_01.columns = ['阈值', '计数差']
tmp_df_01 = tmp_df_01.sort_values(by=['计数差'], ascending=[True])
tmp_df_01 = tmp_df_01.reset_index(drop=True)
yuzhi = tmp_df_01['阈值'][0]
infer_df = infer_df[infer_df['pred'] > yuzhi]
print(yuzhi)
print(infer_df.shape)
assert len(infer_df) > 0
if os.path.exists('/data/final/') :
    infer_df[['User_ID', 'Item_ID']].to_csv('/data/test/eval_script_v1/eval/submit.txt', index=False, sep='\t',
                                            header=None)
infer_df[['User_ID', 'Item_ID']].to_csv('./result.txt', index=False, sep='\t', header=None)
infer_df[['User_ID', 'Item_ID']].to_csv('./code/result.txt', index=False, sep='\t', header=None)

