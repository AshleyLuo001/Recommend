import pandas as pd
import sys
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings("ignore")
path = sys.argv[1]
seed = sys.argv[2]
test_df_path = path + 'test_df_v40/'
infer_df_path = path + 'infer_df_v40/'


print('seed:',seed,'开始')
model_xgb = joblib.load('./weight/model_'+seed+'.pkl')
feat_cols = joblib.load('./weight/feat_cols.pkl')
test_df_tmp = pd.read_feather(test_df_path+'test_df.feather')

xgb_test = xgb.DMatrix(test_df_tmp[feat_cols])
# xgb_test = joblib.load(test_df_path+'xgb_test.pkl')
test_df_tmp['pred'] = model_xgb.predict(xgb_test)
test_df_tmp = test_df_tmp[['User_ID', 'Item_ID', 'pred']].copy()
print(test_df_tmp.shape)
infer_df = test_df_tmp[['User_ID', 'Item_ID', 'pred']]
infer_df.to_feather(infer_df_path+'infer_df_'+seed+'.feather')
print('seed:',seed,'完成')