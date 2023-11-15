import polars as pl
import sys


part=sys.argv[1]# 第0个参数是.py文件
raw_data_path=sys.argv[2]
output_path=sys.argv[3]


print('*'*20,f'part{part}开始计算','*'*20)
round2_user_0=pl.read_csv(raw_data_path+"round2_user_"+str(part)+".txt",separator='\t',has_header=False,low_memory=True,new_columns=['User_ID','Item_ID','Behavior_Type','User_Geohash','Item_Category','Time'])
round2_user_0=round2_user_0.select(['User_ID','Item_ID','Behavior_Type','Item_Category','Time'])
round2_user_0=round2_user_0.with_columns(
    [
        pl.col("Time")
        .str.split_exact(" ",n=1)
        .struct.rename_fields(["日期", "小时"])
        .alias("fields"),
    ]
).unnest("fields")
round2_user_0=round2_user_0.select(['User_ID','Item_ID','Behavior_Type','Item_Category','Time','日期'])

#--------------把需要全量数据的部分提前在这里计算，后面再合并5个文件--------------------------
# 计算用户统计
tmp_df_01=round2_user_0.select(['User_ID','日期','Behavior_Type','Time']).groupby(['User_ID','日期','Behavior_Type']).count()
tmp_df_01=tmp_df_01.to_pandas(use_pyarrow_extension_array=True)
tmp_df_01.to_feather(output_path+str(part)+'_user_cnt.feather')
# 计算用户活跃的最后一个小时
# # tmp_df_01=round2_user_0.select(['User_ID','Time']).groupby(['User_ID']).max()
# tmp_df_01=round2_user_0.select(['User_ID','Behavior_Type','Time']).groupby(['User_ID','Behavior_Type']).max()
# tmp_df_01=tmp_df_01.to_pandas(use_pyarrow_extension_array=True)
# # tmp_df_01.to_feather(output_path+str(part)+'_user_last_time.feather')
# tmp_df_01.to_feather(output_path+str(part)+'_user_last_time_v2.feather')
# 后五天用户商品计数，用户计算后五天衰减值的排序特征
user_item_last5_cnt=round2_user_0.filter(pl.col('日期')>'2014-12-13')
user_item_last5_cnt=user_item_last5_cnt.groupby(['User_ID','Item_ID','Item_Category','Behavior_Type','日期']).count()
user_item_last5_cnt_feather=user_item_last5_cnt.to_pandas(use_pyarrow_extension_array=True)
user_item_last5_cnt_feather[['User_ID','Item_ID','Behavior_Type','日期','count']].to_feather(output_path+str(part)+'_user_item_last5_cnt.feather')
user_item_last5_cnt=user_item_last5_cnt.select(['User_ID','Item_ID','Item_Category','日期'])
#--------------把子集用户品类的部分取出来，后面再合并5个文件--------------------------
# 读取商品子集
part2_item=pl.read_csv(raw_data_path+"part2_item.txt",separator='\t',has_header=False)
part2_item=part2_item.rename({'column_1':'Item_ID','column_2':'Item_Geohash','column_3':'Item_Category'}).select(['Item_ID','Item_Category'])
part2_item_category=part2_item.groupby(['Item_Category']).count().select(['Item_Category'])
# 只取子集品类部分计算
round2_user_0=round2_user_0.join(part2_item_category,how='inner',on=['Item_Category'])
# 计算品类统计
tmp_df_01=round2_user_0.select(['Item_Category','日期','Behavior_Type','Time']).groupby(['Item_Category','日期','Behavior_Type']).count()
tmp_df_01=tmp_df_01.to_pandas(use_pyarrow_extension_array=True)
tmp_df_01.to_feather(output_path+str(part)+'_itemcategory_cnt.feather')
# 近五天里用户品类对
user_item_last5_cnt=user_item_last5_cnt.join(part2_item_category,how='inner',on=['Item_Category'])
test_label_category_tmp1=user_item_last5_cnt.groupby(['User_ID','Item_ID','Item_Category']).min()
test_label_category_tmp2=test_label_category_tmp1.groupby(['User_ID','Item_Category']).min()
test_label_category_tmp2=test_label_category_tmp2.to_pandas(use_pyarrow_extension_array=True)
test_label_category_tmp2.to_feather(output_path+str(part)+'_test_label_category.feather')
# 计算用户品类统计
tmp_df_01=round2_user_0.select(['User_ID','Item_Category','日期','Behavior_Type','Time']).groupby(['User_ID','Item_Category','日期','Behavior_Type']).count()
tmp_df_01=tmp_df_01.to_pandas(use_pyarrow_extension_array=True)
tmp_df_01.to_feather(output_path+str(part)+'_user_category_cnt.feather')
# 用户品类第一次相关特征
tmp_df_02=round2_user_0.select(['User_ID','Item_Category','Behavior_Type','Time']).groupby(['User_ID','Item_Category','Behavior_Type']).min()
tmp_df_02=tmp_df_02.to_pandas(use_pyarrow_extension_array=True)
tmp_df_02.to_feather(output_path+str(part)+'_user_category_first_time.feather')
# 用户品类最后一次相关特征
tmp_df_03=round2_user_0.select(['User_ID','Item_Category','Behavior_Type','Time']).groupby(['User_ID','Item_Category','Behavior_Type']).max()
tmp_df_03=tmp_df_03.to_pandas(use_pyarrow_extension_array=True)
tmp_df_03.to_feather(output_path+str(part)+'_user_category_last_time.feather')

#--------------把子集用户商品的部分取出来，后面再合并5个文件--------------------------
# 只取子集品类部分计算
round2_user_0=round2_user_0.join(part2_item.select(['Item_ID']),how='inner',on=['Item_ID'])
# 用户商品对计数,用户后续构建用户和商品和品类的统计
tmp_df_01=round2_user_0.select(['User_ID','Item_ID','Item_Category','日期','Behavior_Type','Time']).groupby(['User_ID','Item_ID','Item_Category','日期','Behavior_Type']).count()
tmp_df_01=tmp_df_01.to_pandas(use_pyarrow_extension_array=True)
tmp_df_01.to_feather(output_path+str(part)+'_user_item_cnt.feather')
# 近五天里用户商品对
test_label_category_tmp1=test_label_category_tmp1.join(part2_item.select(['Item_ID']),how='inner',on=['Item_ID'])
tmp_df_01=test_label_category_tmp1.groupby(['User_ID','Item_ID']).min()
tmp_df_01=tmp_df_01.to_pandas(use_pyarrow_extension_array=True)
tmp_df_01.to_feather(output_path+str(part)+'_test_label.feather')
# # 用户商品第一次相关特征
# tmp_df_02=round2_user_0.select(['User_ID','Item_ID','Item_Category','Behavior_Type','Time']).groupby(['User_ID','Item_ID','Item_Category','Behavior_Type']).min()
# tmp_df_02=tmp_df_02.to_pandas(use_pyarrow_extension_array=True)
# tmp_df_02.to_feather(output_path+str(part)+'_user_item_first_time.feather')
# 用户商品最后一次相关特征
tmp_df_03=round2_user_0.select(['User_ID','Item_ID','Item_Category','Behavior_Type','Time']).groupby(['User_ID','Item_ID','Item_Category','Behavior_Type']).max()
tmp_df_03=tmp_df_03.to_pandas(use_pyarrow_extension_array=True)
tmp_df_03.to_feather(output_path+str(part)+'_user_item_last_time.feather')

print('*'*20,f'part{part}计算完成','*'*20)