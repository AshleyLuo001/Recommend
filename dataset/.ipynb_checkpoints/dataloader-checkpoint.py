import polars as pl
import sys
part=sys.argv[1]# 第0个参数是.py文件
raw_data_path=sys.argv[2]
output_path=sys.argv[3]
# print('*'*20,f'part{part}开始计算','*'*20)
round2_user_0=pl.read_csv(raw_data_path+"round2_user_"+str(part)+".txt",separator='\t',has_header=False)
round2_user_0=round2_user_0.rename({'column_1':'User_ID','column_2':'Item_ID','column_3':'Behavior_Type','column_4':'User_Geohash','column_5':'Item_Category','column_6':'Time'})
round2_user_0=round2_user_0.with_columns((round2_user_0['Time'].apply(lambda x:x.split(' ')[0])).alias("日期"))

#--------------把需要全量数据的部分提前在这里计算，后面再合并5个文件--------------------------
# 计算用户统计
tmp_df_01=round2_user_0.select(['User_ID','日期','Behavior_Type','Time']).groupby(['User_ID','日期','Behavior_Type']).count()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+str(part)+'_user_cnt.feather')
# 计算用户geo数
# tmp_df_01=round2_user_0.select(['User_ID','User_Geohash','Behavior_Type']).groupby(['User_ID','User_Geohash']).count()
# tmp_df_01=tmp_df_01.to_pandas()
# tmp_df_01.to_feather(output_path+str(part)+'_user_geo_cnt.feather')
# 计算品类统计
tmp_df_01=round2_user_0.select(['Item_Category','日期','Behavior_Type','Time']).groupby(['Item_Category','日期','Behavior_Type']).count()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+str(part)+'_itemcategory_cnt.feather')
# 计算用户活跃的最后一个小时
tmp_df_01=round2_user_0.select(['User_ID','Time']).groupby(['User_ID']).max()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+str(part)+'_user_last_time.feather')
# 近五天里用户商品对里的近十天统计
tmp_df_01=round2_user_0.filter(pl.col("日期") >'2014-12-13').select(['User_ID','Item_ID','Item_Category','Time'])
tmp_df_01=tmp_df_01.groupby(['User_ID','Item_ID','Item_Category']).min()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+str(part)+'_user_item_last5_cnt_list.feather')

tmp_df_02=round2_user_0.filter(pl.col("日期") >='2014-12-09').select(['User_ID','Item_ID','Item_Category','日期','Behavior_Type','Time'])
tmp_df_02=tmp_df_02.groupby(['User_ID','Item_ID','Item_Category','日期','Behavior_Type']).count()
tmp_df_02=tmp_df_02.to_pandas()
tmp_df_02.to_feather(output_path+str(part)+'_user_item_last5_cnt.feather')
# all_item
tmp_df_01=round2_user_0.select(['Item_ID','Item_Category','Time']).groupby(['Item_ID','Item_Category']).min()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+str(part)+'_all_item.feather')
#--------------把子集用户品类的部分取出来，后面再合并5个文件--------------------------
# 读取商品子集
part2_item=pl.read_csv(raw_data_path+"part2_item.txt",separator='\t',has_header=False)
part2_item=part2_item.rename({'column_1':'Item_ID','column_2':'Item_Geohash','column_3':'Item_Category'}).select(['Item_ID','Item_Category'])
part2_item_category=part2_item.groupby(['Item_Category']).count().select(['Item_Category'])
# 只取子集品类部分计算
round2_user_0=round2_user_0.join(part2_item_category,how='inner',on=['Item_Category'])
# 计算用户品类统计
tmp_df_01=round2_user_0.select(['User_ID','Item_Category','日期','Behavior_Type','Time']).groupby(['User_ID','Item_Category','日期','Behavior_Type']).count()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+str(part)+'_user_category_cnt.feather')
# 用户品类第一次相关特征
tmp_df_02=round2_user_0.select(['User_ID','Item_Category','Behavior_Type','Time']).groupby(['User_ID','Item_Category','Behavior_Type']).min()
tmp_df_02=tmp_df_02.to_pandas()
tmp_df_02.to_feather(output_path+str(part)+'_user_category_first_time.feather')
# 用户品类最后一次相关特征
tmp_df_03=round2_user_0.select(['User_ID','Item_Category','Behavior_Type','Time']).groupby(['User_ID','Item_Category','Behavior_Type']).max()
tmp_df_03=tmp_df_03.to_pandas()
tmp_df_03.to_feather(output_path+str(part)+'_user_category_last_time.feather')

#--------------把子集用户商品的部分取出来，后面再合并5个文件--------------------------
# 只取子集品类部分计算
round2_user_0=round2_user_0.join(part2_item.select(['Item_ID']),how='inner',on=['Item_ID'])
# 用户商品对计数,用户后续构建用户和商品和品类的统计
tmp_df_01=round2_user_0.select(['User_ID','Item_ID','Item_Category','日期','Behavior_Type','Time']).groupby(['User_ID','Item_ID','Item_Category','日期','Behavior_Type']).count()
tmp_df_01=tmp_df_01.to_pandas()
tmp_df_01.to_feather(output_path+str(part)+'_user_item_cnt.feather')
# 用户商品第一次相关特征
tmp_df_02=round2_user_0.select(['User_ID','Item_ID','Item_Category','Behavior_Type','Time']).groupby(['User_ID','Item_ID','Item_Category','Behavior_Type']).min()
tmp_df_02=tmp_df_02.to_pandas()
tmp_df_02.to_feather(output_path+str(part)+'_user_item_first_time.feather')
# 用户商品最后一次相关特征
tmp_df_03=round2_user_0.select(['User_ID','Item_ID','Item_Category','Behavior_Type','Time']).groupby(['User_ID','Item_ID','Item_Category','Behavior_Type']).max()
tmp_df_03=tmp_df_03.to_pandas()
tmp_df_03.to_feather(output_path+str(part)+'_user_item_last_time.feather')