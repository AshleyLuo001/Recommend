FROM registry.cn-shanghai.aliyuncs.com/1598521844/aicas_2023:v39.1

# 预分配200GB存储空间
RUN fallocate -l 150G /data/
