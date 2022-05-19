## CVPR 22 NAS workshop 竞赛
### 最终排名
第四名
### 竞赛链接
https://aistudio.baidu.com/aistudio/competition/detail/150/0/introduction

### 文档
* base_component: 该文件夹包括了一些可行的算法
  * Bagging
  * RankSVM
  * 基于Paddle-Paddle/PyTorch的RankNet
  * Stacking-XGBoost/CatBoost
* utils: 工具类
* training_prediction.py: 训练框架
* analysis：分析工具

### 数据说明
* 三个超参数，网络深度，注意力头数，膨胀系数
* 其中膨胀系数是指Transformer中MLP的膨胀系数