## CVPR 22 NAS workshop 竞赛
### 竞赛链接
https://aistudio.baidu.com/aistudio/competition/detail/150/0/introduction

### 当前思路
* 基于XGBoost调参

### 文档
* base_component: 该文件夹包括了一些可行的算法
  * Bagging
  * RankSVM
  * Rank Decision Tree（尚未完成）
  * MTL Decision Tree （尚未完成）
* utils: 工具类
* xgb_prediction: 基于XGBoost的训练框架

### 潜在可行的思路
* 集成学习
* 特征工程
* 数据增强
  * 三个超参数，网络深度，注意力头数，膨胀系数
  * 其中膨胀系数是指Transformer中MLP的膨胀系数