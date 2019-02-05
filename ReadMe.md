### 代码结构

```
|- matrix.hpp.in 通过make指定编译cpu还是gpu版本自动生成matrix.hpp头文件
|- include
    |- data_loader.hpp 对语料库文件进行加载提取
    |- model.hpp 保存参数，持久化模型，支持inference
    |- optimizer.hpp 提供优化方法，针对模型和数据集学习参数
    |- optimizer 
    	|- sgd_optimizer.hpp 基于sgd的优化算法
    |- evaluator.hpp 评估指标
    |-cpu
        |- cpu_mat.hpp 基于cpu的矩阵
    	|- mat_factory.hpp 生成并初始化基于cpu的矩阵
|- src 提供include的实现代码
    |-cpu
        |- cpu_mat.cpp
	|- cpu_mat_factory.cpp
    |-cuda
        |- gpu_mat.cu
	|- gpu_mat_factory.cu
    |- optimizer
        |- sgd_optimizer.cpp
    |- train.cpp
    |- test.cpp
    |- model.cpp
    |- data_loader.cpp
    |- acc_evaluator.cpp
|- test 
    |- test_mat.cpp 测试矩阵库
|- tools
    |- data_process.py 对文本分类的原始数据进行处理，生成供后续程序使用的featurefile, train.dat, val.dat, test.dat几个文件
```

#### 设计原则

原则接口抽象和实现分离：
+ 利用matrix.hpp.in在进行编译时选择相应的cpu或gpu的矩阵计算（如矩阵乘法），可以实现不同计算平台的扩展，相应平台的matrix由相应平台的factory产生
+ 支持不同的optimizer扩展


### 依赖

+ google test(https://github.com/google/googletest)
	+ version:1.8.0
	+ 用于单元测试

+ eigen
	+ version 3.3.4-4
	+ 用于cpu的矩阵计算加速

+ 验证平台ubuntu 18.04

### 入口函数

+ 训练
	+ train.cpp
+ 预测
	+ test.cpp
+ 数据预处理
	data_process.py
### 后续优化 

+ 特征工程，更好的文本加工
	+ 单词重要性赋权（如tf-idf）
	+ 去掉不常用词
	+ 对英文单词做标准化
+ 目前的层次化分类是人肉标注的，可以利用层次聚类来完成类的层次划分
+ 支持数据读取和计算的pipline
+ 支持cuda blas
+ 支持python api
+ 学习下facebook实现
