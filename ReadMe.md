### 代码建设中
### 代码结构

```
|- matrix.hpp.in 通过make指定编译cpu还是gpu版本
|- include
    |- mat_factory.hpp
    |- data_loader.hpp
    |- model.hpp
    |- optimizer.hpp
    |- evaluator.hpp
    |-cpu
        |- cpu_mat.hpp
|- src
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
|- tools
    |- text2feature.py 
    |- hierarchical_label_builder.py
    |- train_val_test.py
```

#### 设计原则

原则接口抽象和实现分离：
+ 利用matrix.hpp接口分别实现基于cpu和gpu的矩阵计算（如矩阵乘法），可以实现不同计算平台的扩展，相应平台的matrix由相应平台的factory产生
+ 支持不同的optimizer扩展


### 依赖

+ google test(https://github.com/google/googletest)
	+ version:1.8.0
	+ 用于单元测试

+ eigen
	+ version 3.3.4-4
	+ 用于cpu的矩阵计算加速

+ 验证平台ubuntu 18.04

### 语料库

http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews

https://www.sogou.com/labs/resource/cs.php

http://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.html

### 入口函数

+ 训练
	+ train.cpp
+ 预测
	+ test.cpp

### 后续优化 

+ 特征工程，更好的文本加工
	+ 过滤停用词
	+ 单词重要性赋权（如tf-idf）
	+ 去掉不常用词
	+ 对英文单词做标准化
+ 利用层次聚类来完成类的层次划分
+ 支持数据读取和计算的pipline
+ 尝试更大data set
+ 支持cuda blas
+ 支持python api
