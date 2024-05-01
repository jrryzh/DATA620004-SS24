# FUDAN_NNLecture_ss24

该repo由四个文件夹组成，分别为ckpts, data, results 和 scripts.

ckpts: 存放模型的权重文件

data: 存放数据集

results: 存放模型可视化结果和训练结果

scripts: 存放模型、数据处理、训练和测试脚本
- model.py: 模型定义
- train.py: 模型训练脚本
- test.py: 模型测试脚本

其中train.py中包含了模型训练的主要逻辑，包括数据加载、模型定义、优化器定义、损失函数定义、训练循环、模型保存等。test.py中包含了模型测试的主要逻辑，包括数据加载、模型加载、模型评估、模型可视化等。

主要对模型定义、训练和测试的脚本示例如下：
```python
# 初始化模型
model = MLPModel(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10, activate_func=activate_func)
# 训练模型
model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=128, initial_learning_rate=1e-4, reg_lambda=0.01, decay_factor=0.9)
# 测试模型
model.test(X_test, y_test)
```

测试模型的示例如下：
```python
# 加载模型
model = MLPModel(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10, activate_func=activate_func)
model.load_model(path='ckpts/model.pth')
# 测试模型
model.test(X_test, y_test)
```
可以根据自己的需求修改模型定义、训练和测试的脚本。

### 运行脚本前的准备工作
1. 安装依赖库numpy. matplotlib
2. 下载t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz, train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz四个数据集文件，并放到data文件夹下
3. 运行train.py脚本，训练模型，并将模型权重保存到ckpts文件夹下
```bash
cd scripts
python train.py
```

4. 选择测试模型，运行test，并将模型评估结果保存到results文件夹下
```bash
cd scripts
python test.py
```