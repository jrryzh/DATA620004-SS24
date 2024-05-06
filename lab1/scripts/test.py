from dataloader import load_mnist
from model import MLPModel  
import numpy as np

def grid_search():
    # 定义超参数的网格搜索
    learning_rates = [1e-3, 1e-4, 1e-5]
    reg_lambdas = [0.01, 0.1, 1.0]
    batch_sizes = [32, 64, 128]
    hidden_sizes1 = [128, 256, 512]
    hidden_sizes2 = [64, 128, 256]
    activate_funcs = ['sigmoid', 'tanh', 'relu']
    # 遍历网格搜索的所有组合
    for lr in learning_rates:
        for reg in reg_lambdas:
            for bs in batch_sizes:
                for hs1 in hidden_sizes1:
                    for hs2 in hidden_sizes2:
                        for activate_func in activate_funcs:
                            # 初始化模型
                            model = MLPModel(input_size=784, hidden_size1=hs1, hidden_size2=hs2, output_size=10, activate_func=activate_func)
                            # 训练模型
                            model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=bs, initial_learning_rate=lr, reg_lambda=reg, decay_factor=0.9)
                            # 测试模型
                            model.test(X_test, y_test)

if __name__ == '__main__':
    # Load data and create model
    X_test, y_test = load_mnist('/Users/jrryzh/Documents/lectures/神经网络/lab1/data/', kind='t10k')
    
    # 对数据集进行归一化
    X_test = X_test - np.mean(X_test) / np.std(X_test)

    # 初始化模型
    model = MLPModel(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10, activate_func='tanh')
    # 加载模型
    model.load_model('/Users/jrryzh/Documents/lectures/神经网络/lab1/models/mlp_model.pkl')
    # 测试模型
    model.test(X_test, y_test)