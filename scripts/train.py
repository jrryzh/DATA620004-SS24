from dataloader import load_mnist
from model import MLPModel  
import numpy as np

def grid_search():
    # 定义超参数的网格搜索
    learning_rates = [1e-3, 1e-4, 1e-5]
    reg_lambdas = [0.01, 0.1]
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
                            print('lr: {}, reg: {}, bs: {}, hs1: {}, hs2: {}, activate_func: {}'.format(lr, reg, bs, hs1, hs2, activate_func))
                            # 初始化模型
                            model = MLPModel(input_size=784, hidden_size1=hs1, hidden_size2=hs2, output_size=10, activate_func=activate_func)
                            # 训练模型
                            model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=bs, initial_learning_rate=lr, reg_lambda=reg, decay_factor=0.9)
                            # 测试模型
                            model.test(X_test, y_test)
                            

if __name__ == '__main__':
    # Load data and create model
    X, y = load_mnist('/Users/jrryzh/Documents/lectures/神经网络/lab1/data/', kind='train')
    X_test, y_test = load_mnist('/Users/jrryzh/Documents/lectures/神经网络/lab1/data/', kind='t10k')

    # 假设 X_train 和 y_train 已经被正确加载
    num_training = int(X.shape[0] * 0.8)  # 计算80%的位置

    # 划分训练集和验证集
    X_train = X[:num_training]
    y_train = y[:num_training]
    X_val = X[num_training:]
    y_val = y[num_training:]
    
    # 对数据集进行归一化
    X_train = X_train - np.mean(X_train) / np.std(X_train)
    X_val = X_val - np.mean(X_val) / np.std(X_val)
    X_test = X_test - np.mean(X_test) / np.std(X_test)

    # # 初始化模型
    # model = MLPModel(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10, activate_func='tanh')
    # # 训练模型
    # model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=256, initial_learning_rate=1e-5, reg_lambda=0.01, decay_factor=0.9)
    # # 测试模型
    # model.test(X_test, y_test)
    
    # 网格搜索  
    grid_search()