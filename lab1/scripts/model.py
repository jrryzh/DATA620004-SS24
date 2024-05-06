import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)

class MLPModel:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activate_func='relu'):
        # 初始化神经网络参数
        # input_size: 输入层大小
        # hidden_size1: 第一个隐藏层大小
        # hidden_size2: 第二个隐藏层大小
        # output_size: 输出层大小
        self.params = {
            'W1': np.random.randn(input_size, hidden_size1) * np.sqrt(1. / input_size),
            'b1': np.zeros(hidden_size1),
            'W2': np.random.randn(hidden_size1, hidden_size2) * np.sqrt(1. / hidden_size1),
            'b2': np.zeros(hidden_size2),
            'W3': np.random.randn(hidden_size2, output_size) * np.sqrt(1. / hidden_size2),
            'b3': np.zeros(output_size)
        }
        self.activate_func = activate_func  # 激活函数
        
    def load(self, file_path):
        # 从指定的文件路径加载模型参数
        with open(file_path, 'rb') as file:
            self.params = pickle.load(file)
    
    def save(self, file_path, best_params):
        # 保存最优模型参数至文件
        with open(file_path, 'wb') as f:
            pickle.dump(best_params, f)
        
    def relu(self, z):
        # ReLU激活函数
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        # ReLU函数的导数
        return (z > 0).astype(z.dtype)  # 返回ReLU函数的导数
    
    def sigmoid(self, z):
        # Sigmoid激活函数
        return 1. / (1. + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        # Sigmoid函数的导数
        return self.sigmoid(z) * (1. - self.sigmoid(z))
    
    def tanh(self, z):
        # Tanh激活函数
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        # Tanh函数的导数
        return 1. - np.tanh(z) ** 2
    
    def softmax(self, z):
        # Softmax激活函数
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / e_z.sum(axis=1, keepdims=True)

    def cross_entropy_derivative(self, p, y):
        # p是模型输出，y是真实标签的类别索引
        y_true = np.zeros_like(p)
        y_true[np.arange(len(y)), y] = 1
        return p - y_true  # 使用独热编码的y进行广播

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]  # 样本数量
        epsilon = 1e-9  # 防止对数为负无穷，添加一个小常数
        log_likelihood = -np.log(y_pred[range(m), y_true] + epsilon)  # 使用加上epsilon的y_pred计算对数似然
        loss = np.sum(log_likelihood) / m  # 计算平均损失
        return loss
    
    def forward(self, X):
        # 第一层计算
        self.Z1 = X.dot(self.params['W1']) + self.params['b1']  # 计算第一层线性加权和
        if self.activate_func =='relu':
            self.A1 = self.relu(self.Z1)  # 使用ReLU激活函数
        elif self.activate_func =='sigmoid':
            self.A1 = self.sigmoid(self.Z1)  # 使用Sigmoid激活函数
        elif self.activate_func == 'tanh':
            self.A1 = self.tanh(self.Z1)  # 使用Tanh激活函数
        # 第二层计算
        self.Z2 = self.A1.dot(self.params['W2']) + self.params['b2']  # 计算第二层线性加权和
        if self.activate_func =='relu':
            self.A2 = self.relu(self.Z2)  # 使用ReLU激活函数
        elif self.activate_func =='sigmoid':
            self.A2 = self.sigmoid(self.Z2)  # 使用Sigmoid激活函数
        elif self.activate_func == 'tanh':
            self.A2 = self.tanh(self.Z2)  # 使用Tanh激活函数
        # 输出层计算
        self.Z3 = self.A2.dot(self.params['W3']) + self.params['b3']  # 计算第三层线性加权和
        self.A3 = self.softmax(self.Z3)  # 使用softmax激活函数
        return self.A3  # 返回输出结果
    
    def backward(self, X, y, learning_rate, reg_lambda):
        # 输出层误差
        delta3 = self.cross_entropy_derivative(self.A3, y)
        # 输出层权重和偏置的梯度
        dW3 = self.A2.T.dot(delta3) + reg_lambda * self.params['W3']
        db3 = np.sum(delta3, axis=0)
        # 第二隐藏层误差
        if self.activate_func =='relu':
            delta2 = delta3.dot(self.params['W3'].T) * self.relu_derivative(self.Z2)
        elif self.activate_func =='sigmoid':
            delta2 = delta3.dot(self.params['W3'].T) * self.sigmoid_derivative(self.Z2)
        elif self.activate_func == 'tanh':
            delta2 = delta3.dot(self.params['W3'].T) * self.tanh_derivative(self.Z2)
        # 第二隐藏层权重和偏置的梯度
        dW2 = self.A1.T.dot(delta2) + reg_lambda * self.params['W2']
        db2 = np.sum(delta2, axis=0)
        # 第一隐藏层误差
        if self.activate_func =='relu':
            delta1 = delta2.dot(self.params['W2'].T) * self.relu_derivative(self.Z1)
        elif self.activate_func =='sigmoid':
            delta1 = delta2.dot(self.params['W2'].T) * self.sigmoid_derivative(self.Z1)
        elif self.activate_func == 'tanh':
            delta1 = delta2.dot(self.params['W2'].T) * self.tanh_derivative(self.Z1)
        # 第一隐藏层权重和偏置的梯度
        dW1 = X.T.dot(delta1)+ reg_lambda * self.params['W1']
        db1 = np.sum(delta1, axis=0)
        # 更新梯度   
        self.params['W1'] -= learning_rate * dW1
        self.params['b1'] -= learning_rate * db1
        self.params['W2'] -= learning_rate * dW2
        self.params['b2'] -= learning_rate * db2
        self.params['W3'] -= learning_rate * dW3
        self.params['b3'] -= learning_rate * db3
        
    def compute_accuracy(self,y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)
    
    def visualize_parameters(self, figure_path):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2行3列的子图
        
        # 可视化权重矩阵
        cax1 = axs[0, 0].imshow(self.params['W1'], aspect='auto', cmap='viridis')
        fig.colorbar(cax1, ax=axs[0, 0])
        axs[0, 0].set_title("Weights of the First Layer")

        cax2 = axs[0, 1].imshow(self.params['W2'], aspect='auto', cmap='viridis')
        fig.colorbar(cax2, ax=axs[0, 1])
        axs[0, 1].set_title("Weights of the Second Layer")

        cax3 = axs[0, 2].imshow(self.params['W3'], aspect='auto', cmap='viridis')
        fig.colorbar(cax3, ax=axs[0, 2])
        axs[0, 2].set_title("Weights of the Third Layer")

        # 可视化偏置向量
        axs[1, 0].bar(np.arange(len(self.params['b1'])), self.params['b1'])
        axs[1, 0].set_title("Biases of the First Layer")

        axs[1, 1].bar(np.arange(len(self.params['b2'])), self.params['b2'])
        axs[1, 1].set_title("Biases of the Second Layer")

        axs[1, 2].bar(np.arange(len(self.params['b3'])), self.params['b3'])
        axs[1, 2].set_title("Biases of the Third Layer")

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig(figure_path)  # 保存图像

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, initial_learning_rate, reg_lambda, decay_factor):
        learning_rate = initial_learning_rate  # 设置初始学习率
        n_samples = X_train.shape[0]  # 获取训练样本数量
        losses = []  # 存储每个epoch的损失
        accuracies = []  # 存储每个epoch的准确率
        best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
        best_params = {}  # 用于存储最优模型参数

        for epoch in range(epochs):  # 循环epochs次
            total_loss = 0.0  # 初始化总损失为0

            permutation = np.random.permutation(n_samples)  # 生成训练样本的随机排列索引
            X_train_shuffled = X_train[permutation]  # 根据随机排列索引对训练数据进行洗牌
            y_train_shuffled = y_train[permutation]  # 根据随机排列索引对标签数据进行洗牌
            
            for i in range(0, n_samples, batch_size):  # 循环处理每个batch的样本
                X_batch = X_train_shuffled[i: i + batch_size]  # 获取当前batch的训练数据
                y_batch = y_train_shuffled[i: i + batch_size]  # 获取当前batch的标签数据
                y_pred = self.forward(X_batch)  # 使用前向传播获取预测结果
                
                total_loss += self.cross_entropy_loss(y_pred, y_batch) + 0.5 * reg_lambda * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])) + np.sum(np.square(self.params['W3'])))  # 计算总损失
                self.backward(X_batch, y_batch, learning_rate, reg_lambda)  # 反向传播更新参数

            y_val_pred = self.forward(X_val)  # 使用前向传播获取验证集的预测结果
            val_loss = self.cross_entropy_loss(y_val_pred, y_val) + 0.5 * reg_lambda * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])) + np.sum(np.square(self.params['W3']))) # 计算验证集损失
            val_acc = self.compute_accuracy(y_val_pred, y_val)  # 计算验证集准确率
            losses.append(val_loss)
            accuracies.append(val_acc)

            if val_loss < best_val_loss:  # 如果当前验证集损失小于之前的最佳损失
                best_val_loss = val_loss  # 更新最佳验证损失
                best_params = self.params.copy()  # 保存最优模型参数
            
            print(f"Epoch {epoch}, Total Loss: {total_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")  # 打印当前epoch的总损失和验证集损失和准确率
            
            learning_rate = learning_rate * decay_factor  # 调整学习率
        
        # 保存最优模型参数
        self.save(f'/Users/jrryzh/Documents/lectures/神经网络/lab1/ckpts/Hidden1{self.params["W1"].shape[1]}_Hidden2{self.params["W2"].shape[1]}_InitLR{initial_learning_rate}_Lambda{reg_lambda}_Decay{decay_factor}_Batch{batch_size}_Epochs{epochs}_parameters_' + 'best_model_params.pkl', best_params)
        # 可视化模型参数
        self.visualize_parameters(f'/Users/jrryzh/Documents/lectures/神经网络/lab1/results/Hidden1{self.params["W1"].shape[1]}_Hidden2{self.params["W2"].shape[1]}_InitLR{initial_learning_rate}_Lambda{reg_lambda}_Decay{decay_factor}_Batch{batch_size}_Epochs{epochs}_parameters.png')  
        
        # 设置图表标题，包含超参数信息
        title = f'Training Curve: LR={initial_learning_rate}, Lambda={reg_lambda}, Decay={decay_factor}, Batch Size={batch_size}, Epochs={epochs}'
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # 1行2列的第1个
        plt.plot(losses, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)  # 1行2列的第2个
        plt.plot(accuracies, label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # 添加主标题
        plt.suptitle(title)

        # 保存图像，文件名也包含超参数
        filename = f'/Users/jrryzh/Documents/lectures/神经网络/lab1/results/Hidden1{self.params["W1"].shape[1]}_Hidden2{self.params["W2"].shape[1]}_InitLR{initial_learning_rate}_Lambda{reg_lambda}_Decay{decay_factor}_Batch{batch_size}_Epochs{epochs}.png'
        plt.savefig(filename)
        print("Saved training curve to file: ", filename)

    def test(self, X_test, y_test):
        # 使用前向传播预测测试集标签
        y_test_pred = self.forward(X_test)
        # 计算测试准确率
        test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == y_test)
        # 打印测试准确率
        print(f"Test Accuracy: {test_accuracy}")
