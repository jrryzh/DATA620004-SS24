# 假设你已经实现了不同的预训练数据集规模和超参数设置
# 使用相同的方法训练和评估模型
results = []
for dataset_size in [1000, 10000, 50000]:
    for learning_rate in [0.01, 0.001]:
        # 使用上述相同的训练和评估流程
        result = train_and_evaluate_simclr(dataset_size, learning_rate)
        results.append(result)

# 可视化结果
import matplotlib.pyplot as plt

# 假设results是一个包含所有实验结果的列表
plt.figure(figsize=(10, 5))
for result in results:
    plt.plot(result['epochs'], result['accuracy'], label=f"Size: {result['size']}, LR: {result['lr']}")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
