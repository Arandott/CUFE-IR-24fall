import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch.nn as nn

#==================== 数据集类定义 ====================
class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

#==================== 钩子函数相关 ====================
# 将所有中间层输出保存到此dict中。键为层名称，值为输出张量。
layer_outputs = {}

def get_forward_hook(name):
    """返回一个forward hook函数，用于捕捉指定层的输出"""
    def hook(module, input, output):
        # output通常为last_hidden_state: [batch_size, seq_len, hidden_size]
        # 我们只取CLS token位置的特征: output[:,0,:]
        # output可能是tuple时，检查类型
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        cls_emb = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        layer_outputs[name] = cls_emb.detach().cpu().numpy()
    return hook

def register_hooks(model):
    """
    递归遍历模型模块，将钩子挂载在BertLayer层上。
    BertForSequenceClassification -> BertModel -> Encoder -> BertLayer(Layers)
    通常BertEncoder包含 .layer 是一个list of BertLayer
    """
    for i, layer_module in enumerate(model.bert.encoder.layer):
        layer_name = f"layer_{i}"
        layer_module.register_forward_hook(get_forward_hook(layer_name))

#==================== 主流程 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载模型与分词器
    model_path = 'results/checkpoint-7265'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 读取测试数据
    # testSet-1000.xlsx列：['id', 'title given by manchine', 'Y/N', 'original title']
    df = pd.read_excel('data/testSet-1000.xlsx')
    texts = df['title given by manchine'].tolist()
    # 将Y/N映射为0/1
    labels = df['Y/N'].apply(lambda x: 1 if x == 'Y' else 0).tolist()
    
    # 创建dataset
    test_dataset = TestDataset(texts, labels, tokenizer, max_length=64)

    # 注册钩子函数
    register_hooks(model)

    # 一次性前向传播（假设一次能放下1000条）
    # 若显存不足，可分批处理，这里为简单起见一次性forward
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    for i in range(len(test_dataset)):
        data = test_dataset[i]
        all_input_ids.append(data['input_ids'])
        all_attention_mask.append(data['attention_mask'])
        all_labels.append(data['labels'])

    all_input_ids = torch.stack(all_input_ids).to(device)         # [N, seq_len]
    all_attention_mask = torch.stack(all_attention_mask).to(device) # [N, seq_len]
    all_labels = torch.stack(all_labels) # 在后续可用于上色标记

    # 红色打印：Start forwarding
    print("\033[1;31mStart forwarding...\033[0m")
    with torch.no_grad():
        # 前向传播，触发hook收集所有层的CLS向量
        outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask)
    # 绿色打印：Forwarding finished
    print("\033[1;32mForwarding finished.\033[0m")

    # 此时layer_outputs中包含了各层的CLS表征
    # layer_outputs的keys为layer_0, layer_1,... 直到最后一层
    # 每个value为 (N, hidden_size) 的numpy数组

    # 使用t-SNE降维并可视化
    # 我们可以对每层都进行t-SNE，然后在同一幅图中以子图展示
    # 假设BERT-base有12层，每层可生成一个散点图
    n_layers = len(layer_outputs.keys())
    fig, axes = plt.subplots(3, 4, figsize=(40, 30))  # 12层 = 3行4列
    axes = axes.flatten()

    all_labels = all_labels.numpy()
    for i, (layer_name, cls_emb) in enumerate(tqdm(layer_outputs.items(), desc="Processing layers")):
        # cls_emb: [N, hidden_size]
        # 进行t-SNE:降到2维
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(cls_emb) # [N, 2]

        # 根据标签上色
        # label=1(Y)为正类，label=0(N)为负类
        pos_idx = (all_labels == 1)
        neg_idx = (all_labels == 0)

        ax = axes[i]
        ax.scatter(emb_2d[pos_idx, 0], emb_2d[pos_idx, 1], c='red', label='Y', alpha=0.6)
        ax.scatter(emb_2d[neg_idx, 0], emb_2d[neg_idx, 1], c='blue', label='N', alpha=0.6)
        ax.set_title(f'{layer_name} T-SNE')
        ax.legend()

    plt.tight_layout()
    plt.savefig("imgs/bert_layers_tsne_visualization.png")
    plt.show()

    print("T-SNE可视化完成！查看 bert_layers_tsne_visualization.png 文件。")
    
    # 将layer_outputs保存到文件
    os.makedirs('outputs', exist_ok=True)
    torch.save(layer_outputs, 'outputs/layer_outputs.pt')
