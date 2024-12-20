# test.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

class TestDataset(Dataset):
    """
    自定义数据集类，用于加载测试数据。
    """
    def __init__(self, texts, labels, tokenizer, max_length=64):
        """
        初始化数据集。

        参数:
        - texts (List[str]): 文本列表。
        - labels (List[int]): 标签列表。
        - tokenizer (transformers.PreTrainedTokenizer): 分词器。
        - max_length (int): 最大序列长度。
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        根据索引返回一个样本。

        参数:
        - idx (int): 样本索引。

        返回:
        - dict: 包含input_ids, attention_mask, labels的字典。
        """
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
            'input_ids': enc['input_ids'].squeeze(0),         # [seq_len]
            'attention_mask': enc['attention_mask'].squeeze(0), # [seq_len]
            'labels': torch.tensor(label, dtype=torch.long)   # 标签到tensor
        }

def extract_cls_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_batch = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()  # [batch_size, hidden_size]
            
            embeddings.append(cls_embeddings)
            labels.extend(label_batch)
    
    embeddings = np.vstack(embeddings)  # [num_samples, hidden_size]
    labels = np.array(labels)
    return embeddings, labels

if __name__ == "__main__":
    # 检查设备是否可用（GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #==================== 加载模型与分词器 ====================
    # 模型检查点路径
    checkpoint_path = 'results/checkpoint-7265'  # 假设这是训练好的模型的checkpoint目录
    
    # 初始化分词器
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    
    # 初始化模型
    print("Loading model...")
    model = BertModel.from_pretrained('bert-large-uncased')
    
    # 尝试从checkpoint加载权重
    if os.path.exists(os.path.join(checkpoint_path, 'pytorch_model.bin')):
        model = BertModel.from_pretrained(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}.")
    else:
        print(f"Checkpoint path {checkpoint_path} does not contain 'pytorch_model.bin'. Using 'bert-large-uncased' pre-trained weights.")
    
    model.to(device)
    
    #==================== 读取测试数据 ====================
    # 测试数据文件路径
    test_data_path = 'data/testSet-1000.xlsx'
    print(f"Loading test data from {test_data_path}...")
    # 读取Excel文件，假设列名为 ['id', 'title given by manchine', 'Y/N', 'original title']
    df = pd.read_excel(test_data_path)
    
    # 提取需要的列
    texts = df['title given by manchine'].tolist()
    # 将 'Y' 映射为1，'N' 映射为0
    labels = df['Y/N'].apply(lambda x: 1 if x == 'Y' else 0).tolist()
    
    #==================== 创建Dataset和DataLoader ====================
    print("Creating Dataset and DataLoader...")
    test_dataset = TestDataset(texts, labels, tokenizer, max_length=64)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    #==================== 提取CLS Embeddings ====================
    print("Extracting CLS embeddings for test set...")
    test_embeddings, test_labels_array = extract_cls_embeddings(model, test_loader, device)
    print(f"Test embeddings shape: {test_embeddings.shape}, Labels shape: {test_labels_array.shape}")
    
    #==================== 保存Embeddings和Labels ====================
    os.makedirs('embeddings', exist_ok=True)
    np.save('embeddings/test_embeddings.npy', test_embeddings)
    np.save('embeddings/test_labels.npy', test_labels_array)
    print("Test embeddings and labels saved to 'embeddings/' directory.")
