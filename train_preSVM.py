# train.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from utils import load_data_from_html  # 确保 utils.py 中有此函数
import numpy as np
from tqdm import tqdm

class TitleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
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
    # 文件路径（请根据实际情况修改）
    neg_file = 'data/negative_trainingSet.html'
    pos_file = 'data/positive_trainingSet.html'
    
    # 加载数据
    all_texts, all_labels = load_data_from_html(neg_file, pos_file)
    
    # 简单划分训练和验证集 (8:2比例)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts,
        all_labels,
        test_size=0.2,        # 验证集比例为20%
        stratify=all_labels,  # 基于标签分层抽样，保证比例平衡
        random_state=42
    )
    
    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    
    # 创建训练集和验证集Dataset
    train_dataset = TitleDataset(train_texts, train_labels, tokenizer, max_length=64)
    val_dataset = TitleDataset(val_texts, val_labels, tokenizer, max_length=64)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = BertModel.from_pretrained('bert-large-uncased')
    
    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 提取训练集的CLS embeddings
    print("Extracting CLS embeddings for training set...")
    train_embeddings, train_labels_array = extract_cls_embeddings(model, train_loader, device)
    print(f"Training embeddings shape: {train_embeddings.shape}, Labels shape: {train_labels_array.shape}")
    
    # 提取验证集的CLS embeddings
    print("Extracting CLS embeddings for validation set...")
    val_embeddings, val_labels_array = extract_cls_embeddings(model, val_loader, device)
    print(f"Validation embeddings shape: {val_embeddings.shape}, Labels shape: {val_labels_array.shape}")
    
    # 保存embeddings和labels到文件
    os.makedirs('embeddings', exist_ok=True)
    np.save('embeddings/train_embeddings.npy', train_embeddings)
    np.save('embeddings/train_labels.npy', train_labels_array)
    np.save('embeddings/val_embeddings.npy', val_embeddings)
    np.save('embeddings/val_labels.npy', val_labels_array)
    print("Embeddings and labels saved to 'embeddings/' directory.")
