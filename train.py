import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import *


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



###############################################
# 主流程
###############################################

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 创建训练集和验证集Dataset
    train_dataset = TitleDataset(train_texts, train_labels, tokenizer, max_length=64)
    val_dataset = TitleDataset(val_texts, val_labels, tokenizer, max_length=64)
    
    # 初始化模型
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=5,   # 根据需要调整
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # 开始训练
    trainer.train()
    
    # 在验证集上预测
    preds_output = trainer.predict(val_dataset)
    y_pred = preds_output.predictions.argmax(axis=1)
    y_true = val_labels
    
    # 输出分类报告（precision, recall, f1-score等）
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))