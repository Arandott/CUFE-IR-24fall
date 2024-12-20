import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm

#==================== 数据集类定义 ====================
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

#==================== 主流程 ====================
if __name__ == "__main__":
    # 检查设备是否可用（GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #==================== 加载模型与分词器 ====================
    # 模型检查点路径
    checkpoint_path = 'results/checkpoint-7265'
    
    # 加载分词器
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 加载模型
    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()  # 设置为评估模式

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

    #==================== 进行预测 ====================
    all_preds = []
    all_true_labels = []
    
    print("Starting predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)           # [batch_size, seq_len]
            attention_mask = batch['attention_mask'].to(device) # [batch_size, seq_len]
            labels = batch['labels'].cpu().numpy()              # [batch_size]
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits                             # [batch_size, num_labels]
            
            # 获取预测结果
            preds = torch.argmax(logits, dim=1).cpu().numpy()   # [batch_size]
            
            # 收集预测和真实标签
            all_preds.extend(preds)
            all_true_labels.extend(labels)
    
    #==================== 生成分类报告 ====================
    print("Generating classification report...")
    report = classification_report(all_true_labels, all_preds, digits=4, target_names=['N', 'Y'])
    print("Classification Report:")
    print(report)
    
    #==================== 保存分类报告到文件 ====================
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Classification report saved to {report_path}")
