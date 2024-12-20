from bs4 import BeautifulSoup


def extract_lines_from_html(html_file):
    """
    从HTML文件中按行提取句子文本。
    假设每行存放一个句子，如果存在HTML标签，会通过BeautifulSoup提取纯文本。
    """
    texts = []
    with open(html_file, 'r', encoding='iso-8859-15') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 使用BeautifulSoup提取纯文本
            soup = BeautifulSoup(line, 'html.parser')
            text = soup.get_text().strip()
            if text:
                texts.append(text)
    return texts


def load_data_from_html(neg_file, pos_file):
    """
    加载数据，将负样本标记为0，正样本标记为1。
    """
    neg_texts = extract_lines_from_html(neg_file)
    pos_texts = extract_lines_from_html(pos_file)
    all_texts = neg_texts + pos_texts
    all_labels = [0]*len(neg_texts) + [1]*len(pos_texts)
    return all_texts, all_labels


def compute_metrics(eval_pred):
    """
    可在Trainer中使用的metrics计算函数。
    这里可返回accuracy等指标，但classification_report通常需要标签和预测列表。
    在训练结束后，我们使用classification_report进行更详细的评估。
    """
    from sklearn.metrics import accuracy_score
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}