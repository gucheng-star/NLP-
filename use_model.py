# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pickle as pkl
import os
from jieba import lcut
import numpy as np
import tkinter as tk
from tkinter import ttk

# 加载词汇表和预训练词向量
vocab_path = os.path.join(os.path.dirname(__file__), 'data', 'vocab.pkl')
embedding_path = os.path.join(os.path.dirname(__file__), 'data', 'embedding_Tencent.npz')
model_path = os.path.join(os.path.dirname(__file__), 'saved_dict', 'lstm.ckpt')

vocab = pkl.load(open(vocab_path, 'rb'))
embedding_pretrained = torch.tensor(np.load(embedding_path)["embeddings"].astype('float32'))

# 超参数设置 (与LSTM/main.py保持一致)
embed = embedding_pretrained.size(1)        # 词向量维度
dropout = 0.5                               # 随机丢弃
num_classes = 2                             # 类别数
pad_size = 50                               # 每句话处理成的长度(短填长切)
hidden_size = 128                           # lstm隐藏层
num_layers = 2                              # lstm层数
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号

# 定义LSTM模型 (与LSTM/main.py中的Model类一致)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        self.lstm = nn.LSTM(embed, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

# 数据处理
"""判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5') :
        return True
    else:
        return False
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str

def getStopWords():
    # 根据用户反馈，代码中没有停用词，因为在处理中文时符号全被去掉，所以返回空列表
    return []

def dataParse_(content, stop_words):
    content = reserve_chinese(content)
    words = lcut(content)
    # 根据用户反馈，停用词处理已不再需要
    # words = [i for i in words if not i in stop_words]
    return words

def predict_(text_o):
    stop_words = getStopWords() # This will now return an empty list
    content = dataParse_(text_o, stop_words)

    # 将文本转换为词ID序列
    words_line = []
    token = content
    if pad_size:
        if len(token) < pad_size:
            token.extend([vocab.get(PAD)] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))

    # 转换为PyTorch Tensor
    x = torch.LongTensor([words_line])

    # 实例化模型并加载权重
    model = Model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 设置为评估模式

    # 进行预测
    with torch.no_grad():
        outputs = model(x)
        pred = torch.max(outputs.data, 1)[1].cpu().numpy()

    labels11 = ['active', 'negative']
    pred_lable = []
    for i in pred:
        pred_lable.append(labels11[i])
    return pred_lable[0]

def main_windows():
    root = tk.Tk()
    root.title('情感分析系统')
    root.geometry('600x450') # 设置窗口初始大小

    # 输入文本框
    input_label = tk.Label(root, text='请输入文本：')
    input_label.grid(row=0, column=0, columnspan=4, pady=5, padx=10, sticky='w')
    input_text = tk.Text(root, height=10, width=70)
    input_text.grid(row=1, column=0, columnspan=4, pady=5, padx=10)

    # 分析结果显示
    result_label_static = tk.Label(root, text='分析结果：', font=("Helvetica", 15))
    result_label_static.grid(row=2, column=0, pady=10, padx=10, sticky='w')
    result_label_dynamic = tk.Label(root, text='     ', font=("Helvetica", 15), fg='blue')
    result_label_dynamic.grid(row=2, column=1, columnspan=3, pady=10, padx=10, sticky='w')

    def start_analysis():
        text_to_analyze = input_text.get("1.0", tk.END).strip()
        if text_to_analyze:
            result = predict_(text_to_analyze)
            result_label_dynamic.config(text=result)
        else:
            result_label_dynamic.config(text="请输入内容")

    def clear_fields():
        input_text.delete("1.0", tk.END)
        result_label_dynamic.config(text="     ")

    # 按钮
    start_button = tk.Button(root, text='开始', font=("Helvetica", 15), command=start_analysis)
    start_button.grid(row=3, column=1, pady=10)
    clear_button = tk.Button(root, text='清空', font=("Helvetica", 15), command=clear_fields)
    clear_button.grid(row=3, column=2, pady=10)

    # 调整列的权重，使输入框和结果标签可以随窗口大小调整
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)
    root.grid_columnconfigure(3, weight=1)
    root.grid_rowconfigure(1, weight=1) # 让文本输入框可以扩展

    root.mainloop()

if __name__ == "__main__":
    main_windows()