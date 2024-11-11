import json
import jieba
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, file_path, max_length=200, vector_size=100, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.max_length = max_length
        self.vector_size = vector_size
        self.test_size = test_size
        self.random_state = random_state

        self.contents = []
        self.labels = []
        self.word_vec_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        data = []

        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去除前后空白字符
                if line:  # 跳过空行
                    # 替换单引号为双引号
                    line = line.replace("'", '"')
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: {e} - '{line}'")

        self.contents = [entry['Content'] for entry in data]
        self.labels = np.array([entry['Label'] for entry in data])

    def tokenize_contents(self):
        tokenized_contents = [list(jieba.cut(content)) for content in self.contents]
        cleaned_contents = [[word for word in sentence if word != '\n'] for sentence in tokenized_contents]
        return cleaned_contents

    def train_word_vector_model(self, cleaned_contents):
        self.word_vec_model = Word2Vec(sentences=cleaned_contents, vector_size=self.vector_size, window=5, min_count=1,
                                       workers=4)

    def vectorize_content(self, content):
        vectors = []
        for word in content:
            if word in self.word_vec_model.wv:  # 如果该词在词向量模型中
                vectors.append(self.word_vec_model.wv[word])  # 将词向量加入列表
            else:
                vectors.append(np.zeros(self.vector_size))  # 用零向量替代
        return np.array(vectors)

    def pad_vectors(self, vectors):
        if len(vectors) < self.max_length:
            padding = np.zeros((self.max_length - len(vectors), self.vector_size))
            vectors = np.vstack([vectors, padding])
        elif len(vectors) > self.max_length:
            vectors = vectors[:self.max_length]
        return vectors

    def process_data(self):
        self.load_data()  # 加载数据
        cleaned_contents = self.tokenize_contents()  # 分词
        self.train_word_vector_model(cleaned_contents)  # 训练词向量模型

        vectorized_contents = [self.vectorize_content(content) for content in cleaned_contents]
        padded_vectorized_contents = np.array([self.pad_vectors(content) for content in vectorized_contents])  # 填充向量
        # 对每个词向量独立进行标准化（沿着第三个维度）
        scaler = StandardScaler()

        # 对每个内容的词向量进行标准化
        features = np.array([scaler.fit_transform(content) for content in padded_vectorized_contents])
        label = self.labels
        return features,label

#         # 拆分数据集
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             features,
#             self.labels,
#             test_size=self.test_size,
#             random_state=self.random_state
#         )
#
#
# # 使用示例
# file_path = './train_data/train_data_32k.txt'  # 你的文本文件路径
# data_processor = DataProcessor(file_path)
# data_processor.process_data()
#
# # 打印结果
# print(data_processor.X_train.shape)
# print(data_processor.X_test.shape)
# print(data_processor.y_train.shape)
# print(data_processor.y_test.shape)
#
# np.save('X_train.npy', data_processor.X_train)
# np.save('X_test.npy', data_processor.X_test)
# np.save('y_train.npy', data_processor.y_train)
# np.save('y_test.npy', data_processor.y_test)