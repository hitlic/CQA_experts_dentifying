# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:55:58 2019

@author: 53445

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
"""
import sqlite3
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import numpy as np
import pandas as pd
from utils import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
import re
try:
    from pyhanlp import HanLP as hanlp
except Exception:
    pass
import logging
logger = logging.getLogger('main.data_tools')


class Index:
    
    question_authors = {}
    author_questions = {}
    
    question2index = {}
    index2question = {}
    
    author_answers = {}
    
    answer2index = {}
    index2answer = {}
    
    def __init__(self):
        self.get_question2index()
        self.get_answer2index()
        self.get_dics()
        
    def get_dics(self):
        con = sqlite3.connect("./data/zhihu.db")
        cur = con.cursor()#用来从zhihu表中提取数据   
        cur.execute("select datas from zhihu" )
        for i in cur:
            data = eval(i[0])
            if data['question']['id'] not in self.question_authors.keys():
                    self.question_authors[data['question']['id']] = []
            if data['author']['id'] not in self.question_authors[data['question']['id']]:  
                    self.question_authors[data['question']['id']].append(data['author']['id'])
                    
            if data['author']['id'] not in self.author_questions.keys():
                    self.author_questions[data['author']['id']] = []
            if str(data['question']['id']) not in self.author_questions[data['author']['id']]:  
                    self.author_questions[data['author']['id']].append(str(data['question']['id']))

            if data['author']['id'] not in self.author_answers.keys():
                    self.author_answers[data['author']['id']] = []
            if str(data['id']) not in self.author_answers[data['author']['id']]:  
                    self.author_answers[data['author']['id']].append(str(data['id']))
            con.commit()  ## 保存
        con.close()
        #print(self.author_answers.keys())
        #print(self.author_questions.keys())
        
    def get_question2index(self):
        con = sqlite3.connect("./data/zhihu.db")
        cur = con.cursor()#用来从zhihu表中提取数据   
        cur.execute("select id from questions ORDER BY id asc" )
        k = 0
        for i in cur:
            self.question2index[i[0]] = k
            self.index2question[k] = i[0]
            k += 1
            con.commit()   # 保存
        con.close()
    
    def get_answer2index(self):
        con = sqlite3.connect("./data/zhihu.db")
        cur = con.cursor()#用来从zhihu表中提取数据   
        cur.execute("select id from answers where cleaned_content!='' ORDER BY id asc" )
        k = 0
        for i in cur:
            self.answer2index[i[0]] = k
            self.index2answer[k] = i[0]
            k += 1
            con.commit()   # 保存
        #print(self.answer2index)
        con.close()
        
def read_q_a():
    '''
    从数据库中读取问题和作者数据，构成Question-Author对
    '''
    con = sqlite3.connect("./data/zhihu.db")
    cur = con.cursor()#用来从zhihu表中提取数据   
    cur.execute("select question_id,author_id from answers" )
    q_a = cur.fetchall()
    con.close()
    return q_a
 

def read_q_a2():
    data = []
    f = open("./data/q_a.csv",encoding='UTF-8')
    line = f.readline()
    while line:
        data.append(line.strip().split(","))
        line = f.readline()
    f.close()
    return data


def build_net():
    '''
    构建网络，如果问题id相同，回答者id不同则在这两回答者之间建立一条边
    '''
    data = read_q_a()
    #data = data[:1350]
    #data = read_q_a2()
    G = nx.Graph()
    #k = 0
   
    #添加边
    for i in data:
        for j in data:
            if j[1] != '0' and i[1] != '0' and i[0] == j[0] and i[1] != j[1] :
                G.add_edge(i[1], j[1], question_id = i[0])
                
    #找到最大联通子图
    largest_components=max(nx.connected_components(G),key=len)  # 高效找出最大的联通成分，其实就是sorted里面的No.1
    print("最大联通子图：", len(largest_components))#最大联通子图:3264(现)
    
    #移除不在最大联通子图中的节点
    for i in data:
        if i[1] in G and i[1] not in largest_components:
            G.remove_node(i[1])
    #建立index2node词典
    index2node = {}
    node2index = {}
    k = 0
    for n in G.nodes():#_iter():
    #for n,nbrdict in G.adjacency_iter():
        G.node[n]['index'] = k
        #print(nbrdict)
        #print(G.node[n]['index'])
        index2node[k] = n
        node2index[n] = k
        k += 1
    return G, index2node, node2index
            

def draw_graph(G):
    '''
    画出G
    '''  
    nx.draw(G,node_size=10)#,with_labels = True)
    #nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
    #plt.rcParams['savefig.dpi'] = 300 #图片像素
    #plt.rcParams['figure.dpi'] = 300 #分辨率
    #plt.savefig("tu2.png")
    plt.show()
    #nx.write_adjlist(G,"net.adjlist")

    
def save_edges(G):
    '''
    将图保存为节点对
    '''
    f = open("edges.csv", "w", encoding='utf-8')
    for n in G.edges():#_iter():
        f.write(n[0]+ " " + n[1])
        f.write("\n")


def merge_data():
    '''
    将两部分数据合并在一起，读入的文件中每一行是一句insert语句
    '''
    con = sqlite3.connect("./data/zhihu.db")
    cur = con.cursor()#用来从zhihu表中提取数据
    f = open("zhihu.txt",encoding='UTF-8')
    line = f.readline()
    while line:
        print(line)
        cur.execute(line)
        line = f.readline()
    f.close()
    con.close()
    
        
def save_authors():
    '''
    将author表中中的数据按字段进一步整理到csv中
    '''
    con = sqlite3.connect("./data/zhihu.db")
    cur = con.cursor()#用来从zhihu表中提取数据
    cur2 = con.cursor()
    cur.execute("select datas from zhihu" )
    fw = open('author.csv','w',encoding="utf-8")
    flag = 0
    for i in cur:
        data = eval(i[0])
        #写入csv文件
        '''
        if flag == 0:
            flag = 1
            fw.write("id\tname\tfollower_count\tuser_type\theadline\tgender\turl\turl_token\tbadge\ttype\tavatar_url\tavatar_url_template\tis_followed\tis_privacy\tis_org\tis_advertiser\n")
        fw.write(data['id'] + "\t" + data['name'] + "\t" + str(data['follower_count']) + "\t" + \
                 data['user_type'] + "\t" + data['headline'] + "\t" + str(data['gender']) + "\t" +\
                 data['url'] + "\t" + data['url_token'] + "\t"  + \
                 str(data['badge']) + "\t" + data['type'] + "\t" + \
                 data['avatar_url'] + "\t" + data['avatar_url_template'] + "\t" + \
                 str(data['is_followed']) + "\t" + str(data['is_privacy']) + "\t" + \
                 str(data['is_org']) + "\t" +  str(data['is_advertiser'])
                  )
        fw.write("\n")
        '''
        #写入数据库
        sql_insert = "REPLACE INTO authors values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        param = (str(data['author']['id']), str(data['author']['name']), str(data['author']['follower_count']),
                 data['author']['user_type'], data['author']['headline'], str(data['author']['gender']), 
                 data['author']['url'], data['author']['url_token'], str(data['author']['badge']), 
                 data['author']['type'], data['author']['avatar_url'], data['author']['avatar_url_template'],
                 data['author']['is_followed'],  data['author']['is_privacy'],  data['author']['is_org'], 
                 data['author']['is_advertiser']
                 )
        cur2.execute(sql_insert, param)
        con.commit()   # 保存
    con.close()
    fw.close()


def save_questions():
    con = sqlite3.connect("./data/zhihu.db")
    cur2 = con.cursor()#用来向nodes表中插入数据
    cur = con.cursor()#用来从zhihu表中提取数据   
    cur.execute("select datas from zhihu" )
    for i in cur:
        data = eval(i[0])
        sql_insert = "REPLACE INTO questions values (?,?,?,?,?,?,?,?)"
        param = (str(data['question']['id']), data['question']['created'], data['question']['question_type'],
                str(data['question']['relationship']), data['question']['title'], data['question']['type'], 
                data['question']['updated_time'], data['question']['url']
                )
        cur2.execute(sql_insert, param)
        con.commit()   # 保存
    con.close()
    
    
def save_answers():
    con = sqlite3.connect("./data/zhihu.db")
    cur2 = con.cursor()#用来向nodes表中插入数据
    cur = con.cursor()#用来从zhihu表中提取数据   
    cur.execute("select datas from zhihu" )
    for i in cur:
        data = eval(i[0])
        sql_insert = "REPLACE INTO answers values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        param = (str(data['id']), data['admin_closed_comment'], str(data['annotation_action']),
                data['answer_type'], data['author']['id'], str(data['can_comment']), 
                data['collapse_reason'], data['collapsed_by'], str(data['comment_count']), 
                data['comment_permission'], data['content'], str(data['created_time']), 
                data['editable_content'], data['excerpt'], data['extras'], 
                data['is_collapsed'], data['is_copyable'], data['is_labeled'], 
                data['is_normal'], data['is_sticky'], str(data['mark_infos']), 
                data['question']['id'], str(data['relationship']), str(data['relevant_info']), 
                data['reshipment_settings'], str(data['reward_info']), str(data['suggest_edit']), 
                data['type'], str(data['updated_time']), data['url'], str(data['voteup_count'])
                )
        cur2.execute(sql_insert, param)
        con.commit()   # 保存
    con.close()
    

def clean_html_label():
    con = sqlite3.connect("./data/zhihu.db")
    cur2 = con.cursor()#用来向nodes表中插入数据
    cur = con.cursor()#用来从zhihu表中提取数据   
    cur.execute("select content,id from answers" )
    for i in cur:
        cleaned_content = re.compile(r'<[^>]+>',re.S).sub('',i[0])
        print(cleaned_content)
        sql_insert = "update answers set 'cleaned_content'=? where id = ?"
        param = (cleaned_content, i[1])
        cur2.execute(sql_insert, param)
        con.commit()   # 保存
    con.close()
    
    
def get_question_title():
    con = sqlite3.connect("./data/zhihu.db")
    cur = con.cursor()#用来从zhihu表中提取数据
    cur.execute("SELECT title FROM questions ORDER BY id ASC" )
    title = []
    for i in cur:
        title.append(i[0])
    con.close()
    return title


def get_answer_content():
    con = sqlite3.connect("./data/zhihu.db")
    cur = con.cursor()#用来从zhihu表中提取数据
    cur.execute("SELECT cleaned_content FROM answers where cleaned_content!='' ORDER BY id asc" )
    cleaned_content = []
    for i in cur:
        cleaned_content.append(i[0])
    con.close()
    return cleaned_content


def get_data_by_sql():
    con = sqlite3.connect("./data/zhihu.db")
    cur = con.cursor()#用来从zhihu表中提取数据
    cur.execute("SELECT sum(voteup_count)*1.0/(SELECT COUNT(author_id) FROM answers as x  where x.author_id = y.author_id) as avg_voteup,author_id FROM answers as y WHERE 1 = 1 GROUP BY author_id; " )
    fw = open('sql_answer.csv','w',encoding="utf-8")
    flag = 0
    for i in cur:
        #写入csv文件
        if flag == 0:
            flag = 1
            fw.write("avg_voteup,author_id\n")
        fw.write(str(i[0]) + "," + str(i[1]))
        fw.write("\n")
    con.close()
    fw.close()
 

def get_repeated_edge():
    data = read_q_a()
    #data = data[:200]
    dic = {}
    #添加边
    for i in data:
        print(i)
        for j in data:
            if i[0] == j[0] and i[1] != j[1] and j[1] != '0' and i[1] != '0':
                lis = [i[1],j[1]]
                lis.sort()
                t = tuple(lis)
                if t not in dic.keys():
                    dic[t] = []
                if i[0] not in dic[t]:  
                    dic[t].append(i[0])
    fw = open("./data/repeated_edges.csv", "w", encoding='utf-8')
    for key,value in dic.items():
        fw.write(str(key) + "\t" + str(value) + "\t" + str(len(value)) + "\n")
    fw.close()
#================================================
def graph2sparse_max(G, index_qid = None, weight = None):
    '''
    按照author的index属性构建邻接矩阵，阵存储成csc_matrix形式
    '''
    row = []
    col =[]
    if index_qid is not None and weight is not None:
        data = []
        question_id = nx.get_edge_attributes(G, 'question_id')
        for n in G.edges():#_iter():
            row.append(G.node[n[0]]['index'])
            col.append(G.node[n[1]]['index'])
            data.append(weight[index_qid.index(str(question_id[n]))])
            row.append(G.node[n[1]]['index'])
            col.append(G.node[n[0]]['index'])
            data.append(weight[index_qid.index(str(question_id[n]))])
            
    else:
        question_id = nx.get_edge_attributes(G, 'question_id')
        for n in G.edges():#_iter():
            row.append(G.node[n[0]]['index'])
            col.append(G.node[n[1]]['index'])
            row.append(G.node[n[1]]['index'])
            col.append(G.node[n[0]]['index'])
        data = np.ones(len(row))
    adj = csc_matrix((data, (row, col)), shape=(len(G),len(G)))
    #print(adj.toarray())
    return adj
    

def loaddata(G):
    '''
    从csv文件中读取特征和标签
    '''
    #datas = pd.read_csv("./data/author_6220_label_50_avgvoteup.csv", header=None, encoding='utf-8')
    #datas = pd.read_csv("./data/author_6220_label_50.csv", header=None, encoding='utf-8')
    datas = pd.read_csv("./data/author_6220_label_0.csv", header=None, encoding='utf-8')
    labels = [-1 for i in range(len(G))]
    features = [0 for i in range(len(G))]
    for i in range(len(datas)):
        #print(datas[0][i])
        if datas[0][i] in G:#float(datas[17][i]),
            features[G.node[datas[0][i]]['index']] = [float(datas[2][i]),
                    float(datas[15][i]),float(datas[14][i]),float(datas[13][i]),float(datas[12][i]),float(datas[5][i])]
            labels[G.node[datas[0][i]]['index']] = int(datas[16][i])
    '''
    print(features)
    print(labels)
    print(len(G))
    '''
    return features,labels


def divide_data(labels):
    '''
    划分数据集
    '''
    idx_train = range(1000)#(300)#
    idx_val = range(1000, 2000)#(300,400)#
    idx_test = range(2000, 3200)#(400,450)#
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return y_train, y_val, y_test, train_mask, val_mask, test_mask

#=====================================================

def labels2onehot(labels, class_num=None, class_labels=None):
    """
    生成句子的情感标记。调用时class_num与class_labels必选其一。
    :param labels: list; 数据的标记列表
    :param class_num: int; 类别总数
    :param class_labels: list; 类别标记，如[0, 1]、['a', 'b']
    :return: numpy array.
    """
    if class_num is None and class_labels is None:
        raise Exception("Parameter eithor class_num or class_labels must be given!  -- by lic")
    if class_labels is not None:
        class_num = len(class_labels)

    def label2onehot(label_):
        if class_labels is None:
            label_index = label_
        else:
            label_index = class_labels.index(label_)
        onehot_label = [0] * class_num
        onehot_label[label_index] = 1
        return onehot_label

    return np.array([label2onehot(label_) for label_ in labels])


def load_embedding(embedding_file):
    """
    加载词向量，返回词典和词向量矩阵
    :param embedding_file: 词向量文件
    :return: tuple, (词典, 词向量矩阵)
    """
    logger.info('loading word dict and word embedding...')
    with open(embedding_file, encoding='utf-8') as f:
        lines =[]
        line = f.readline()#略过第一行
        line = f.readline()
        while line:
            lines.append(line)
            line = f.readline()
        #lines = f.readlines()
        embedding_tuple = [tuple(line.strip().split(' ', 1)) for line in lines]
        embedding_tuple = [(t[0].strip().lower(), list(map(float, t[1].split()))) for t in embedding_tuple]
    embedding_matrix = []
    embedding_dim = len(embedding_tuple[0][1])
    embedding_matrix.append([0] * embedding_dim)  # 首行全为0，表示未登录词
    word_dict = dict()
    word_dict[''] = 0  # 空字符串表示未登录词
    word_id = 1
    for word, embedding in embedding_tuple:
        if word_dict.get(word) is None:
            word_dict[word] = word_id
            word_id += 1
            embedding_matrix.append(embedding)
    return word_dict, np.asarray(embedding_matrix, dtype=np.float32)


def drop_empty_texts(texts, labels):
    """
    去除预处理后句子为空的评论
    :param texts: id形式的文本列表
    :param labels: 标记数据
    :return: tuple of arrays. 非空句子列表，非空标记列表
    """
    logger.info("clear empty sentences ...")
    non_zero_idx = [id_ for id_, text in enumerate(texts) if len(text) != 0]
    texts_non_zero = np.array([texts[id_] for id_ in non_zero_idx])
    labels_non_zero = np.array([labels[id_] for id_ in non_zero_idx])
    return texts_non_zero, labels_non_zero


def make_dictionary_by_text(words_list):
    """
    构建词典（不使用已训练词向量时构建词典）
    :param words: list; 全部数数的词序列
    :return: tuple; 两个词典，word to int， int to word
    """
    logger.info("make dictionary by text ...")
    word_counts = Counter(words_list)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    id_to_word = {id_: word for id_, word in enumerate(sorted_vocab, 1)}
    word_to_id = {word: id_ for id_, word in id_to_word.items()}
    word_to_id[''] = 0
    id_to_word[0] = ''
    return word_to_id, id_to_word


def segment(text):
    '''
    使用HanLP对中文句子进行分词
    '''
    try:
        seg_result = hanlp.segment(text)
        return [term.word for term in seg_result]
    except Exception:
        return text.split()


def sentences2wordlists(sentence_list, lang='EN'):
    """
    将句子切分成词列表
    :param sentence_list: 句子列表
    :return: 词列表的列表
    """
    logger.info("word cutting ...")
    word_list_s = []
    for sentence in sentence_list:
        if lang == 'EN':  # 英文分词
            word_list = sentence.split()
        else:  # 中文分词
            word_list = segment(sentence)
        word_list_s.append(word_list)
    return word_list_s


def wordlists2idlists(word_list_s, word_to_id):
    """
    句子列表转id列表的列表
    :param word_list_s: 词列表的列表
    :param word_to_id: 词典
    :return: list of ints. id形式的句子列表
    """
    logger.info("convert word list to id list ...")
    sent_id_list = []
    for word_list in word_list_s:
        sent_id_list.append([word_to_id.get(word, 0) for word in word_list])
    return np.array(sent_id_list)


def dataset_padding(text_ids, sent_len):
    """
    句子id列表左侧补0
    :param text_ids: id形式的句子列表
    :param seq_ken:  int, 最大句长
    :return: numpy array.  补0后的句子
    """
    logger.info("padding dataset ...")
    textids_padded = np.zeros((len(text_ids), sent_len), dtype=int)
    for i, row in enumerate(text_ids):
        textids_padded[i, -len(row):] = np.array(row)[:sent_len]

    return np.array(textids_padded)


def dataset_split(texts, labels, train_percent, random_seed=None):
    """
    训练、开发、测试集划分，其中训练集比例为train_percent，开发集和测试各集为0.5(1-train_percent)
    :param text: 数据集x
    :param labels: 数据集标记
    :param train_percent: 训练集所占比例
    :return: (train_x, train_y, val_x, val_y, test_x, test_y)
    """
    logger.info("split dataset ...")
    # 检测x与y长度是否相等
    assert len(texts) == len(labels)
    # 随机化数据
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(texts))
    texts_shuf = np.array(texts)[shuf_idx]
    labels_shuf = np.array(labels)[shuf_idx]

    # 切分数据
    split_idx = int(len(texts_shuf)*train_percent)
    train_x, val_x = texts_shuf[:split_idx], texts_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(val_x)*0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == "__main__":
    #save_authors()
    #save_questions()
    #save_answers()
    
    G, index2node, node2id = build_net()
    #draw_graph(G)
    
    #adj = graph2sparse_max(G)
    #features, labels = loaddata(G, index2node)
    
    #get_data_by_sql()
    #get_repeated_edge()
    
    #clean_html_label()
    
    #questions = get_question_title()
    #answers = get_answer_content()
    
    #index = Index()
    
    
    