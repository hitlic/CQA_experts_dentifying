# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:15:03 2019

@author: 53445
"""

from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from scipy.sparse import  lil_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from model import Model
from efc import Embedding_FC
from itertools import chain
import jieba
from gensim.models.doc2vec import Doc2Vec
from data_process import *
import gc


def get_answer(id2node):
    '''
    每个node的answer不拼接在一起，doc长度大于node总数，返回doc和 node与doc下标的对应字典
    '''
    con = sqlite3.connect("./data/zhihu.db")
    cur = con.cursor()#用来从zhihu表中提取数据
    node2answerid = {}
    doc = []
    for value in id2node.values():
        node2answerid[value] = []
        for i in cur.execute("select cleaned_content from answers where author_id = '" + str(value) +"' order by 'created_time' desc;").fetchall():
            doc.append(jieba.lcut(i[0]))
            node2answerid[value].append(len(doc)-1)
    con.close()
    print("get_answer end")
    return doc, node2answerid

def get_text_feature(model, id2node,node2id):
    doc, node2answerid = get_answer(id2node)
    textfeature = {i:model.docvecs[i] for i, doc in enumerate(doc)}
    k = 10
    text_feature = {}
    for node,answerid in node2answerid.items():
        text_feature[node2id[node]] = []
        for i in range(k):
            if len(answerid)>i:
                text_feature[node2id[node]].extend(textfeature[answerid[i]])
            else:
                text_feature[node2id[node]].extend([0. for j in range(len(textfeature[0]))])
    return text_feature


epochs = 200
dropout = 0.5
early_stopping = 10     # Tolerance for early stopping (# of epochs).
max_degree = 3          # 'Maximum Chebyshev polynomial degree. 切多雪夫多项式的最高次数'


G, index2authorid, node2id = build_net()
adj = graph2sparse_max(G)
author_features, labels_list= loaddata(G)

print(sum(labels_list))
labels = labels2onehot(labels_list, class_num = 2)
mask = [[i] for i in range(len(labels))]
questions1 = get_question_title()
answers1 = get_answer_content()
texts1 = questions1 + answers1

index = Index()

cv = CountVectorizer(max_features = 20)#创建词袋数据结构
cv.fit(texts1)
q_cv_fit=cv.transform(questions1)
a_cv_fit=cv.transform(answers1)
q = q_cv_fit.toarray()
a = a_cv_fit.toarray()

q_feature = []
for i in range(len(index2authorid)):
    t1 = [0 for i in range(q.shape[1])]
    for question_id in index.author_questions[index2authorid[i]]:
        t1 += q[index.question2index[question_id]]
    q_feature.append(t1)
    
a_feature = []
for i in range(len(index2authorid)):
    t2 = [0 for i in range(a.shape[1])]
    for answer_id in index.author_answers[index2authorid[i]]:
        t2 += a[index.answer2index[answer_id]]
    a_feature.append(t2)

f = np.hstack((q_feature, a_feature))

features = np.hstack((f,author_features))
features = preprocess_features(features)

    
    
texts = sentences2wordlists(texts1, lang='CH')
embedding_matrix = None  # 设词向量矩阵为None
#vocab_to_int, embedding_matrix = load_embedding("./data/sgns.zhihu.char")  # 英文词向量
vocab_to_int, int_to_vocab = make_dictionary_by_text(list(chain.from_iterable(texts)))
questions = wordlists2idlists(questions1, vocab_to_int)
answers = wordlists2idlists(answers1, vocab_to_int)
q_sent_len = 10#50
a_sent_len = 50#5000
questions = dataset_padding(questions, sent_len=q_sent_len) 
answers = dataset_padding(answers, sent_len=a_sent_len) 
index = Index()
embed_params = Parameters(**{'class_num': 2,#y_train.shape[1],         # 类别数量
                       'learning_rate': 0.01,    # 学习率
                       'vocab_size':len(vocab_to_int),
                       'embed_dim':5,#100
                       'max_q_sent_len': q_sent_len,
                       'max_a_sent_len': a_sent_len,
                       'embed_matrix':embedding_matrix,
                       'rnn_dims': [4],#256
                       'isBiRNN':False,
                       'l2reg':0.001,
                       'index2authorid':index2authorid,
                       'author_questions':index.author_questions,
                       'question2index':index.question2index,
                       'author_answers':index.author_answers,
                       'answer2index':index.answer2index,
                       'voteups': author_features,
                       'refine':False,
                       })
embed = Embedding_FC(embed_params)
placeholders = embed.placeholder_dict()
sess = tf.Session()

def evaluate2(questions, answers, labels, mask, placeholders):
    t_test = time.time()
    # 构造验证的feed_dict
    feed_dict_val = construct_feed_dict2(questions, answers, labels, mask, placeholders)
    outs_val = sess.run([embed.loss, embed.accuracy, embed.report], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1],outs_val[2],(time.time() - t_test)

sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('./log', sess.graph)
cost_val = []
x_train, y_train, x_val, y_val, x_test, y_test = dataset_split(mask, labels, train_percent=1)
ffeatures = None

for epoch in range(20):
    t = time.time()#【返回当前时间的时间戳】
    # 构造训练feed_dict#features, 
    feed_dict = construct_feed_dict2(questions, answers, y_train, x_train,  placeholders)
    feed_dict.update({placeholders['dropout']: dropout})
    # Training step
    outs = sess.run([embed.optimizer, embed.loss, embed.accuracy, embed.predictions], feed_dict=feed_dict)
    '''
    # 验证--------------features, 
    cost, acc, _,duration = evaluate2(questions,answers, y_val, x_val, placeholders)
    cost_val.append(cost)
    # Print results  
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))    
    # early stop -------------
    if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping+1):-1]):
        print("Early stopping...")
        break
    '''
    ffeatures = outs[3]
print("Optimization Finished!")

t_features = np.hstack((ffeatures,author_features))   
    
gc.collect()
num_supports = 1# num_supports 矩阵多项式的项数
support = [preprocess_adj(adj)]  # 矩阵多项式各项，renormalization
labels = labels2onehot(labels_list, class_num = 2)
params = Parameters(**{'num_supports': 1,      # 卷积核多项式最高次数
                       'class_num': 2,         # 类别数量
                       'feature_size': features[2],       # 特征维度
                       'hidden_dims': [16,16],     # 各隐层输出维度
                       'weight_decay': 5e-4,  # L2正则化参数
                       'learning_rate': 0.01,   # 学习率
                       'feature2': t_features
                       })

model = Model(params)
placeholders = model.placeholder_dict()
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()

    # 构造验证的feed_dict
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)

    outs_val = sess.run([model.loss, model.accuracy, model.report], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1],outs_val[2], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('./log', sess.graph)
accc=[]
# 训练==================
for i in range(10):
    cost_val = []
    y_train, y_val, y_test, train_mask, val_mask, test_mask = divide_data(labels)
    for epoch in range(epochs):
    
        t = time.time()#【返回当前时间的时间戳】
        # 构造训练feed_dict
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: dropout})
    
        # Training step
        outs = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed_dict)
    
        # 验证--------------
        cost, acc,_, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
    
        # Print results
        
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
        
        # early stop -------------
        if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping+1):-1]):
            print("Early stopping...")
            break
    
    print("Optimization Finished!")
    
    # 测试==================
    test_cost, test_acc, report,test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    
    idx_test = range(2000, 3200)#(400,450)#
    print(classification_report(report[0][idx_test], report[1][idx_test]))
    accc.append(test_acc)
avg_acc = np.array(accc).sum()/10
print(accc)
print(avg_acc)
