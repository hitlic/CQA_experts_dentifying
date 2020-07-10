# coding: utf-8
"""
训练
"""
from __future__ import division
from __future__ import print_function


import time
import tensorflow as tf
from scipy.sparse import  lil_matrix
from model import Model
from data_process import *
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec
import jieba

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
G, index2authorid,node2id= build_net()
voteups, labels_list= loaddata(G)
adj = graph2sparse_max(G)
questions1 = get_question_title()
answers1 = get_answer_content()
texts1 = questions1 + answers1

index = Index()

model = Doc2Vec.load('doc2vec_10_1000_new') 
doc, node2answerid = get_answer(index2authorid)
text_feature = get_text_feature(model,index2authorid,node2id)
t_feature = []
for i in range(0,3263):
    t_feature.append(text_feature[i])
features = np.hstack((t_feature,voteups))
# 先对特征矩阵行归一化，然后以元组形式输出(稀疏阵元素坐标, 稀疏阵元素值, 稀疏阵shape)
features = preprocess_features(features)#【shape是(2708*1433)，但实际上是用三个dense tensor表示？】
support = [preprocess_adj(adj)]  # 矩阵多项式各项，renormalization
num_supports = 1# num_supports 矩阵多项式的项数

labels = labels2onehot(labels_list, class_num = 2)

params = Parameters(**{'num_supports': 1,      # 卷积核多项式最高次数
                       'class_num': 2,         # 类别数量
                       'feature_size': features[2],       # 特征维度
                       'hidden_dims': [16],     # 各隐层输出维度
                       'weight_decay': 5e-4,  # L2正则化参数
                       'learning_rate': 0.01    # 学习率
                       })
#【在形参前面加上*与**，称为动态参数；
#加*时，函数可接受任意多个参数，全部放入一个元祖中；
#加**时，函数接受参数时，返回为字典】

# 创建模型，input_dim为节点特征的维度，logging控制tf的histogram日志是否开启
model = Model(params)
placeholders = model.placeholder_dict()

sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()

    # 构造验证的feed_dict
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.report], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1],outs_val[2],(time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('./log', sess.graph)


accc=[]
for i in range(10): 
    y_train, y_val, y_test, train_mask, val_mask, test_mask = divide_data(labels)
    cost_val = []
    # 训练==================
    for epoch in range(epochs):
    
        t = time.time()#【返回当前时间的时间戳】
        # 构造训练feed_dict
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: dropout})
    
        # Training step
        outs = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed_dict)
    
        # 验证--------------
        cost, acc, _,duration = evaluate(features, support, y_val, val_mask, placeholders)
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
    test_cost, test_acc, report, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    idx_test = range(2000, 3200)#(400,450)#
    print(classification_report(report[0][idx_test], report[1][idx_test]))
    accc.append(test_acc)
avg_acc = np.array(accc).sum()/10
print(accc)
print(avg_acc)