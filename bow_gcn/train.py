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

epochs = 200
dropout = 0.5
early_stopping = 10     # Tolerance for early stopping (# of epochs).

questions1 = get_question_title()
answers1 = get_answer_content()
texts = questions1 + answers1
#cv = CountVectorizer(max_features = 20)#创建词袋数据结构
cv = TfidfVectorizer(binary=False,decode_error='ignore',max_features = 20)
cv.fit(texts)
q_cv_fit=cv.transform(questions1)
a_cv_fit=cv.transform(answers1)
q = q_cv_fit.toarray()
a = a_cv_fit.toarray()

G, index2authorid = build_net()
adj = graph2sparse_max(G)

index = Index()
features_list, labels_list= loaddata(G)
features = np.array(features_list)
#features = lil_matrix(features_list)
labels = labels2onehot(labels_list, class_num = 2)

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
features = np.hstack((features_list, f))
# 先对特征矩阵行归一化，然后以元组形式输出(稀疏阵元素坐标, 稀疏阵元素值, 稀疏阵shape)
features = preprocess_features(features)#【shape是(2708*1433)，但实际上是用三个dense tensor表示？】
support = [preprocess_adj(adj)]  # 矩阵多项式各项，renormalization
num_supports = 1# num_supports 矩阵多项式的项数


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