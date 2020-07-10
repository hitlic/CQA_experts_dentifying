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
from itertools import chain
from sklearn.metrics import classification_report
from data_process import *


# Set random seed
#seed = 123
#np.random.seed(seed)
#tf.set_random_seed(seed)


# Settings
#dataset = 'cora'     # 'cora', 'citeseer', 'pubmed'
epochs = 200
dropout = 0.5
early_stopping = 10     # Tolerance for early stopping (# of epochs).
max_degree = 3          # 'Maximum Chebyshev polynomial degree. 切多雪夫多项式的最高次数'


# Load dataq
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
#以cora数据集为例：
#  	 adj：邻接矩阵(2078*2078)，print(adj.shape)【2708*2708】
# 	 features：特征(2078*1433)，词典大小1433【特征维度：1433】
# 	 y_train：训练集标签(2708*7)，共7个类别
# 	 y_val：验证集标签(2708*7)
# 	 y_test：测试集标签(2708*7)
# 	 train_mask, val_mask, test_mask：(2708,)布尔数组


#==================================
questions1 = get_question_title()
answers1 = get_answer_content()
texts = questions1 + answers1
texts = sentences2wordlists(texts, lang='CH')
embedding_matrix = None  # 设词向量矩阵为None
vocab_to_int, embedding_matrix = load_embedding("./data/sgns.weibo.word")  # 英文词向量
#vocab_to_int, int_to_vocab = make_dictionary_by_text(list(chain.from_iterable(texts)))
questions = wordlists2idlists(questions1, vocab_to_int)
answers = wordlists2idlists(answers1, vocab_to_int)
q_sent_len = 10#50
a_sent_len = 50#5000
questions = dataset_padding(questions, sent_len=q_sent_len) 
answers = dataset_padding(answers, sent_len=a_sent_len) 
index = Index()
#==================================
G, index2authorid = build_net()
num_features_nonzero = len(index2authorid)
'''
print(index2authorid[5])
print(index.author_questions[index2authorid[5]])
for i in index.author_questions[index2authorid[5]]:
    print(index.question2index[i])
    print(questions1[index.question2index[i]])
print(index.author_answers[index2authorid[5]])
for i in index.author_answers[index2authorid[5]]:
    print(index.answer2index[i])
    print(answers1[index.answer2index[i]])
'''
adj = graph2sparse_max(G)
voteups, labels_list= loaddata(G)
#features = lil_matrix(features_list)
labels = labels2onehot(labels_list, class_num = 2)
#y_train, y_val, y_test, train_mask, val_mask, test_mask = divide_data(labels)
# 先对特征矩阵行归一化，然后以元组形式输出(稀疏阵元素坐标, 稀疏阵元素值, 稀疏阵shape)
#features = preprocess_features(features)#【shape是(2708*1433)，但实际上是用三个dense tensor表示？】
support = [preprocess_adj(adj)]  # 矩阵多项式各项，renormalization
num_supports = 1# num_supports 矩阵多项式的项数
params = Parameters(**{'num_supports': 1,      # 卷积核多项式最高次数
                       'class_num': 2,         # 类别数量
                       #'feature_size': 80,#features[2],       # 特征维度
                       'hidden_dims': [16],     # 各隐层输出维度
                       'weight_decay': 5e-4,  # L2正则化参数
                       'learning_rate': 0.01,    # 学习率
                       #===========后加入的参数=============
                       'vocab_size':len(vocab_to_int),
                       'embed_dim':5,
                       'max_q_sent_len': q_sent_len,
                       'max_a_sent_len': a_sent_len,
                       'rnn_dims': [4],#256
                       'isBiRNN':False,
                       'index2authorid':index2authorid,
                       'author_questions':index.author_questions,
                       'question2index':index.question2index,
                       'author_answers':index.author_answers,
                       'answer2index':index.answer2index,
                       'voteups': voteups,
                       'embed_matrix':embedding_matrix,
					       'refine':False,
                       })
#【在形参前面加上*与**，称为动态参数；
#加*时，函数可接受任意多个参数，全部放入一个元祖中；
#加**时，函数接受参数时，返回为字典】

# 创建模型，input_dim为节点特征的维度，logging控制tf的histogram日志是否开启
model = Model(params)
placeholders = model.placeholder_dict()
sess = tf.compat.v1.Session()


# Define model evaluation function#features,
def evaluate(questions, answers, support, labels, mask, num_features_nonzero, placeholders):
    t_test = time.time()
    # 构造验证的feed_dict
    feed_dict_val = construct_feed_dict(questions, answers, support, labels, mask, num_features_nonzero, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.report], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1],outs_val[2],  (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('./log', sess.graph)


accc=[]

y_train, y_val, y_test, train_mask, val_mask, test_mask = divide_data(labels)
# 训练==================

for i in range(10):
    cost_val = []
    for epoch in range(epochs):
        t = time.time()#【返回当前时间的时间戳】
        # 构造训练feed_dict#features, 
        feed_dict = construct_feed_dict(questions, answers, support, y_train, train_mask, num_features_nonzero, placeholders)
        feed_dict.update({placeholders['dropout']: dropout})
        # Training step
        outs = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed_dict)
        # 验证--------------features, 
        cost, acc, _,duration = evaluate(questions,answers, support, y_val, val_mask, num_features_nonzero, placeholders)
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
    test_cost, test_acc, report,test_duration = evaluate(questions, answers, support, y_test, test_mask, num_features_nonzero, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    
    idx_test = range(2000, 3200)#(400,450)#
    print(classification_report(report[0][idx_test], report[1][idx_test]))

    accc.append(test_acc)
avg_acc = np.array(accc).sum()/10
print(accc)
print(avg_acc)