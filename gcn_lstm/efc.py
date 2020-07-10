# coding: utf-8
"""
模型
@author Liuchen
2019
"""
import tensorflow as tf
import inits
import time
import logging
import numpy as np
import utils

from scipy.sparse import  lil_matrix
import metrics
logger = logging.getLogger('main.dnn_model')


class Embedding_FC:
    def __init__(self, params=None):
        if params and not isinstance(params, utils.Parameters):
            raise Exception(f'hyper_params must be an object of {type(utils.Parameters)} --- by LIC')
        # 默认参数
        default_params = {
            'class_num': 2,         # 类别数量
        }
        params.default(default_params)
        #params.hidden_dims.append(params.class_num)  # 输出维度（类别个数）

        self.params = params  # 所有模型需要的参数都保存在其中

        self.name = f'gcn{time.time()}'
        self.vars = []

        self.build()  # 构建模型

    def inputs_layer(self):
        """
        输入层
        定义placeholder
        """
        with tf.name_scope('input_layer'):
            self.labels = tf.placeholder(tf.int32, shape=[None, self.params.class_num], name="labels")
            self.dropout = tf.placeholder_with_default(0., shape=(), name="dropout")
            self.mask = tf.placeholder(tf.int32, shape=[None, 1], name="mask")
            self.questions = tf.placeholder(tf.int32, [None, self.params.max_q_sent_len], name='questions')
            self.answers = tf.placeholder(tf.int32, [None, self.params.max_a_sent_len], name='answers')

        return self.questions,self.answers
    
    def placeholder_dict(self):
        """
        返回模型中的placeholders
        """
        placeholders = {}
        placeholders['labels'] = self.labels
        placeholders['mask'] = self.mask
        placeholders['dropout'] = self.dropout
        placeholders['questions'] = self.questions
        placeholders['answers'] = self.answers
        return placeholders

    def embedding_layer(self, questions, answers):
        """
        词向量层
        """
        with tf.name_scope("embedding_layer"):
            if self.params.get('embed_matrix') is None:   # 若无已训练词向量
                q_embedding = tf.Variable(tf.random_uniform((self.params.vocab_size,
                                                           self.params.embed_dim), -1, 1), name="q_sentence_embed_matrix")
                a_embedding = tf.Variable(tf.random_uniform((self.params.vocab_size,
                                                           self.params.embed_dim), -1, 1), name="a_sentence_embed_matrix")
            else:                           # 若已有词向量
                q_embedding = tf.Variable(self.params.embed_matrix,
                                        trainable=self.params.refine, name="q_sentence_embed_matrix")
                a_embedding = tf.Variable(self.params.embed_matrix,
                                        trainable=self.params.refine, name="a_sentence_embed_matrix")
            
            q_embed= tf.nn.embedding_lookup(q_embedding, questions)
            a_embed = tf.nn.embedding_lookup(a_embedding, answers)
            #q_embed = tf.reduce_sum(q_embed, 2)
            #a_embed = tf.reduce_sum(a_embed, 2)
        return q_embed,a_embed#tf.concat([q_embed, a_embed], axis=-1)

 
    def q_rnn_layer(self, embed):
        """
        RNN层
        """
        with tf.variable_scope("q_rnn_layer"):#tf.name_scope("rnn_layer"):
            embed = tf.nn.dropout(embed, keep_prob=1-self.dropout)  # dropout
            # --- 可选的RNN单元
            # tf.contrib.rnn.BasicRNNCell(size)
            # tf.contrib.rnn.BasicLSTMCell(size)
            # tf.contrib.rnn.LSTMCell(size)
            # tf.contrib.rnn.GRUCell(size, activation=tf.nn.relu)
            # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
            # tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(size)

            if not self.params.isBiRNN:
                lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.params.rnn_dims]
                drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1-self.dropout) for lstm in lstms]
                cell = tf.contrib.rnn.MultiRNNCell(drops)  # 组合多个 LSTM 层
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
                # lstm_outputs -> batch_size * max_len * n_hidden
            else:
                lstms_l = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.params.rnn_dims]
                lstms_r = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.params.rnn_dims]
                drops_l = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1-self.dropout) for lstm in lstms_l]
                drops_r = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1-self.dropout) for lstm in lstms_r]
                cell_l = tf.contrib.rnn.MultiRNNCell(drops_l)
                cell_r = tf.contrib.rnn.MultiRNNCell(drops_r)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(  # 双向LSTM
                    cell_l,  # 正向LSTM单元
                    cell_r,  # 反向LSTM单元
                    inputs=embed,
                    dtype=tf.float32,
                )  # outputs -> batch_size * max_len * n_hidden; state(最终状态，为h和c的tuple) -> batch_size * n_hidden
                lstm_outputs = tf.concat(outputs, -1)  # 合并双向LSTM的结果
            outputs = lstm_outputs[:, -1]  # 返回每条数据的最后输出
        return outputs
    
    
    def a_rnn_layer(self, embed):
        """
        RNN层
        """
        with tf.variable_scope("a_rnn_layer"):#tf.name_scope("rnn_layer"):
            embed = tf.nn.dropout(embed, keep_prob=1-self.dropout)  # dropout
            # --- 可选的RNN单元
            # tf.contrib.rnn.BasicRNNCell(size)
            # tf.contrib.rnn.BasicLSTMCell(size)
            # tf.contrib.rnn.LSTMCell(size)
            # tf.contrib.rnn.GRUCell(size, activation=tf.nn.relu)
            # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
            # tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(size)

            if not self.params.isBiRNN:
                lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.params.rnn_dims]
                drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1-self.dropout) for lstm in lstms]
                cell = tf.contrib.rnn.MultiRNNCell(drops)  # 组合多个 LSTM 层
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
                # lstm_outputs -> batch_size * max_len * n_hidden
            else:
                lstms_l = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.params.rnn_dims]
                lstms_r = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.params.rnn_dims]
                drops_l = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1-self.dropout) for lstm in lstms_l]
                drops_r = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1-self.dropout) for lstm in lstms_r]
                cell_l = tf.contrib.rnn.MultiRNNCell(drops_l)
                cell_r = tf.contrib.rnn.MultiRNNCell(drops_r)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(  # 双向LSTM
                    cell_l,  # 正向LSTM单元
                    cell_r,  # 反向LSTM单元
                    inputs=embed,
                    dtype=tf.float32,
                )  # outputs -> batch_size * max_len * n_hidden; state(最终状态，为h和c的tuple) -> batch_size * n_hidden
                lstm_outputs = tf.concat(outputs, -1)  # 合并双向LSTM的结果
            outputs = lstm_outputs[:, -1]  # 返回每条数据的最后输出
        return outputs
    
    def sort_feature(self, q_rnn_outputs, a_rnn_outputs):
        '''
        找到author按照index顺序排列的每个人回答的qustion_id的下标：
        author_index -> author_id -> question_id -> question_index
        因为author_id和question_id都是一大串，question存成list 的时候就产生了一个下标顺序，
        构建网络的时候也对每个author（即节点）添加了节点的index属性，
        邻接矩阵按照author的index的顺序构造的，
        所以特征矩阵也需要按照author的index顺序排列。
        '''
        features_list = []
        for i in range(len(self.params.index2authorid)):
            #t1 = tf.zeros([1,q_rnn_outputs.shape[1]])
            t1 = [0 for j in range(q_rnn_outputs.shape[1])]
            #t2 = tf.zeros([1,a_rnn_outputs.shape[1]])
            t2 = [0 for j in range(a_rnn_outputs.shape[1])]
            for question_id in self.params.author_questions[self.params.index2authorid[i]]:
                t1 += q_rnn_outputs[self.params.question2index[question_id]]
            for answer_id in self.params.author_answers[self.params.index2authorid[i]]:
                t2 += a_rnn_outputs[self.params.answer2index[answer_id]]
            #t = tf.concat([[t1],[t2]], axis = 1, name='concat')
            features_list.append([t2][0])
            #features_list.append(t[0])
        features = tf.stack(features_list, axis = 0)
        features = tf.concat([features, self.params.voteups], axis = 1, name='concat')
        
        #features = self.params.voteups
        self.labels = tf.gather(self.labels,self.mask)
        features= tf.gather(features,self.mask)
        
        self.labels = tf.reshape(self.labels,shape=[-1,2],name=None)
        features = tf.reshape(features,shape=[-1,features.shape[2]],name=None)
        return features
    

    def fc_layer(self, inputs):
        """
        全连接层
        """
        # initializer = tf.contrib.layers.xavier_initializer()  # xavier参数初始化，暂没用到
        with tf.name_scope("fc_layer"):
            inputs = tf.nn.dropout(inputs, keep_prob= 1-self.dropout, name='drop_out')  # dropout
            # outputs = tf.contrib.layers.fully_connected(inputs, self.hypers.fc_size, activation_fn=tf.nn.relu)
            outputs = tf.layers.dense(inputs, 10, activation=tf.nn.relu)
        return outputs


    def output_layer2(self, inputs):
        with tf.name_scope("output_layer"):
            inputs = tf.layers.dropout(inputs, rate= 1-self.dropout)
            outputs = tf.layers.dense(inputs, 2, activation=None)
            #outputs = tf.reshape(outputs,shape=[-1,2],name=None)
            # outputs = tf.contrib.layers.fully_connected(inputs, self.hypers.class_num, activation_fn=None)
        return outputs
    
    
    def set_loss2(self):
        """
        损失函数
        """
        # softmax交叉熵损失
        with tf.name_scope("loss_scope"):
            reg_loss = tf.contrib.layers.apply_regularization(  # L2正则化
                tf.contrib.layers.l2_regularizer(self.params.l2reg),
                tf.trainable_variables()
            )
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.predictions, labels=self.labels)) + reg_loss   # ---GLOBAL---损失函数

    def set_accuracy2(self):
        """
        准确率
        """
        with tf.name_scope("accuracy_scope"):
            correct_pred = tf.equal(tf.argmax(self.predictions, axis=1), tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # ---GLOBAL---准确率
            self.report = [tf.argmax(self.labels, 1), tf.argmax(self.predictions, 1)]
    def set_optimizer2(self):
        """
        优化器
        """
        with tf.name_scope("optimizer"):
            # --- 可选优化算法
            # self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


    def build(self):
        """
        DNN模型构建
        """
        questions,answers = self.inputs_layer()
        q_embedding, a_embedding = self.embedding_layer(questions,answers)
        q_rnn_outputs = self.q_rnn_layer(q_embedding)
        a_rnn_outputs = self.a_rnn_layer(a_embedding)
        features = self.sort_feature(q_rnn_outputs, a_rnn_outputs)
        #features = self.sort_feature(q_embedding, a_embedding)
        self.a = self.fc_layer(features)        
        self.predictions = self.output_layer2(self.a)
        #self.predictions = tf.nn.softmax(self.outputs)

        self.set_loss2()
        self.set_optimizer2()
        self.set_accuracy2()
