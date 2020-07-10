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
import utils
import metrics
logger = logging.getLogger('main.dnn_model')


class Model:
    def __init__(self, params=None):
        if params and not isinstance(params, utils.Parameters):
            raise Exception(f'hyper_params must be an object of {type(utils.Parameters)} --- by LIC')
        # 默认参数
        default_params = {
            'num_supports': 1,      # 卷积核多项式最高次数
            'class_num': 2,         # 类别数量
            'hidden_dims': [16],    # 各隐层输出维度
            'weight_decay': 0.0005  # L2正则化参数
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
            # 卷积核多项式
            self.supports = [tf.sparse_placeholder(
                tf.float32, name=f"support{i}") for i in range(self.params.num_supports)]
            # 特征矩阵
            self.features = tf.sparse_placeholder(tf.float32, shape=tf.constant(
                self.params.feature_size, dtype=tf.int64), name="features")
            
            # 标签矩阵
            self.labels = tf.placeholder(tf.float32, shape=(None, self.params.class_num), name="labels")
            # dropout
            self.dropout = tf.placeholder_with_default(0., shape=(), name="dropout")
            # 标签mask，用于标出训练实例
            self.labels_mask = tf.placeholder(tf.int32, name="labels_mask")
            # 非零特征数量
            self.num_features_nonzero = tf.placeholder(
                tf.int32, name="num_features_nonzero")  # helper variable for sparse dropout

        return self.features

    def placeholder_dict(self):
        """
        返回模型中的placeholders
        """
        placeholders = {}
        placeholders['labels'] = self.labels
        placeholders['labels_mask'] = self.labels_mask
        placeholders['features'] = self.features
        placeholders['support'] = self.supports
        placeholders['num_features_nonzero'] = self.num_features_nonzero
        placeholders['dropout'] = self.dropout
        return placeholders

    def gcn_layer_gen(self, _input, input_dim, output_dim, layer_name, act_fun, sparse_input=False, record_vars=False):
        """
        GCN层计算图
        """
        with tf.variable_scope(layer_name):
            THWs = []

            if sparse_input:#【如果使用了稀疏矩阵，则需去除特征向量中的空值】
                H = utils.sparse_dropout(_input, 1 - self.dropout, self.num_features_nonzero)#【去除含空值之后维度不会出问题吗？】
            else:
                H = tf.nn.dropout(_input, 1-self.dropout)

            for i, suport in enumerate(self.supports):
                W = inits.glorot([input_dim, output_dim], name=f'{layer_name}_weight_{i}') 
                #【第一轮：名为：layer1 W为(1433*16) 第二轮：名为：layer2 W为(16*7)】
                if record_vars:
                    self.vars.append(W)
                HW = utils.dot(H, W, sparse=sparse_input)
                THW = utils.dot(suport, HW, sparse=True)
                THWs.append(THW)
            output = tf.add_n(THWs)#【THW列表元素相加】
            # bias = inits.zeros([output_dim], name='bias')
            # output += bias
            output = act_fun(output)
        return output

    def gcn_layers(self, _inputs):
        """
        利用GCN计算图，构建一个或多个GCN层
        """
        with tf.name_scope('gcn_layers'):
            input_dim = self.params.feature_size[1]#【=1433】
            layer_input = _inputs

            dim1 = self.params.hidden_dims[0]#【=16，隐层维度】
            dim2 = self.params.hidden_dims[1]#【=7是输出维度，即类别个数】

            H = self.gcn_layer_gen(layer_input, input_dim, dim1, 'layer1', tf.nn.relu, True, True)
            H = self.gcn_layer_gen(H, dim1, dim2, 'layer2', tf.nn.relu, False, False) #lambda x: x,

            outputs = H
        return outputs
    
    def fc_layer(self, inputs):
        """
        全连接层
        """
        # initializer = tf.contrib.layers.xavier_initializer()  # xavier参数初始化，暂没用到
        with tf.name_scope("fc_layer"):
            inputs = tf.concat([inputs, self.params.feature2], axis = 1, name='concat')
            inputs = tf.nn.dropout(inputs, keep_prob= 1-self.dropout, name='drop_out')  # dropout
            # outputs = tf.contrib.layers.fully_connected(inputs, self.hypers.fc_size, activation_fn=tf.nn.relu)
            outputs = tf.layers.dense(inputs, 32, activation=tf.nn.relu)
        return outputs    
    def fc_layer2(self, inputs):
        """
        全连接层
        """
        # initializer = tf.contrib.layers.xavier_initializer()  # xavier参数初始化，暂没用到
        with tf.name_scope("fc_layer2"):
            inputs = tf.concat([inputs, self.params.feature2], axis = 1, name='concat')
            inputs = tf.nn.dropout(inputs, keep_prob= 1-self.dropout, name='drop_out')  # dropout
            # outputs = tf.contrib.layers.fully_connected(inputs, self.hypers.fc_size, activation_fn=tf.nn.relu)
            outputs = tf.layers.dense(inputs, 32, activation=tf.nn.relu)
        return outputs    
    def output_layer(self, inputs):
        """
        输出层
        """
        with tf.name_scope("output_layer"):
            #outputs = tf.nn.softmax(inputs)
            inputs = tf.layers.dropout(inputs, rate= 1-self.dropout)
            outputs = tf.layers.dense(inputs, 2, activation=None)
        return outputs

    def set_loss(self):
        """
        损失函数
        """
        # Weight decay loss
        with tf.name_scope("loss"):
            self.loss = 0
            for var in self.vars:
                self.loss += self.params.weight_decay * tf.nn.l2_loss(var)
            # Cross entropy error
            self.loss += metrics.masked_softmax_cross_entropy(self.outputs, self.labels, self.labels_mask)

    def set_accuracy(self):
        """
        准确率
        """
        with tf.name_scope('accuracy'):
            self.accuracy = metrics.masked_accuracy(self.outputs, self.labels, self.labels_mask)
            self.report = [tf.argmax(self.labels, 1), tf.argmax(self.outputs, 1)]

    def set_optimizer(self):
        """
        优化器
        """
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss)

    def build(self):
        """
        DNN模型构建
        """
        inputs = self.inputs_layer()
        gcn_outputs = self.gcn_layers(inputs)
        outputs = self.fc_layer(gcn_outputs)
        outputs = self.fc_layer2(outputs)
        self.outputs = self.output_layer(outputs)
        self.predict = tf.nn.softmax(self.outputs)

        self.set_loss()
        self.set_optimizer()
        self.set_accuracy()


class Embedding:
    def __init__(self, params=None):
        self.params = params  # 所有模型需要的参数都保存在其中
        self.build()
    
    def inputs_layer(self):
        with tf.name_scope('input_layer'):
            self.questions = tf.placeholder(tf.int32, [None, self.params.max_q_sent_len], name='questions')
            self.answers = tf.placeholder(tf.int32, [None, self.params.max_a_sent_len], name='answers')
        return self.questions,self.answers
    
    
    def placeholder_dict(self):
        placeholders = {}
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
            q_embed = tf.reduce_sum(q_embed, 2)
            a_embed = tf.reduce_sum(a_embed, 2)
        return q_embed,a_embed#tf.concat([q_embed, a_embed], axis=-1)
    
    
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
            t1 = t1/len(self.params.author_questions[self.params.index2authorid[i]])
            for answer_id in self.params.author_answers[self.params.index2authorid[i]]:
                t2 += a_rnn_outputs[self.params.answer2index[answer_id]]
            t2 = t2/len(self.params.author_answers[self.params.index2authorid[i]])
            t = tf.concat([[t1],[t2]], axis = 1, name='concat')
            features_list.append(t[0])
        features = tf.stack(features_list, axis = 0)
        features = tf.concat([features, self.params.author_features], axis = 1, name='concat')
        #features = tf.reshape(features,shape=[-1,features.shape[2]],name=None)
        return features
    
    def build(self):
        questions,answers = self.inputs_layer()
        q_embedding, a_embedding = self.embedding_layer(questions,answers)
        self.outputs = self.sort_feature(q_embedding, a_embedding)
