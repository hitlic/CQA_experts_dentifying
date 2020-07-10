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
        params.hidden_dims.append(params.class_num)  # 输出维度（类别个数）

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
                H = tf.nn.dropout(_input, 1 - self.dropout)

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
            input_dim = self.params.feature_size[1]
            layer_input = _inputs

            dim1 = self.params.hidden_dims[0]#【=16，隐层维度】
            dim2 = self.params.hidden_dims[1]#【=7是输出维度，即类别个数】

            H = self.gcn_layer_gen(layer_input, input_dim, dim1, 'layer1', tf.nn.relu, True, True)
            H = self.gcn_layer_gen(H, dim1, dim2, 'layer2', lambda x: x, False, False)

            outputs = H
        return outputs

    def output_layer(self, _inputs):
        """
        输出层
        """
        outputs = _inputs
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
        self.outputs = self.output_layer(gcn_outputs)
        self.predict = tf.nn.softmax(self.outputs)

        self.set_loss()
        self.set_optimizer()
        self.set_accuracy()
