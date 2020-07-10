import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from scipy.sparse import  lil_matrix
import tensorflow as tf

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => 训练实例（节点）特征向量，scipy.sparse.csr.csr_matrix
    ind.dataset_str.tx => 测试实例特征向量，scipy.sparse.csr.csr_matrix
    ind.dataset_str.allx => labeled and unlabeled training instances
        (a superset of ind.dataset_str.x)， scipy.sparse.csr.csr_matrix
    ind.dataset_str.y => 已标记训练实例的one-hot标记，numpy.ndarray
    ind.dataset_str.ty => 已标记测试实例的one-hot标记，numpy.ndarray
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => 网络，{index: [index_of_neighbor_nodes]} as collections.defaultdict object;
    ind.dataset_str.test.index => graph中测试集的索引, for the inductive setting as list object.

    所有数据需保存为python pickle

    :param dataset_str: 数据集名字
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    #【allx的shape：(1708, 1433)；ally的shape：(1708, 7)】
    objects = []

    # 读入各数据
    for name in names:
        with open("data/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects) # pylint: disable=unbalanced-tuple-unpacking
    #print(ally.shape)
    #p = np.sum(ally,axis=1)
    # 导入测试集下标
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)  # 下标排序

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()  # 换一种稀疏矩阵格式lil(适合逐个添加元素，并且能快速获取行相关的数据)
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # 邻接矩阵
    #print(adj.shape)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]#【将labels中test_idx_reorder对应的行转成test_idx_range对应的行，例如将labels的1708行移到2692行】

    idx_test = test_idx_range.tolist()#【1708~2707】
    idx_train = range(len(y))#【0~139】
    idx_val = range(len(y), len(y)+500)#【140~639】
    
    train_mask = sample_mask(idx_train, labels.shape[0])#【0~139行标记为1】
    val_mask = sample_mask(idx_val, labels.shape[0])#【140~639标记为1】
    test_mask = sample_mask(idx_test, labels.shape[0])#【1078~2707标记为1】

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]#【0~139行赋值为labels的0~139】
    y_val[val_mask, :] = labels[val_mask, :]#【140~639赋值为labels的140~639】
    y_test[test_mask, :] = labels[test_mask, :]#【1078~2707赋值为labels的1078~2707】

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.
    将稀疏矩阵转为元组表示

    TensorFlow使用三个dense tensor来表达一个sparse tensor：indices、values、dense_shape。
        假如我们有一个dense tensor：
            [[1, 0, 0, 0]
            [0, 0, 2, 0]
            [0, 0, 0, 0]]
        那么用SparseTensor表达这个数据对应的三个dense tensor如下：
            indices：[[0, 0], [1, 2]]
            values：[1, 2]
            dense_shape：[3, 4]
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):#【如果mx不是是coo_matrix类型】
            mx = mx.tocoo()#【转换为coo_matrix格式】
        coords = np.vstack((mx.row, mx.col)).transpose()#【13264*2】
        values = mx.data#【132664】
        shape = mx.shape#【2708*2708】
        return coords, values, shape

    if isinstance(sparse_mx, list):#【isinstance() 函数来判断一个对象是否是一个已知的类型】
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation
    特征矩阵行归一化，【特征逆阵*特征阵】
    rowsum = np.array(features.sum(1))#【axis=0按列相加；axis=1按行相加】
    r_inv = np.power(rowsum, -1).flatten()#【numpy.power(x1, x2)对x1求x2次方。x2可以是数字，也可以是数组，但是x1和x2的列数要相同。】
    r_inv[np.isinf(r_inv)] = 0.#【返回一个判断是否是无穷的bool型数组，是的话赋值为0】
    r_mat_inv = sp.diags(r_inv)#【用r_inv构建对角阵】
    features = r_mat_inv.dot(features)#【r_mat_inv点乘features】
    #print(features.shape )
    
    """
    
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    for i in range(features.shape[1]):
        #a = (features[:,i] - mean)/std
        if mean[i] != 0 and std[i] != 0:
            features[:,i] = (features[:,i] - mean[i])/std[i]
    #features = lil_matrix(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.
    论文renormalization，并稀疏阵坐标形式输出
    """
    adj = sp.coo_matrix(adj)#【这一步有什么用？为什么一定要转换成coo格式？】
    rowsum = np.array(adj.sum(1))#【adj按列相加】
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()#【对每一个数求-0.5次幂】
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.#【返回一个判断是否是无穷的bool型数组，是的话赋值为0】
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()#【D^(1/2)AD^(1/2)】


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    论文中的renormalization trick，内容位于式（7）（8）之间
    以三元组形式输出，(稀疏阵元素坐标, 稀疏阵元素值, 稀疏阵shape)
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))#【先A=A+I(单位阵)，后计算D^(1/2)AD^(1/2)】
    return sparse_to_tuple(adj_normalized)#【之后转换SparseTensor】


def construct_feed_dict(questions, answers, support, labels, labels_mask,num_features_nonzero, placeholders):
    """Construct feed dictionary."""#features,
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    #feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))}) 
    
    feed_dict.update({placeholders['questions']: questions})
    feed_dict.update({placeholders['answers']: answers})
    feed_dict.update({placeholders['num_features_nonzero']: num_features_nonzero})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for _ in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors.
    稀疏张量dropout
    """
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)#【tf.floor向下取整；tf.cast强制类型转换为bool】
    pre_out = tf.sparse_retain(x, dropout_mask)#【去除x中的空值】
    return pre_out * (1./keep_prob)#【为什么不是*keepprob】


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense).
    矩阵相乘，可正常矩阵，也可以稀疏矩阵
    """
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Parameters:
    '''
    超参数对像，用于存储与管理各种参数
    '''

    def __init__(self, **args):
        self.__dict__.update(args)  # 将参数加入到self中

    def __add__(self, hps):
        '''
        重载+，使两个超参对像可以相加
        '''
        if not isinstance(hps, Parameters):
            raise Exception(f'{type(self)} and {type(hps)} cannot be added together！！！ --- by LIC ')
        param_dict = dict()
        param_dict.update(self.__dict__)
        param_dict.update(hps.__dict__)
        return Parameters(** param_dict)

    def to_str(self, short=False):
        '''
        输出参数为字符串
        '''
        params = sorted(self.__dict__.items(), key=lambda item: item[0])
        output = ''
        for param, value in params:
            if short:
                output += f'{param}-{str(value)[:7]}__'
            else:
                output += f'{param}-{value}__'
        return output[:-2]

    def __str__(self):
        return self.to_str()

    def to_dict(self):
        '''
        将全部超参数输出为字典
        '''
        return self.__dict__

    def get(self, attr_name):
        '''
        获取参数值，若不存在返回None
        '''
        return self.__dict__.get(attr_name)

    def set(self, key, value):
        '''
        添加或更新一个参数
        '''
        self.__dict__[key] = value

    def default(self, default_params):
        '''
        设置默认参数，仅添加缺失的参数，不改变已有参数
        '''
        for key, value in default_params.items():
            if self.__dict__.get(key) is None:
                self.__dict__[key] = value

    def update(self, params):
        '''
        仅更新已有参数，不添加新参数
        '''
        param_set = params
        if isinstance(params, Parameters):
            param_set = params.to_dict()
        for key, value in param_set.items():
            if self.__dict__.get(key) is not None:
                self.__dict__[key] = value

    def extend(self, params):
        '''
        添加新参数，同时更新已有参数
        '''
        self.__dict__.update(params)