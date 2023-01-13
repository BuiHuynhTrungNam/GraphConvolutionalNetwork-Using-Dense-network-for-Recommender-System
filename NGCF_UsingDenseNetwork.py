#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))


# In[2]:


#!pip install sklearn


# In[3]:


'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''


# #Preparation

# ##Load data
# 
# Chuẩn bị data, đọc từ file tạo ma trận kề

# In[4]:



import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import collections

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        # self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        self.coo_R = self.R.tocoo()

        self.all_head_list, self.all_tail_list, self.all_v_list = self._get_all_entity_list_()

    def get_adj_mat(self):
        try:
            t1 = time()

            norm_adj_mat_1 = sp.load_npz(self.path + '/s_norm_adj_mat_1.npz')

            print('already load adj matrix', norm_adj_mat_1.shape, time() - t1)

        except Exception:
            norm_adj_mat_1 = self.create_adj_mat()

            sp.save_npz(self.path + '/s_norm_adj_mat_1.npz', norm_adj_mat_1)

        return norm_adj_mat_1

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_symetric(adj):
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            d_inv_sqrt_last = np.power(rowsum, -0.4).flatten()
            d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
            d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)

            return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt_last).tocoo()

        norm_adj_mat_1 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]))
        # norm_adj_mat_1 = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

        print('already normalize adjacency matrix', time() - t2)
        return norm_adj_mat_1.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def _get_all_entity_list_(self):

        all_head_list = list(self.coo_R.row)
        all_tail_list = list(self.coo_R.col)
        all_v_list = list(self.coo_R.data)

        org_h_dict = dict()
        for idx, h in enumerate(all_head_list):
            if h not in org_h_dict:
                org_h_dict[h] = [[], []]
            org_h_dict[h][0].append(all_tail_list[idx])
            org_h_dict[h][1].append(all_v_list[idx])

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = np.array(org_t_list)[sort_order]
            sort_v_list = np.array(org_v_list)[sort_order]

            sorted_h_dict[h] = [sort_t_list, sort_v_list]

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_v_list = [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_v_list += list(vals[1])

        return new_h_list, new_t_list, new_v_list


# #Unility

# ##Evaluation --metrics

# In[5]:


import numpy as np
from sklearn.metrics import roc_auc_score

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


# ##Helper

# In[6]:


import os
import re

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


# #Testing implementation

# In[7]:


class Arg:
  def __init__(self,dataset,regs='[1e-5,1e-5,1e-2]',embed_size=64,layer_size='[64,64,64]'               ,lr=0.001,save_flag=0,pretrain=0,batch_size=1024,epoch=1000,
               verbose=1,node_dropout='[0.1,0.1,0.1]',mess_dropout='[0.1,0.1,0.1]', Ks='[20, 40, 60, 80, 100]'):
    
    self.model='ngcf'

    #Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.
    self.alg_type='ngcf'

    # Model storage location
    self.weights_path=''

    # Data storage location
    self.data_path='../Data/'

    # Dataset name: gowalla, yelp2018, amazon-book
    self.dataset=dataset

    #Regularizations , default =[1e-5,1e-5,1e-2]
    self.regs=regs

    #Embedding size, default =64
    self.embed_size=embed_size

    #Output sizes of every layer, default =[64]
    self.layer_size=layer_size

    #Learning rate, default=0,01
    self.lr=lr

    #0: Disable model saver, 1: Activate model saver
    self.save_flag=save_flag

    # 0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.
    self.pretrain=pretrain
    
    #Training batch size
    self.batch_size=batch_size

    # Training epoch size
    self.epoch=epoch

    #Interval of evaluation, default=1
    self.verbose=verbose

    #0: Disable node dropout, 1: Activate node dropout
    self.node_dropout_flag=0
    #Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout
    self.node_dropout=node_dropout
    #Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.
    self.mess_dropout=mess_dropout

    #Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.
    self.adj_type='norm'

    #0 for NAIS_prod, 1 for NAIS_concat
    self.gpu_id=0

    #Output sizes of every layer
    self.Ks=Ks

    #Specify the test type from {part, full}, indicating whether the reference is done in mini-batch
    self.test_flag='part'

    #0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels
    self.report=0
    
    self.proj_path=''
    self.model_type=' '



# In[8]:


# import utility.metrics as metrics
# from utility.parser import parse_args
# from utility.load_data import *
import multiprocessing
import heapq
#import utility.metrics as metrics

cores = multiprocessing.cpu_count() // 2

args=Arg(dataset='gowalla', regs= '[1e-5,1e-5,1e-2]', embed_size=64, layer_size='[64,64,64]',lr= 0.00001         ,save_flag=1 ,pretrain=0 ,batch_size=1024 ,epoch=1000,verbose=1,node_dropout='[0.1,0.1,0.1]',mess_dropout='[0.1,0.1,0.1]')
#args = parse_args()
Ks = eval(args.Ks)


data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        # print(u_batch_id, len(user_batch))

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                else:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.]*len(eval(args.layer_size))})
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch})
            else:
                if len(user_batch) < u_batch_size:
                    user_batch_ = user_batch + [user_batch[-1]]*(u_batch_size - len(user_batch))
                    # print('After mask operation, len of user batch:{}'.format(len(user_batch_)))
                    rate_batch = sess.run(model.batch_ratings, {model.users: user_batch_,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.] * len(eval(args.layer_size))

                                                                })
                    rate_batch = rate_batch[0:len(user_batch), :]
                else:
                    rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.] * len(eval(args.layer_size))
                                                                })

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result


# In[9]:


len(data_generator.sample()[0]) # batch size


# #Model implementation

# In[10]:


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# from utility.helper import *
# from utility.batch_test import *

class RGCF(object):

    def __init__(self, data_config):
        # argument settings
        self.model_type = 'rgcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)

        self.n_layers = data_config['n_layers']
        print('aaaaaaaaaaaaaaaaaaaaaa')
        print(n_layers)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.decay = data_config['decay']

        self.verbose = args.verbose

        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.compat.v1.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.compat.v1.placeholder(tf.float32, shape=[None])

        # initialization of model parameters
        self.weights = self._init_weights()

        self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        # self.u_bias = tf.nn.embedding_lookup(self.weights['u_bias'], self.users)
        # self.pos_i_bias = tf.nn.embedding_lookup(self.weights['i_bias'], self.pos_items)
        # self.neg_i_bias = tf.nn.embedding_lookup(self.weights['i_bias'], self.neg_items)

        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,transpose_b=True)
        # + tf.matmul(self.weights['c'], self.weights['i_bias'], transpose_a=False, transpose_b=True)

        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        
        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        # all_weights['u_bias'] = tf.Variable(tf.zeros([self.n_users, 1]), name='u_bias')
        # all_weights['i_bias'] = tf.Variable(tf.zeros([self.n_items, 1]), name='i_bias')
        # all_weights['c'] = tf.ones([args.batch_size*2, 1])

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat


    def _create_ngcf_embed(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = ego_embeddings

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = side_embeddings

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            #bi_embeddings = bi_embeddings

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, rate=1 - (1 - 0.1))

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += norm_embeddings
            
             # convolutional layer.
            #embeddings = tf.nn.leaky_relu(tf.matmul(norm_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(side_embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, rate=1 - (1 - 0.1))

            all_embeddings += mlp_embeddings
        
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) \
                     # + self.u_bias + self.pos_i_bias
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) \
                     # + self.u_bias + self.neg_i_bias

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer / args.batch_size
        # regularizer_b = tf.nn.l2_loss(self.u_bias) + tf.nn.l2_loss(self.pos_i_bias) + tf.nn.l2_loss(self.neg_i_bias)
        # regularizer_b = regularizer_b / args.batch_size

        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # mf_loss = tf.negative(tf.reduce_mean(maxi))

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer \
                   # + self.decay * regularizer_b

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)



if __name__ == '__main__':
    print('lr-->', args.lr)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    adj_type = 'norm_1'

    for decay in [1e-4]:
        print('decay-->', decay)
        for n_layers in [3]:
            print('layer-->', n_layers)
            data_generator.print_statistics()
            config = dict()
            config['n_users'] = data_generator.n_users
            config['n_items'] = data_generator.n_items
            config['decay'] = decay
            config['n_layers'] = n_layers

            norm_1 = data_generator.get_adj_mat()

            config['norm_adj'] = norm_1

            t0 = time()

            model = RGCF(data_config=config)

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
            # sess = tf.Session()

            sess.run(tf.compat.v1.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')


            """
            *********************************************************
            Train.
            """
            loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
            stopping_step = 0
            should_stop = False

            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
                n_batch = data_generator.n_train // args.batch_size + 1

                for idx in range(n_batch):
                    users, pos_items, neg_items = data_generator.sample()
                    _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run(
                        [model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                        feed_dict={model.users: users, model.pos_items: pos_items,
                                   model.neg_items: neg_items})
                    loss += batch_loss
                    mf_loss += batch_mf_loss
                    emb_loss += batch_emb_loss
                    reg_loss += batch_reg_loss


                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                if (epoch + 1) % 10 != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                            epoch, time() - t1, loss, mf_loss, emb_loss)
                        print(perf_str)
                    continue

                t2 = time()
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)

                t3 = time()

                loss_loger.append(loss)
                rec_loger.append(ret['recall'])
                pre_loger.append(ret['precision'])
                ndcg_loger.append(ret['ndcg'])
                hit_loger.append(ret['hit_ratio'])

                if args.verbose > 0:
                    perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                               'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                               (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss,
                                ret['recall'][0],
                                ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                    print(perf_str)
                    fr=open('./RGCF_ConKB.txt','a')
                    fr.write(perf_str)
                    fr.write('\n')
                    fr.close()

                cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                            stopping_step, expected_order='acc',
                                                                            flag_step=5)

#                 if should_stop == True:
#                     break

            recs = np.array(rec_loger)
            pres = np.array(pre_loger)
            ndcgs = np.array(ndcg_loger)
            hit = np.array(hit_loger)

            best_rec_0 = max(recs[:, 0])
            idx = list(recs[:, 0]).index(best_rec_0)

            final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                          '\t'.join(['%.5f' % r for r in pres[idx]]),
                          '\t'.join(['%.5f' % r for r in hit[idx]]),
                          '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
            print(final_perf)

            save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
            ensureDir(save_path)
            f = open(save_path, 'a')

            f.write(
                'embed_size=%d, lr=%.5f, regs=%s, <<adj_type>>=%s\n\t%s\n'
                % (args.embed_size, args.lr, decay, adj_type, final_perf))
            f.close()


# In[ ]:




