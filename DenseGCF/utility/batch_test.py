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


import utility.metrics as metrics
from utility.load_data import *
import multiprocessing
import heapq

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
    auc = metrics.auc(ground_truth=r, prediction=posterior)
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
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

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

