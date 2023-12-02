import collections
import os
import numpy as np
import logging
import scipy.sparse as sp

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    logging.info("================== preparing data ===================")
    n_entity, n_relation, kg = load_kg(args)
    train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set,n_users,n_items,ui_adj,ui_adj_orign = load_rating(args)
    logging.info("contructing users' kg triple sets ...")
    user_triple_sets = kg_propagation(args, kg, user_init_entity_set, args.user_triple_set_size, True)
    logging.info("contructing items' kg triple sets ...")
    item_triple_sets = kg_propagation(args, kg, item_init_entity_set, args.item_triple_set_size, False)
    ui_adjdency, u_adjdency = construct_ui_adjdency(args)
    return train_data, eval_data, test_data, n_entity, n_relation, user_triple_sets, item_triple_sets,n_users,n_items,ui_adj,ui_adjdency,u_adjdency,ui_adj_orign


def load_rating(args):
    rating_file = '../data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    return dataset_split(rating_np)


def dataset_split(rating_np):
    logging.info("splitting dataset to 6:2:2 ...")
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    
    user_init_entity_set, item_init_entity_set, user_to_item_dict = collaboration_propagation(rating_np, train_indices)
    
    train_indices = [i for i in train_indices if rating_np[i][0] in user_init_entity_set.keys()]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_init_entity_set.keys()]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_init_entity_set.keys()]
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]
    global n_users
    n_users = max(max(train_data[:, 0]), max(eval_data[:, 0]))+1
    n_items = max(max(train_data[:, 1]), max(eval_data[:, 1]))+1
    ui_adj_orign,ui_adj = generate_ui_adj(rating_np, train_data)

    return train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set,n_users,n_items,ui_adj,ui_adj_orign
    
def generate_ui_adj(rating, train_rating):
    n_user, n_item = len(set(rating[:, 0])), len(set(rating[:, 1]))
    ui_adj_orign = sp.coo_matrix(
        (train_rating[:, 2], (train_rating[:, 0], train_rating[:, 1])), shape=(n_user, n_item)).todok()
    ui_adj_orign_1 = sp.bmat([[sp.csr_matrix((ui_adj_orign.shape[0], ui_adj_orign.shape[0])), ui_adj_orign],
                                [ui_adj_orign.T, sp.csr_matrix((ui_adj_orign.shape[1], ui_adj_orign.shape[1]))]])
    ui_adj = sp.bmat([[None, ui_adj_orign],
                      [ui_adj_orign.T, None]], dtype=np.float32)
    ui_adj = ui_adj.todok()
    ui_adj = ui_adj.tocsr()[:n_users, :].tocoo()
    return ui_adj_orign_1,ui_adj



def collaboration_propagation(rating_np, train_indices):
    logging.info("contructing users' initial entity set ...")
    user_history_item_dict = dict()
    item_history_user_dict = dict()
    item_neighbor_item_dict = dict()
    user_to_item_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_item_dict:
                user_history_item_dict[user] = []
            user_history_item_dict[user].append(item)
            if item not in item_history_user_dict:
                item_history_user_dict[item] = []
            item_history_user_dict[item].append(user)
        
    logging.info("contructing items' initial entity set ...")
    for item in item_history_user_dict.keys():
        item_nerghbor_item = []
        for user in item_history_user_dict[item]:
            item_nerghbor_item = np.concatenate((item_nerghbor_item, user_history_item_dict[user]))
        item_neighbor_item_dict[item] = list(set(item_nerghbor_item))

    item_list = set(rating_np[:, 1])
    for item in item_list:
        if item not in item_neighbor_item_dict:
            item_neighbor_item_dict[item] = [item]
    return user_history_item_dict, item_neighbor_item_dict, user_to_item_dict


def load_kg(args):
    kg_file = '../data/' + args.dataset + '/kg_final'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg


def construct_kg(kg_np):
    logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def kg_propagation(args, kg, init_entity_set, set_size, is_user):

    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        if is_user and args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h,r,t = [],[],[]
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]

            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
                    
            if len(h) == 0:
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace= (len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets


def kg_propagation_user(args, kg, init_entity_set, set_size, is_user):
    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        if is_user and args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h, r, t = [], [], []
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]

            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])

            if len(h) == 0:
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace=(len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets

def construct_ui_adjdency(args):
    print('construct_ui_adjdency begin...')
    adjui_file = '../data/' + args.dataset + '/ui_adj'
    if os.path.exists(adjui_file + '.npy'):
        ui_adjdency = np.load(adjui_file + '.npy')
    else:
        ui_adjdency = np.loadtxt(adjui_file + '.txt', dtype=np.float32)
        np.save(adjui_file + '.npy', ui_adjdency)
    print(ui_adjdency)
    adju_file = '../data/' + args.dataset + '/u_adjdency'
    if os.path.exists(adju_file + '.npy'):
        u_adjdency = np.load(adju_file + '.npy')
    else:
        u_adjdency = np.loadtxt(adju_file + '.txt', dtype=np.float32)
        np.save(adju_file + '.npy', u_adjdency)
    print(u_adjdency)
    return ui_adjdency, u_adjdency