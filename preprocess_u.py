import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

RATING_FILE_NAME = dict({'movie': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie': ',', 'book': ';', 'music': '\t'})

def convert_rating():
    file = '../data/' + DATASET + '/ratings_final.txt'
    print(file)
    data = []
    for line in open(file, encoding='utf-8').readlines()[1:]:
        user = line.strip().split('\t')[0]
        item = line.strip().split('\t')[1]
        score = line.strip().split('\t')[2]
        data.append([int(user), int(item), int(score)])
    print('data done')
    writer = open('../data/' + DATASET + '/ratings.txt', 'w', encoding='utf-8')
    for da in data:
        writer.write('%d\t%d\t%.2f\n' % (da[0], da[1], da[2]))
    writer.close()

def create_ua1adj():
    data = []
    users = []
    items = []

    file = '../data/'+DATASET+'/ratings_final.txt'
    print(file)
    for line in open(file,encoding='utf-8').readlines():
        user = line.strip().split('\t')[0]
        item = line.strip().split('\t')[1]
        score = line.strip().split('\t')[2]
        data.append([int(user),int(item),int(score)])
        users.append(user)
        items.append(item)

    n_user = max(users)+1
    n_item = max(items)+1
    ua1_adjdency = np.zeros(n_user,n_item)
    for i in data:
        user = i[0]
        item = i[1]
        ua1_adjdency[user][item] = i[3]
    writer = open('../data/' + DATASET + '/ua1_adjdency.txt', 'w', encoding='utf-8')
    print(ua1_adjdency.shape[0], ua1_adjdency.shape[1])
    for row in range(ua1_adjdency.shape[0]):
        for col in range(ua1_adjdency.shape[1]):
            writer.write('%.2f\t' % (ua1_adjdency[row][col]))
        writer.write('\n')
    writer.close()
    return ua1_adjdency,n_user,n_item

def create_ua2adj():
    data = []
    users = []
    items = []

    file = '../data/'+DATASET+'/ratings_final.txt'
    print(file)
    for line in open(file,encoding='utf-8').readlines():
        user = line.strip().split('\t')[0]
        item = line.strip().split('\t')[1]
        score = line.strip().split('\t')[2]
        data.append([int(user),int(item),int(score)])
        users.append(user)
        items.append(item)
    n_user = max(users)+1
    n_item = max(items)+1
    ua2_adjdency = np.zeros(n_user,n_item)
    for i in data:
        user = i[0]
        item = i[1]
        ua2_adjdency[user][item] = i[4]
    writer = open('../data/' + DATASET + '/ua2_adjdency.txt', 'w', encoding='utf-8')
    print(ua2_adjdency.shape[0], ua2_adjdency.shape[1])
    for row in range(ua2_adjdency.shape[0]):
        for col in range(ua2_adjdency.shape[1]):
            writer.write('%.2f\t' % (ua2_adjdency[row][col]))
        writer.write('\n')
    writer.close()
    return ua2_adjdency,n_user,n_item

def create_ua3adj():
    data = []
    users = []
    items = []

    file = '../data/'+DATASET+'/ratings_final.txt'
    print(file)
    for line in open(file,encoding='utf-8').readlines():
        user = line.strip().split('\t')[0]
        item = line.strip().split('\t')[1]
        score = line.strip().split('\t')[2]
        data.append([int(user),int(item),int(score)])
        users.append(user)
        items.append(item)

    n_user = max(users)+1
    n_item = max(items)+1
    ua3_adjdency = np.zeros(n_user,n_item)
    for i in data:
        user = i[0]
        item = i[1]
        ua3_adjdency[user][item] = i[5]
    writer = open('../data/' + DATASET + '/ua1_adjdency.txt', 'w', encoding='utf-8')
    print(ua3_adjdency.shape[0], ua3_adjdency.shape[1])
    for row in range(ua3_adjdency.shape[0]):
        for col in range(ua3_adjdency.shape[1]):
            writer.write('%.2f\t' % (ua3_adjdency[row][col]))
        writer.write('\n')
    writer.close()
    return ua3_adjdency,n_user,n_item


def create_uiadj():
    data = []
    users = []
    items = []
    file = '../data/' + DATASET + '/ratings_final.txt'
    print(file)
    for line in open(file, encoding='utf-8').readlines():
        user = line.strip().split('\t')[0]
        item = line.strip().split('\t')[1]
        score = line.strip().split('\t')[2]
        data.append([int(user), int(item), float(score)])
        users.append(int(user))
        items.append(int(item))
    n_user = max(users) + 1
    n_item = max(items) + 1
    ui_adj = np.zeros((n_user, n_user))
    for user in users:
        ui_adj[user][user] = 1
    print(ui_adj)
    writer = open('../data/' + DATASET + '/ui_adj.txt', 'w', encoding='utf-8')
    print(ui_adj.shape[0], ui_adj.shape[1])
    for row in range(ui_adj.shape[0]):
        for col in range(ui_adj.shape[1]):
            writer.write('%.2f\t' % (ui_adj[row][col]))
        writer.write('\n')
    writer.close()
    return ui_adj

def create_ui_adjdency():
    data = []
    users = []
    items = []
    file = '../data/' + DATASET + '/ratings_final.txt'
    print(file)
    for line in open(file, encoding='utf-8').readlines():
        user = line.strip().split('\t')[0]
        item = line.strip().split('\t')[1]
        score = line.strip().split('\t')[2]
        data.append([int(user), int(item), float(score)])
        users.append(int(user))
        items.append(int(item))
    n_user = max(users) + 1
    n_item = max(items) + 1
    print(n_user, n_item)
    ui_adjdency = np.zeros((n_user, n_item))
    for i in data:
        user = int(i[0])
        item = int(i[1])
        ui_adjdency[user][item] = float(i[2])
    print(ui_adjdency)
    writer = open('../data/' + DATASET + '/ui_adjdency.txt', 'w', encoding='utf-8')
    print(ui_adjdency.shape[0], ui_adjdency.shape[1])
    for row in range(ui_adjdency.shape[0]):
        for col in range(ui_adjdency.shape[1]):
            writer.write('%.2f\t' % (ui_adjdency[row][col]))
        writer.write('\n')
    writer.close()
    return ui_adjdency, n_user, n_item

def create_u_adjdency():
    ui_adjdency, n_user, n_item = create_ui_adjdency()
    print(n_user, n_item)
    u_adjdency = np.zeros((n_user, n_user))
    u_ser = dict()
    print(type(ui_adjdency))
    ui_adjdency = np.int64(ui_adjdency > 0).T
    print(ui_adjdency)
    for it in range(0, n_item):
        key = it
        u_ser[key] = list()
        for us in range(0, n_user):
            if(ui_adjdency[it][us] == 1):
                u_ser[key].append(us)
    for us in u_ser.values():
        for p in range(0, len(us)):
            for q in range(0, len(us)):
                if(us[p] == us[q]):
                    u_adjdency[us[p]][us[q]] = 1
                else:
                    u_adjdency[us[p]][us[q]] += 1
    print(np.max(u_adjdency))
    u_adjdency = cosine_similarity(u_adjdency)
    row, col = np.diag_indices_from(u_adjdency)
    u_adjdency[row, col] = 1
    u_adjdency = np.around(u_adjdency, decimals = 4)
    writer = open('../data/' + DATASET + '/u_adjdency.txt', 'w', encoding='utf-8')
    print(ui_adjdency.shape[0], ui_adjdency.shape[1])
    print(u_adjdency)
    for row in range(u_adjdency.shape[0]):
        for col in range(u_adjdency.shape[1]):
            writer.write('%.4f\t' % (u_adjdency[row][col]))
        writer.write('\n')
    writer.close()






if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='music', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d
    convert_rating()
    create_u_adjdency()
    create_uiadj()
    print('done')