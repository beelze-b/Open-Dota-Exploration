
# coding: utf-8

# In[ ]:

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import glob
import datetime
import itertools


# In[ ]:

np.random.seed(1)


# In[ ]:

import os
import os.path
import gc


# In[ ]:

import argparse
parser = argparse.ArgumentParser(description = "Please insert the train flag")


# In[ ]:

parser.add_argument('-t', '--train', action = "store",
                    help='If true, we train and save. Else, otherwise.', required = True)


# In[ ]:

my_args = vars(parser.parse_args())
trainFlag = my_args['train']
trainFlag = trainFlag.lower() in ("True", "t", "true", "1", 1)


# In[ ]:

print datetime.datetime.now()
validFilePaths = []
for f in os.listdir("data/anomaly_data"):
    filePath = os.path.join("data/anomaly_data", f)
    if os.path.isdir(filePath):
        continue
    if os.stat(filePath).st_size <= 3:
        continue
    validFilePaths.append(filePath)
    
numF = int(0.1 * len(validFilePaths))
print 'Using this many files {0}'.format(numF)
validFilePaths = np.random.choice(validFilePaths, numF, replace=False)
df_list = (pandas.read_csv(f) for f in validFilePaths)
df = pandas.concat(df_list, ignore_index=True)
df = df[df['radiant_win'].notnull()]


# In[ ]:

print df.shape
columns = df.columns
df_catInteger_features_example = filter(lambda x: 'hero_id' in x, columns)


# In[ ]:

from itertools import chain
# these will require string processing on the column names to work
numericalFeatures = ['match_id', 'positive_votes', 'negative_votes', 'first_blood_time', 'radiant_win',
                    'duration', 'kills', 'deaths', 'assists', 'apm', 'kpm', 'kda', 'hero_dmg',
                    'gpm', 'hero_heal', 'xpm', 'totalgold', 'totalxp', 'lasthits', 'denies',
                    'tower_kills', 'courier_kills', 'gold_spent', 'observer_uses', 'sentry_uses',
                    'ancient_kills', 'neutral_kills', 'camps_stacked', 'pings', 'rune_pickups']
categoricalIntegerFeatures = ['barracks_status', 'tower_status', 'hero_id'] 
                              #'item0', 'item1', 'item2', 'item3', 'item4', 'item5']
categoricalFullFeatures = ['patch']
numFeatures = [filter(lambda x: z in x, columns) for z in numericalFeatures]
categoricalIntegerFeatures  = [filter(lambda x: z in x, columns) for z in categoricalIntegerFeatures]
catFull = [filter(lambda x: z in x, columns) for z in categoricalFullFeatures]
numFeatures = list(chain(*numFeatures))
categoricalIntegerFeatures = list(chain(*categoricalIntegerFeatures))
catFull = list(chain(*catFull))


# In[ ]:

df_numerical = df[numFeatures]
df_numerical.loc[:, 'radiant_win'] = df_numerical.loc[:, 'radiant_win'].apply(lambda x : int(x))
df_cat_num = df[categoricalIntegerFeatures]
df_cat = df[catFull]

#scipy sparse
vectorizer = DictVectorizer(sparse = True)
df_cat = vectorizer.fit_transform(df_cat.fillna('NA').to_dict(orient="records"))

#scipy sparse
# need to make sure that the categorical columns see all fields during training
enc = OneHotEncoder(sparse = True)
fitMatrix = dict.fromkeys(map(lambda x: unicode(x), categoricalIntegerFeatures))

towerColumns = [filter(lambda x: z in x, columns) for z in ['tower_status']]
towerAllCategories = list(set(reduce(lambda x, y: x+y, [df_cat_num[i].values.tolist() for i in towerColumns[0]])))


heroColumns = [filter(lambda x: z in x, columns) for z in ['hero_id']]
heroesAllCategories = list(set(range(1, 115)                         + reduce(lambda x, y: x+y, [df_cat_num[i].values.tolist() for i in heroColumns[0]])))
heroesAllCategories = list(itertools.chain.from_iterable(itertools.repeat(heroesAllCategories,                             1+len(towerAllCategories)/len(heroesAllCategories))))[:len(towerAllCategories)]

barrackColumns = [filter(lambda x: z in x, columns) for z in ['barracks_status']]
barrackAllCategories = list(set(reduce(lambda x, y: x+y, [df_cat_num[i].values.tolist() for i in barrackColumns[0]])))
barrackAllCategories = list(itertools.chain.from_iterable(itertools.repeat(barrackAllCategories,                             1+len(towerAllCategories)/len(barrackAllCategories))))[:len(towerAllCategories)]




for column in heroColumns[0]:
    fitMatrix[column] = heroesAllCategories
for column in towerColumns[0]:
    fitMatrix[column] = towerAllCategories
for column in barrackColumns[0]:
    fitMatrix[column] = barrackAllCategories

fitMatrix = pandas.DataFrame.from_dict(fitMatrix)
# order of columns matters
fitMatrix = fitMatrix[df_cat_num.columns.values.tolist()]
enc.fit(fitMatrix)
df_cat_num = enc.transform(df_cat_num)


# In[ ]:

from scipy.sparse import coo_matrix, hstack

df_cat_num = coo_matrix(df_cat_num)
df_cat = coo_matrix(df_cat)
df = hstack([df_cat_num, df_numerical])


# In[ ]:

# df = pandas.concat([df_numerical, df_cat, df_cat_num], ignore_index=True)


# In[ ]:

x = np.random.rand(df.shape[0])
mask = np.where(x < 0.7)[0]
mask1 = np.where(np.logical_and(x >= 0.7, x < 0.9))[0] 
mask2 = np.where(x >= 0.9)[0]


# In[ ]:

df_train = df.tocsr()[mask, :]
df_validation = df.tocsr()[mask1, :]
df_test = df.tocsr()[mask2, :]


# In[ ]:

NumFeatures = df.shape[1]
layer_size = [10, 10, NumFeatures]
learning_rate = 0.1


# In[ ]:

print NumFeatures


# In[ ]:

print df_train.shape


# In[ ]:

x = tf.placeholder(tf.float32, [None, NumFeatures])
y = x
#encoders
weights_1 = tf.Variable(tf.random_normal([NumFeatures, layer_size[0]]), name='weights_1')
bias_1 = tf.Variable(tf.random_normal([layer_size[0]]), name='bias_1')
weights_2 = tf.Variable(tf.random_normal([layer_size[0], layer_size[1]]), name='weights_2')
bias_2 = tf.Variable(tf.random_normal([layer_size[1]]), name='bias_2')
    
#decoders
weights_3 = tf.Variable(tf.random_normal([layer_size[1], layer_size[2]]), name='weights_3')
bias_3 = tf.Variable(tf.random_normal([layer_size[2]]), name='bias_3')
  
layer1 = tf.nn.relu(tf.matmul(x, weights_1, a_is_sparse=True) + bias_1)
layer2 = tf.nn.relu(tf.matmul(layer1, weights_2, a_is_sparse=True, b_is_sparse=True) + bias_2)
output = tf.nn.relu(tf.matmul(layer2, weights_3, a_is_sparse=True, b_is_sparse=True) + bias_3)
    
cost = tf.reduce_mean(tf.reduce_sum(tf.pow(y-output, 2), 1))
rank = tf.rank(cost)

momentum = 0.5
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
    
variable_dict = {'weights_1': weights_1, 'weights_2': weights_2, 'weights_3': weights_3,
                     'bias_1': bias_1, 'bias_2': bias_2, 'bias_3': bias_3}
saver = tf.train.Saver(variable_dict)
init = tf.global_variables_initializer()

ckpoint_dir = os.path.join(os.getcwd(), 'model-backups/model.ckpt')


# In[ ]:

flatten = lambda l: [item for sublist in l for item in sublist]

def test(test_data):
    batch = test_data.tolil()
    ind = [[[i, batch.rows[i][j]] for j in range(len(batch.rows[i]))] for i in range(batch.shape[0])]
    ind = flatten(ind)
    dat = np.nan_to_num(flatten(batch.data))
    batch = tf.sparse_to_dense(ind, [batch.shape[0], batch.shape[1]], dat)
    data = batch.eval()
    data = data.astype(np.float32)
    layer1 = tf.nn.relu(tf.matmul(data, weights_1, a_is_sparse=True) + bias_1)
    layer2 = tf.nn.relu(tf.matmul(layer1, weights_2, a_is_sparse=True, b_is_sparse=True) + bias_2)
    output = tf.nn.relu(tf.matmul(layer2, weights_3, a_is_sparse=True, b_is_sparse=True) + bias_3)
    residuals = tf.reduce_sum(tf.abs(output - batch), axis = 1)
    residuals = residuals.eval()
    indices = np.argsort(residuals)[::-1]
    return data[indices[0:10], :], output.eval()[indices[0:10], :], indices


# In[ ]:

def train():
    numEpochs = 1000
    numBatches = 100
    batchSize = int(round(0.001 * df_train.shape[0]))
    for epochIter in xrange(numEpochs):
        print 'Epoch: {0}'.format(epochIter)
        gc.collect()
        if epochIter % 100 == 0:
            saver.save(sess, ckpoint_dir)
        for batchItr in xrange(numBatches):
            indices = np.random.choice(range(df_train.shape[0]), batchSize, replace=False)
            batch = df_train[indices, :].tolil()
            ind = [[[i, batch.rows[i][j]] for j in range(len(batch.rows[i]))] for i in range(batch.shape[0])]
            ind = flatten(ind)
            dat = np.nan_to_num(flatten(batch.data))
            batch = tf.sparse_to_dense(ind, [batch.shape[0], batch.shape[1]], dat)
            batch = batch.eval()
            sess.run(optimizer, feed_dict = {x : batch})

with tf.Session() as sess:
    if sess.run(rank) != 0:
        raise Exception("Wrong dimenions of cost")
    if (trainFlag):
        sess.run(init)
        train()
    else:
        print 'Doing test'
        saver.restore(sess, ckpoint_dir)
        anomalies, output, indices_highest_anomaly = test(df_test)
        np.savetxt("data/anomalies.csv", anomalies, delimiter=",")
        np.savetxt("data/output.csv", output, delimiter=",")
        np.savetxt('data/indices.csv', indices_highest_anomaly, delimiter = ',')


# In[ ]:

print 'Done'
print datetime.datetime.now()

