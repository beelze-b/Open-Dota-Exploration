
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
    
numF = int(0.5 * len(validFilePaths))
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
enc = OneHotEncoder(sparse = True)
df_cat_num = enc.fit_transform(df_cat_num)


# In[ ]:

from scipy.sparse import coo_matrix, hstack

df_cat_num = coo_matrix(df_cat_num)
df_cat = coo_matrix(df_cat)
df = hstack([df_cat_num, df_numerical])


# In[ ]:

# df = pandas.concat([df_numerical, df_cat, df_cat_num], ignore_index=True)


# In[ ]:

np.random.seed(1)
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
weights_4 = tf.Variable(tf.random_normal([layer_size[2], NumFeatures]), name='weights_4')
bias_4 = tf.Variable(tf.random_normal([NumFeatures]), name='bias_4')
    
layer1 = tf.nn.relu(tf.matmul(x, weights_1, a_is_sparse=True) + bias_1)
layer2 = tf.nn.relu(tf.matmul(layer1, weights_2, a_is_sparse=True, b_is_sparse=True) + bias_2)
layer3 = tf.nn.relu(tf.matmul(layer2, weights_3, a_is_sparse=True, b_is_sparse=True) + bias_3)
output = tf.nn.relu(tf.matmul(layer3, weights_4, a_is_sparse=True, b_is_sparse=True) + bias_4)
    
cost = tf.reduce_mean(tf.pow(y-output, 2))
momentum = 0.5
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
    
variable_dict = {'weights_1': weights_1, 'weights_2': weights_2, 'weights_3': weights_3, 'weights_4': weights_4,
                     'bias_1': bias_1, 'bias_2': bias_2, 'bias_3': bias_3, 'bias_4': bias_4}
saver = tf.train.Saver(variable_dict)
init = tf.global_variables_initializer()

ckpoint_dir = os.path.join(os.getcwd(), 'model-backups/model.ckpt')


# In[ ]:

def train():
    numEpochs = 1000
    numBatches = 100
    batchSize = int(round(0.001 * df_train.shape[0]))
    flatten = lambda l: [item for sublist in l for item in sublist]
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
    if (trainFlag):
        sess.run(init)
        train()
    else:
        saver.restore(sess, ckpoint_dir)


# In[ ]:

print 'Done'
print datetime.datetime.now()

