import pandas as pd
import numpy as np
import sys
import math
import argparse
import csv
import json
from tqdm.autonotebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import multiprocessing

device = torch.device("cuda:2") 

user_item_train={}
counter=0
with open('train_eventid_filtered.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        user_item_train[row[0]]=row[1:]

user_item_test={}
counter=0
with open('test_eventid_filtered.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        user_item_test[row[0]]=row[1:]

with open('eventid2token.json','r') as f:
    eventid2token=json.load(f)
    
user_token_train = {}
for user_id, item_ids in user_item_train.items():
    user_token_train[user_id] = list(set([ eventid2token[item_id] for item_id in item_ids if item_id in eventid2token]))
    
user_token_test = {}
for user_id, item_ids in user_item_test.items():
    user_token_test[user_id] = list(set([ eventid2token[item_id] for item_id in item_ids if item_id in eventid2token]))

with open('../../../tmp2/kp/data/token2eventid.json','r') as f:
    token2eventid=json.load(f)

# obtain positive training samples
train_user_ids = [ [user_id]* len(tokens)  for user_id, tokens in user_token_train.items()]
train_user_ids = np.hstack(train_user_ids)
test_user_ids = [ [user_id]* len(tokens)  for user_id, tokens in user_token_test.items()]
test_user_ids = np.hstack(test_user_ids)
train_positive_ids = np.hstack(user_token_train.values())
test_item_ids = np.hstack(user_token_test.values())

user2idx = {user:idx for idx, user in enumerate(list(set(train_user_ids.tolist() + test_user_ids.tolist())))}
item2idx = {item:idx for idx, item in enumerate(list(set(train_positive_ids.tolist() + test_item_ids.tolist())))}  
idx2user={v: k for k, v in user2idx.items()}
idx2item={v: k for k, v in item2idx.items()}

#load pre-trained word embed
with open('../../../tmp2/kp/data/feature/train/eventid_to_tfidf_3000.json','r') as f:
    tfidf_train=json.load(f)
with open('../../../tmp2/kp/data/feature/test/eventid_to_tfidf_3000.json','r') as f:
    tfidf_test=json.load(f)
    
#load pre-trained word embed
weights_matrix = np.zeros((num_items, 3000))
tfidf = {**tfidf_train, **tfidf_test}
for idx, item_token in idx2item.items():
        weights_matrix[idx] = tfidf[str(token2eventid[item_token])]

train_user_ids = np.array([user2idx[user_id]  for user_id in  train_user_ids])
test_user_ids = np.array([user2idx[user_id]  for user_id in  test_user_ids])
train_positive_ids = np.array([item2idx[item_id]  for item_id in  train_positive_ids])
test_item_ids = np.array([item2idx[item_id]  for item_id in  test_item_ids])

num_items = max(item2idx.values()) + 1
num_users = max(user2idx.values()) + 1

train_user_ids = np.tile(train_user_ids, 2)
train_positive_ids.shape, train_user_ids.shape
train_labels = np.array(([1] * (len(train_user_ids)//2)) + ([0] * (len(train_user_ids)//2)))

all_item_ids = set(np.hstack(user_token_train.values()))
negative_sample_size = 1
negative_item_ids={}

def neg_sample(user_id):
    return np.random.choice(list(all_item_ids - set(user_token_train[user_id])), size=negative_sample_size * len(user_token_train[user_id]))

# build NCF model
class NCF(nn.Module):
    def __init__(self, nb_users, nb_items,
                 mf_dim=64, mlp_layer_sizes=[256, 256, 128, 64], dropout=0):
        
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(NCF, self).__init__()
        nb_mlp_layers = len(mlp_layer_sizes)

        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2)
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)
        self.dropout = dropout

        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])  # noqa: E501

        self.final = nn.Linear(mlp_layer_sizes[-1] + mf_dim, 1)

        self.mf_user_embed.weight.data.normal_(0., 0.01)
        self.mf_item_embed.weight.data.normal_(0., 0.01)
        self.mlp_user_embed.weight.data.normal_(0., 0.01)
        self.mlp_item_embed.weight.data.normal_(0., 0.01)

        def glorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)
        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            glorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, user, item, sigmoid=False):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)
            if self.dropout != 0:
                xmlp = nn.functional.dropout(xmlp, p=self.dropout, training=self.training)

        x = torch.cat((xmf, xmlp), dim=1)
        x = self.final(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return 


# start running NCF model
model=NCF(num_users,num_items)
model=model.to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss()
loss_fn = loss_fn.to(deviceepochs=10
file = open('log.txt', 'w')
for _ in range(epochs):
    model.train()
    
    
    #neg sample
    pool = multiprocessing.Pool(30)
    negative_item_ids=pool.map(neg_sample, list(user_token_train.keys()))
    negative_item_ids={k:v for k,v in zip(list(user_token_train.keys()),negative_item_ids)}
    negative_item_ids_h = np.hstack(negative_item_ids.values())
    negative_item_ids_h = np.array([item2idx[item_id]  for item_id in  negative_item_ids_h])
    train_item_ids= np.concatenate([train_positive_ids, negative_item_ids_h])
    train_labels = np.array(([1] * (len(train_user_ids)//2)) + ([0] * (len(train_user_ids)//2)))
    indices=np.arange(len(train_user_ids))
    np.random.shuffle(indices)
    
    
    real_train_item_ids,real_train_user_ids,real_train_labels=train_item_ids[indices],train_user_ids[indices],train_labels[indices]
    batch_size=2**13
    print(len(real_train_item_ids),len(real_train_user_ids))
    pbar=tqdm(range(len(real_train_user_ids)//batch_size))
    for batch in pbar:
        optimizer.zero_grad()
        users,items=torch.Tensor([real_train_user_ids[batch*batch_size:(batch+1)*batch_size]]).long().to(device), torch.Tensor([real_train_item_ids[batch*batch_size:(batch+1)*batch_size]]).long().to(device)
        outputs=model(users.reshape(-1,),items.reshape(-1,))
        label=torch.Tensor([real_train_labels[batch*batch_size:(batch+1)*batch_size]]).to(device)
        loss = loss_fn(outputs.t(), label).mean()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss":loss.item()})
        
#evaluation
model.eval()
ranked_item=[]
hits=[]
for user_id in tqdm(range(num_users)):
    optimizer.zero_rad()
    users,items=torch.Tensor([user_id]).expand(num_items).long().to(device), torch.Tensor(np.arange(num_items)).long().to(device)
    outputs=model(users,items, sigmoid=True)
    indices = torch.argsort(outputs.squeeze(-1),descending=True)
    ranked_item.append(indices.cpu().numpy())

pbar=tqdm(enumerate(zip(test_user_ids, test_item_ids)),total=len(test_user_ids))

for i,(user_id, target_item_id) in pbar:
    hits.append(ranked_item[user_id]==target_item_id)

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

mrr=mean_reciprocal_rank(hits)
print(mrr)