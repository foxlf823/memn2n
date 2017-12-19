from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from itertools import chain
from six.moves import range, reduce
import numpy as np
import argparse

import torch
from memn2n import *
from helperfunc import *

parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', '-lr', default=0.01, type=float, help='Learning rate for SGD.')
parser.add_argument('--evaluation_interval', '-ei', default=10, type=int, help='Evaluate and print results every x epochs')
parser.add_argument('--batch_size', '-bs', default=32, type=int, help='Batch size for training.')
parser.add_argument('--hops', '-hops', default=3, type=int, help='Number of hops in the Memory Network.')
parser.add_argument('--epochs', '-epochs', default=1000, type=int, help='Number of epochs to train for.')
parser.add_argument('--embedding_size', '-embsz', default=20, type=int, help='Embedding size for embedding matrices.')
parser.add_argument('--memory_size', '-memsz', default=50, type=int, help='Maximum size of memory.')
parser.add_argument('--data_dir', '-data', default='data/tasks_1-20_v1-2/en/', help='Directory containing bAbI tasks')


print("List all parameters...")
args = parser.parse_args()
args_dict = vars(args)
for key,value in args_dict.items():
    print(key+": "+str(value))
print()

# detect whether cuda is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("cuda is used!!!")
else:
    print("cuda is not supported, use cpu")

# We use task 1 as default.
task_id = 1
print("Started Task:", task_id)

# task data
train, test = load_task(args.data_dir, task_id)
data = train + test

# 's' is list of list, 'chain.from_iterable(s)' gets all the words from them
# '+ q + a' makes a big list of all the words in [s, q, a]; 'set' removes all the same words
# 'x | y' denotes union, so 'reduce' iterates each [s, q, a] to union all the words
# 'sorted' rearrange these words to give 'vocab'  
vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# max sentence number in 's' for all [s, q, a]
max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
# max token number in a sentence
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
# max token number in a query
query_size = max(map(len, (q for _, q, _ in data)))
# memory should be able to store all the story sentences
memory_size = min(args.memory_size, max_story_size)

# Add time words/indexes to word_idx, e.g., time1. Here the value is also string, e.g., time1.
for i in range(memory_size):
    word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

vocab_size = len(word_idx) + 1 # +1 for nil word, nil word id is zero
sentence_size = max(query_size, sentence_size) # for the position
sentence_size += 1  # +1 for time words

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)
print("Vocabulary size", vocab_size)

# train/validation/test sets
S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
# split 10% data in the training set as the development set
trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

print(testS[0])

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

batch_size = args.batch_size
# if batch_size is 32, batches will be [(0,32), (32,64), ..., (864, 896)]
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))

model = MemN2N(use_cuda, batch_size, vocab_size, sentence_size, memory_size, args.embedding_size,
                   hops=args.hops, nonlin=None, encoding=position_encoding)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)
criterion = nn.CrossEntropyLoss()

if use_cuda:
    model.cuda()

for t in range(1, args.epochs+1):
    
    np.random.shuffle(batches)
    
    total_cost = 0.0
    train_acc = 0.0
    
    for start, end in batches:
        s = trainS[start:end] # s (batch_size, memory_size, sentence_size)
        q = trainQ[start:end] # q (batch_size, sentence_size)
        a = trainA[start:end] # a (batch_size, vocab_size)
        
        s = prepareInput(use_cuda, s)
        q = prepareInput(use_cuda, q)
        a = prepareInput(use_cuda, a)
        
        a = transferIntoTarget(a)
        
        s = prepareInput1(use_cuda, s, vocab_size)
        q = prepareInput1(use_cuda, q, vocab_size)

        
        optimizer.zero_grad()
        
        cost_t = model.batch_fit(s, q, a, criterion)
        total_cost += cost_t.data[0]
        
        cost_t.backward()
        # gradient clip
        nn.utils.clip_grad_norm(model.parameters(), 40.0)
        # add random noise
        for para in model.parameters():
            gn = torch.zeros(para.grad.data.size())
            gn = torch.normal(gn, std=1e-3)
            if use_cuda:
                gn = gn.cuda()
            para.grad.data += gn
        
        optimizer.step()
        
    if t % args.evaluation_interval == 0:
        train_preds = []
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            
            s = prepareInput(use_cuda, s)
            q = prepareInput(use_cuda, q)
            
            s = prepareInput1(use_cuda, s, vocab_size)
            q = prepareInput1(use_cuda, q, vocab_size)
            
            pred = model.predict(s, q)
            train_preds += list(pred.data.cpu().numpy())
        
        s = prepareInput(use_cuda, valS)
        q = prepareInput(use_cuda, valQ)
        
        s = prepareInput1(use_cuda, s, vocab_size)
        q = prepareInput1(use_cuda, q, vocab_size)

        val_preds = model.predict(s, q)
        # in the official docs, this should be accuracy_score(y_true, y_pred)
        # but reverse parameters don't influence the results
        train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
        val_acc = metrics.accuracy_score(val_preds.data.cpu().numpy(), val_labels)
        
        print('-----------------------')
        print('Epoch', t)
        print('Total Cost:', total_cost)
        print('Training Accuracy:', train_acc)
        print('Validation Accuracy:', val_acc)
        print('-----------------------')
        
    if train_acc > 0.99:
        break

s = prepareInput(use_cuda, testS)
q = prepareInput(use_cuda, testQ)  
s = prepareInput1(use_cuda, s, vocab_size)
q = prepareInput1(use_cuda, q, vocab_size) 
test_preds = model.predict(s, q)
test_acc = metrics.accuracy_score(test_preds.data.cpu().numpy(), test_labels)
print("Testing Accuracy:", test_acc)

