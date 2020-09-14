#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main table alignment procedure of TableNet 
"""

from model import *
from dataloader import DataLoader

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split




# %%
# we store the pre-loaded embeddings so that we do not load that every time we do changes in the class.
tables = dict()
entity_cats = dict()
cat_tax = dict()
cat_level = dict()

emb_w2v = None
emb_n2v = None

verse_emb = None


# %%  Load the data and train the models
# You can download the data from http://l3s.de/~fetahu/wiki_tables/data/
# base_dir = '/home/fetahu/wiki_tables/data/'
base_dir = '/mnt/zr/downstream/wiki_tables/data/'

print("Start building data loader...")
# create the class DataLoader
loader = DataLoader()

loader.cat_taxonomy_path = base_dir + 'category_data/flat_cat_taxonomy.tsv.gz'
# loader.table_data_path = base_dir + 'table_data/html_data/structured_html_table_data_ground_truth.json.gz'
loader.table_data_path = base_dir + 'table_data/html_data/structured_html_table_data.json.gz'
loader.word2vec_path = base_dir + 'embeddings/glove.6B.300d.emb.gz'
loader.node2vec_path = base_dir + 'embeddings/category_entity_label_node2vec.emb.gz'
loader.entity_category_path = base_dir + 'category_data/article_cats_201708.tsv.gz'
loader.table_data_labels = base_dir + 'gt_data/table_pair_evaluation_eq_sub_irrel_labels.tsv'
loader.out_dir = base_dir
print("finished initializing the data loader~")

# load first the embeddings
print("start to load the embeddings...")
word2vec = None
node2vec = None
if word2vec is None and node2vec is None:
    loader.load_embeddings()
    word2vec = loader.word2vec
    node2vec = loader.node2vec
else:
    loader.word2vec = word2vec
    loader.node2vec = node2vec
print('Loaded word embeddings with %d entries and node2vec embeddings with %d entries' % (len(loader.word2vec.vocab), len(loader.node2vec.vocab)))

# construct the vocabularies
loader.construct_vocab()
print('finished construct_vocab: Loaded the word and entity vocabularies')
print("Length of entity_cats: ", len(entity_cats))

# entity_cats = None
print("start loading eitity categories...")
if entity_cats is None or len(entity_cats) == 0:
    loader.load_entity_cats()
    entity_cats = loader.entity_cats
else:
    loader.entity_cats = entity_cats
print('Loaded the entity categories for %d entities' % (len(loader.entity_cats)))

print("start loading category taxonomy...")
if cat_tax is None or len(cat_tax) == 0:
    loader.load_flat_cat_tax()
    cat_tax = loader.cat_tax
    cat_level = loader.cat_level
else:
    loader.cat_tax = cat_tax
    loader.cat_level = cat_level
print('Loaded the category taxonomy with %d entries' % (len(loader.cat_tax)))

print("start loading data alignment pairs...")
tables = None
if tables is not None:
    loader.tables = tables
    loader.load_alignment_pairs()
else:
    loader.load_alignment_pairs()
    tables = loader.table
print('Loaded the table data with %d entries' % (len(loader.tables)))

print("start construct the evaluation data...")
loader.construct_eval_data()
loader.load_evaluation_data()
print('Constructed the evaluation data for all tables')

# create the matrices for the embeddings
print("start creating the word2vec embedding matrix...")
emb_w2v = loader.get_word2vec_matrix(256)
print("start creating the node2vec embedding matrix...")
emb_n2v = loader.get_node2vec_matrix(256)
print("finish creating both matrices~")


# %%
X = []
X_subj = []
X_val = []
Y = []

for inst in loader.eval_pairs:
    X += inst['col_name']
    X_subj += inst['col_subject']
    X_val += inst['col_values']
    Y += [inst['label']]

encoder = LabelBinarizer()
Y_fit = encoder.fit_transform(Y)
print(encoder.classes_)

LEN = max(set([len(x) for x in X]) | set([len(x) for x in X_val]) | set([len(x) for x in X_subj]))

X_val = pad_sequences(X, maxlen=LEN,value=loader.vocab_w2v['UNK'], padding='pre')
X_val_subj = pad_sequences(X_subj, maxlen=LEN,value=loader.vocab_n2v['UNK'], padding='pre')
X_val_val = pad_sequences(X_val, maxlen=LEN,value=loader.vocab_n2v['UNK'], padding='pre')

X_train, X_test, y_train, y_test = train_test_split(X_val, Y_fit, test_size=0.3, random_state=1)
X_train_subj, X_test_subj, y_train_subj, y_test_subj = train_test_split(X_val_subj, Y_fit, test_size=0.3, random_state=1)
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_val_val, Y_fit, test_size=0.3, random_state=1)
