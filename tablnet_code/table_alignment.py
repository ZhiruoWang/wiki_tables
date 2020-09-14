#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main table alignment procedure of TableNet 
"""

from model import build_bilstm_col_model, build_bilstm_col_subject_model, build_bilstm_col_subject_val_model, \
    build_bilstm_baseline_w2v_lca_val, build_bilstm_baseline_w2v_lca, build_bilstm_baseline_w2v, \
    build_lstm_baseline_w2v_lca, build_lstm_baseline_w2v, build_lstm_baseline_w2v_lca_val
from dataloader import DataLoader

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm


# %%
'''Compute P/R/F1 from the confusion matrix. '''
def evaluation_metrics_report(mat, labels_, method_, epochs=10):
    num_classes = len(mat)
    scores = dict()
    avg_p = []
    avg_r = []
    avg_f1 = []
    for i in range(0, num_classes):
        p = mat[i,i] / float(sum(mat[:,i]))
        r = mat[i,i] / float(sum(mat[i,:]))
        f1 = 2 * (p * r) / (p + r)
        scores[i] = (p, r, f1)
        avg_p.append(p)
        avg_r.append(r)
        avg_f1.append(f1)
    outstr = 'Evaluation results for ' + method_ + ' Epochs: ' + str(epochs) + '\n'
    for key in scores:
        label = labels_[key]
        val_1 = scores[key][0]
        val_2 = scores[key][1]
        val_3 = scores[key][2]
        outstr += ('%s\tP=%.3f\tR=%.3f\tF1=%.3f\n' % (label, val_1, val_2, val_3))
    avg_p_score = sum(avg_p) / len(avg_p)
    avg_r_score = sum(avg_r) / len(avg_r)
    avg_f1_score = sum(avg_f1) / len(avg_f1)
    outstr += 'AVG\tAvg-P=%.3f\tAvg-R=%.3f\tAvg-F1=%.3f\n' % (avg_p_score, avg_r_score, avg_f1_score)
    return outstr



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


# %% TableNet Model -- Column Title Word Representation
tablenet_desc = build_bilstm_col_model(len(emb_w2v), 256, emb_w2v, 50, LEN)
tablenet_desc.fit(x=X_train, y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_tablenet_val = tablenet_desc.predict([X_test])
mat_tablenet_val = cm(y_test.argmax(axis=1), y_pred_tablenet_val.argmax(axis=1))
print(evaluation_metrics_report(mat_tablenet_val, encoder.classes_, 'TableNet - Column', 30))


# %% TableNet Model - Column LCA representation
tablenet_desc_lca = build_bilstm_col_subject_model(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
tablenet_desc_lca.fit(x=[X_train, X_train_subj], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_tablenet_desc_lca = tablenet_desc_lca.predict([X_test, X_test_subj])
mat_tablenet_desc_lca = cm(y_test.argmax(axis=1), y_pred_tablenet_desc_lca.argmax(axis=1))
print(evaluation_metrics_report(mat_tablenet_desc_lca, encoder.classes_, 'TableNet - Column+LCA', 30))


# %% TableNet Model - Column title, LCA and Value representation
tablenet_desc_val_lca = build_bilstm_col_subject_val_model(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
tablenet_desc_val_lca.fit(x=[X_train, X_train_subj, X_train_val], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_tablenet_col_lca_val = tablenet_desc_val_lca.predict([X_test, X_test_subj, X_test_val])
m_tablenet_col_lca_val = cm(y_test.argmax(axis=1), y_pred_tablenet_col_lca_val.argmax(axis=1))
print(evaluation_metrics_report(m_tablenet_col_lca_val, encoder.classes_, 'TableNet - Column+VAL+LCA', 50))


# %% TableNet Model - Column title, and value representation
tablenet_desc_val = build_bilstm_col_subject_model(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
tablenet_desc_val.fit(x=[X_train, X_train_val], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_tablenet_col_val = tablenet_desc_val.predict([X_test, X_test_val])
m4 = cm(y_test.argmax(axis=1), y_pred_tablenet_col_val.argmax(axis=1))
print(evaluation_metrics_report(m4, encoder.classes_, 'TableNet - Column+VAL', 30))


# %% BiLSTM baseline - column title, LCA, value representation
bilstm_baseline_lca = build_bilstm_baseline_w2v_lca_val(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
bilstm_baseline_lca.fit(x=[X_train, X_train_subj, X_train_val], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_blst_lca_val = bilstm_baseline_lca.predict([X_test, X_test_subj, X_test_val])
blstm_lca_scores = cm(y_test.argmax(axis=1), y_pred_blst_lca_val.argmax(axis=1))
print(evaluation_metrics_report(blstm_lca_scores, encoder.classes_, 'BiLSTM - Column+VAL+LCA', 10))


# %% BiLSTM baseline - column title, VAL representation
bilstm_baseline_w2v_val = build_bilstm_baseline_w2v_lca(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
bilstm_baseline_w2v_val.fit(x=[X_train, X_train_val], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_b_w2v_val = bilstm_baseline_w2v_val.predict([X_test, X_test_subj])
b_w2v_val_mat = cm(y_test.argmax(axis=1), y_pred_b_w2v_val.argmax(axis=1))
print(evaluation_metrics_report(b_w2v_val_mat, encoder.classes_, 'BiLSTM - Column+VAL', 50))


# %% BiLSTM baseline - column title, LCA representation
bilstm_baseline_w2v_lca = build_bilstm_baseline_w2v_lca(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
bilstm_baseline_w2v_lca.fit(x=[X_train, X_train_subj], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_b_w2v_lca = bilstm_baseline_w2v_lca.predict([X_test, X_test_subj])
b_w2v_lca_mat = cm(y_test.argmax(axis=1), y_pred_b_w2v_lca.argmax(axis=1))
print(evaluation_metrics_report(b_w2v_lca_mat, encoder.classes_, 'BiLSTM - Column+LCA', 10))


# %% BiLSTM baseline - column title representation
bilstm_baseline_w2v = build_bilstm_baseline_w2v(len(emb_w2v), 256, emb_w2v, 50, LEN)
bilstm_baseline_w2v.fit(x=X_train, y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_b_w2v = bilstm_baseline_w2v.predict(X_test)
b_w2v_mat = cm(y_test.argmax(axis=1), y_pred_b_w2v.argmax(axis=1))
print(evaluation_metrics_report(b_w2v_mat, encoder.classes_, 'BiLSTM - Column', 50))


# %% LSTM baseline - column title, and VAL representation
model_lstm_w2v_val = build_lstm_baseline_w2v_lca(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
model_lstm_w2v_val.fit(x=[X_train, X_train_val], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_lstm_w2v_val = model_lstm_w2v_val.predict([X_test, X_test_subj])
lstm_w2v_val_mat = cm(y_test.argmax(axis=1), y_pred_lstm_w2v_val.argmax(axis=1))
print(evaluation_metrics_report(lstm_w2v_val_mat, encoder.classes_, 'LSTM - Column + VAL', 30))


# %% LSTM baseline - column title, and LCA representation
model_lstm_w2v = build_lstm_baseline_w2v_lca(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
model_lstm_w2v.fit(x=[X_train, X_train_subj], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_lstm_w2v_lca = model_lstm_w2v.predict([X_test, X_test_subj])
lstm_w2v_lca_mat = cm(y_test.argmax(axis=1), y_pred_lstm_w2v_lca.argmax(axis=1))
print(evaluation_metrics_report(lstm_w2v_lca_mat, encoder.classes_, 'LSTM - Column + LCA', 10))


# %% LSTM baseline - column title representation
model_lstm_w2v_col = build_lstm_baseline_w2v(len(emb_w2v), 256, emb_w2v, 50, LEN)
model_lstm_w2v_col.fit(x=X_train, y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_lstm_w2v_col = model_lstm_w2v_col.predict(X_test)
lstm_w2v_col_mat = cm(y_test.argmax(axis=1), y_pred_lstm_w2v_col.argmax(axis=1))
print(evaluation_metrics_report(lstm_w2v_col_mat, encoder.classes_, 'LSTM - Column', 50))


# %% LSTM baseline - column title, LCA, value representation
model_lstm_w2v_lca_val = build_lstm_baseline_w2v_lca_val(len(emb_w2v), 256, emb_w2v, len(emb_n2v), 256, emb_n2v, 50, LEN)
model_lstm_w2v_lca_val.fit(x=[X_train, X_train_subj, X_train_val], y=y_train, batch_size=100, epochs=50, validation_split=0.1)
y_pred_lstm_w2v_col_lca_val = model_lstm_w2v_lca_val.predict([X_test, X_test_subj, X_test_val])
lstm_w2v_col_lca_val_mat = cm(y_test.argmax(axis=1), y_pred_lstm_w2v_col_lca_val.argmax(axis=1))
print(evaluation_metrics_report(lstm_w2v_col_lca_val_mat, encoder.classes_, 'LSTM - Column+VAL+LCA', 10))
