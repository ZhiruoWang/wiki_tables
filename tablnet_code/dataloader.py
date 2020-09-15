#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reproduce the table alignment modules of TableNet 
"""

# %%
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import numpy as np
import gzip
import json
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
import random
from table import WikiTable
import re, os, sys


# %%
class DataLoader(object):
    def __init__(self):
        # the stop words, which we will use to skip stop words from the column titles
        self.stops = set(stopwords.words('english'))     # nltk.download('stopwords')
        self.word2vec_path = ''
        self.node2vec_path = ''
        self.cat_taxonomy_path = ''
        self.entity_category_path = ''
        self.table_data_path = ''
        self.table_data_labels = ''
        self.out_dir = ''

        # data structures that which will hold the necessary data for training the models
        self.entity_cats = dict()
        self.word2vec = None
        self.node2vec = None
        self.vocab_n2v = dict()
        self.vocab_w2v = dict()
        self.DELIM = 0
        self.table_pairs = []
        self.tables = dict()
        self.table_rep = dict()
        self.eval_pairs = []
        self.cat_tax = dict()
        self.cat_parents = dict()
        self.cat_level = dict()


    '''Load the word2vec and the node2vec embeddings. '''
    def load_embeddings(self):
        self.word2vec = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=False)
        self.node2vec = KeyedVectors.load_word2vec_format(self.node2vec_path, binary=False)

    def construct_vocab(self):
        self.vocab_w2v = {key: idx + 2 for idx, key in enumerate(self.word2vec.vocab)}

        # keep zero for the unknown words
        self.vocab_w2v['UNK'] = 0
        self.vocab_w2v['COL_TOKEN_SPLIT'] = 1

        self.vocab_n2v = {key: idx + 2 for idx, key in enumerate(self.node2vec.vocab)}
        self.vocab_n2v['UNK'] = 0
        self.vocab_n2v['COL_TOKEN_SPLIT'] = 1

        # the delimiter which we use to stitch tables together
        self.DELIM_N2V = [len(self.vocab_n2v)]
        self.DELIM_W2V = [len(self.vocab_w2v)]


    '''Load the table pairs for alignment with their table data values and their corresponding label. '''
    def load_alignment_pairs(self):
        # load the table data only for these entities and the table alignment pairs
        self.table_pairs = []
        with open(self.table_data_labels, "r", encoding='utf-8') as fr:
            eval_data = fr.readlines()
        for line in eval_data:
            data = line.strip().split('\t')
            tbl_id_a, tbl_id_b, label = data[-3], data[-2], data[-1]
            self.table_pairs.append((tbl_id_a, tbl_id_b, label))
        # load the table data
        self.load_tables()


    '''Load the table data for a set of entities of interest. '''
    def load_tables(self):
        # read the table data and take only the tables for the entities in the entity index.
        fin = gzip.open(self.table_data_path, 'rt')

        # return the tables as a dict with the table id as an index.
        self.tables = dict()

        for line in fin:
            if len(line.strip()) == 0:
                continue
            tbl_json = json.loads(line)
            entity = tbl_json['entity']       # string, e.g. 'Metropolis_Gold'
            sections = tbl_json['sections']   # list
            for section in sections:
                tables_json = section['tables']
                for table in tables_json:
                    tbl = WikiTable()
                    tbl.load_json(json.dumps(table), entity, section, int(table['id']), col_meta_parse=True)
                    tbl.markup = table['markup']
                    self.tables[tbl.table_id] = tbl
                    

    '''Load the evaluation dataset. '''
    def load_evaluation_data(self):
        for table_pair in self.table_pairs:
            table_pair_dict = dict()
            table_pair_dict['col_name'] = []
            table_pair_dict['col_values'] = []
            table_pair_dict['col_subject'] = []

            print("load-eval-data: ", table_pair)
            table_a = self.table_rep[table_pair[0]]
            table_b = self.table_rep[table_pair[1]]

            # generate the different representations
            label = table_pair[2]

            # concatenate the features for the different tables
            ep_col_name = np.concatenate((np.concatenate(table_a['col_name'].values()), self.DELIM_W2V, np.concatenate(table_b['col_name'].values())), axis=0)
            ep_col_values = np.concatenate((np.concatenate(table_a['col_values'].values()), self.DELIM_N2V, np.concatenate(table_b['col_values'].values())), axis=0)
            ep_col_subject = np.concatenate((np.concatenate(table_a['col_subject'].values()), self.DELIM_N2V, np.concatenate(table_b['col_subject'].values())), axis=0)

            # ep_col_name = np.concatenate((np.concatenate(table_a['col_name'].values(), axis=0), self.DELIM_W2V, np.concatenate(table_b['col_name'].values(), axis=0)), axis=0)
            # ep_col_values = np.concatenate((np.concatenate(table_a['col_values'].values(), axis=0), self.DELIM_N2V, np.concatenate(table_b['col_values'].values(), axis=0)), axis=0)
            # ep_col_subject = np.concatenate((np.concatenate(table_a['col_subject'].values(), axis=0), self.DELIM_N2V, np.concatenate(table_b['col_subject'].values(), axis=0)), axis=0)

            table_pair_dict['col_name'].append(ep_col_name)
            table_pair_dict['col_values'].append(ep_col_values)
            table_pair_dict['col_subject'].append(ep_col_subject)

            table_pair_dict['table_a'] = table_pair[0]
            table_pair_dict['table_b'] = table_pair[1]
            table_pair_dict['label'] = label
            self.eval_pairs.append(table_pair_dict)



    '''
    Constructs the evaluation data which we will use to train our DL model. 
    In this case we will represent the data in the following ways:
        1)  The simplest form of the data representation is in terms of the column names.
        2)  We augment the representation with the entities or values present in a column, 
            in case the values do not link to entities or are not textual, 
            then we will represent the column data with a UNK vector.
        3)  Finally, in the case where (2) reflects entities, we will additionally represent
            the data with the corresponding column label (i.e., the LCA category of entities)
    '''
    def construct_eval_data(self):
        self.table_rep.clear()
        # generate the appropriate table representation that we can use for the deep learning models.
        for table_id in self.tables:
            table = self.tables[table_id]
            sub_table_rep = dict()

            # generate the column representation, for the columns for which we do not have a word we assign UNK
            col_names_rep = dict()
            col_val_rep = dict()
            col_subj_rep = dict()

            for col_idx, column in enumerate(table.column_meta_data):
                col_subj_rep[col_idx] = []
                col_val_rep[col_idx] = []
                col_names_rep[col_idx] = []

                # generate the word embedding for the column name
                col_names_rep[col_idx] = self.column_title_tb_idx(column, self.vocab_w2v)

                # generate the column representation based on the entities
                col_values = table.column_meta_data[column]

                # take the subset of values which exist in our entity-category index
                sub_vals = []
                for val in col_values:
                    val_label = re.sub(' ', '_', val)
                    if val_label in self.vocab_n2v:
                        sub_vals.append(val_label)
                for val in sub_vals:
                    col_val_rep[col_idx].append(self.vocab_n2v[val])
                col_val_rep[col_idx].append(self.vocab_n2v['COL_TOKEN_SPLIT'])

                # get the lca category for the values in this column (in case they represent entities)
                if len(col_values) != 0:
                    lca_cats = self.find_lca_category(col_values)
                    
                    if lca_cats is not None:
                        for cat in lca_cats: 
                            cat = 'Category:' + re.sub(' ', '_', cat)
                            if cat not in self.vocab_n2v:
                                continue
                            col_subj_rep[col_idx].append(self.vocab_n2v[cat])
                    else:
                        col_subj_rep[col_idx].append(0)

                # distinguish between the column representations
                col_subj_rep[col_idx].append(self.vocab_n2v['COL_TOKEN_SPLIT'])
                col_names_rep[col_idx].append(self.vocab_w2v['COL_TOKEN_SPLIT'])
                col_val_rep[col_idx].append(self.vocab_n2v['COL_TOKEN_SPLIT'])
            sub_table_rep['col_name'] = col_names_rep
            sub_table_rep['col_values'] = col_val_rep
            sub_table_rep['col_subject'] = col_subj_rep
            self.table_rep[table_id] = sub_table_rep


    '''
    For a given set of seed entities return their common LCA category. This is in a way representing the subject or class of the given entities.
    '''
    def find_lca_category(self, entities):
        for entity in entities:
            if entity not in self.entity_cats:
                continue
            if entity not in self.cat_parents:
                self.cat_parents[entity] = []
            #load all the categories and the parent categories for an entity
            for cat in self.entity_cats[entity]:
                self.load_cat_parents(cat, self.cat_parents[entity])

        # find the common categories
        entity = entities[0]
        common_cats = set()
        index = 0
        for entity in entities:
            if entity not in self.cat_parents:
                continue
            if index == 0:
                common_cats = set(self.cat_parents[entity])
                index += 1
            else:
                common_cats.intersection(self.cat_parents[entity])

        # get the lowest matching category
        if len(common_cats) != 0:
            common_cat_level = [self.cat_level[cat] for cat in common_cats if cat in self.cat_level]
            if len(common_cat_level) != 0:
                max_level = max(common_cat_level)
                return [cat for cat in common_cats if self.cat_level[cat] == max_level]

        return None


    '''Since a column name might have different words, we split and aggregate the word vectors from the resulting words. '''
    def column_title_tb_idx(self, column, vocab):
        col_words = column.lower().split()
        col_rep = []

        if len(col_words) == 1:
            if col_words[0] in vocab and col_words[0] not in self.stops:
                col_rep.append(vocab[col_words[0]])
            else:
                col_rep.append(vocab['UNK'])
        else:
            for col in col_words:
                if col in vocab and col not in self.stops:
                    col_rep.append(vocab[col])
                else:
                    col_rep.append(vocab['UNK'])
        col_rep.append(vocab['COL_TOKEN_SPLIT'])
        return col_rep


    '''Load the parents of a category up to the root. '''
    def load_cat_parents(self, cat, parents):
        if cat in self.cat_tax and cat not in parents:
            sub_parents = self.cat_tax[cat]
            parents.append(cat)
            for parent in sub_parents:
                self.load_cat_parents(parent, parents)


    '''Load the category taxonomy where each node has contains its parents. '''
    def load_flat_cat_tax(self):
        for line in gzip.open(self.cat_taxonomy_path, "r"):
            data = line.decode('utf-8').strip().split('\t')
        # for line in gzip.open(self.cat_taxonomy_path, 'rt'):
        #     data = line.strip().split('\t')

            parent_cat = data[0]
            child_cat = data[2]

            if parent_cat not in self.cat_level:
                self.cat_level[parent_cat] = int(data[1])
            if child_cat not in self.cat_level:
                self.cat_level[child_cat] = int(data[3])

            if child_cat not in self.cat_tax:
                self.cat_tax[child_cat] = []
            self.cat_tax[child_cat].append(parent_cat)


    '''Loads the entity categories. '''
    def load_entity_cats(self):
        for line in gzip.open(self.entity_category_path, 'r'):
            data = line.decode('utf-8').strip().split('\t')
        # for line in gzip.open(self.entity_category_path, 'rt'):
        #     data = line.strip().split('\t')
            # print("Data: ", data)
            if len(data) != 2:
                continue
            if data[0] not in self.entity_cats:
                self.entity_cats[data[0]] = []
            self.entity_cats[data[0]].append(data[1])


    '''The word2vec embedding matrix. '''
    def get_word2vec_matrix(self, emb_dim):
        # This will be the embedding matrix
        embeddings = 1 * np.random.randn(len(self.vocab_w2v) + 2, emb_dim)
        embeddings[0] = 0  # So that the padding will be ignored

        # Build the embedding matrix
        for word, index in self.vocab_w2v.items():
            if word in self.word2vec.vocab:
                embeddings[index] = self.word2vec.word_vec(word)[:emb_dim]

        # del self.word2vec
        return embeddings


    '''The node2vec embedding matrix. '''
    def get_node2vec_matrix(self, emb_dim):
        # This will be the embedding matrix
        embeddings = 1 * np.random.randn(len(self.vocab_n2v) + 2, emb_dim)
        embeddings[0] = 0  # So that the padding will be ignored

        # Build the embedding matrix
        for word, index in self.vocab_n2v.items():
            if word in self.node2vec.vocab:
                embeddings[index] = self.node2vec.word_vec(word)[:emb_dim]
        # del self.node2vec
        return embeddings
