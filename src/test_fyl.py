from os.path import join
import time
import os
import pickle

import numpy as np

from caching import from_cache
from si import ExclusOptimiser
from utils import (load_data, load_data_sample)

# # arguments from terminal
# WORK_FOLDER = sys.argv[1]
# DATA_SET_NAME = sys.argv[2]
# DATA_FILE = sys.argv[3]
# EMBS_FILE = sys.argv[4]

# '''
# immune data
# '''
# WORK_FOLDER = '../data/immune'
# DATA_SET_NAME = 'Human Immune'
# DATA_FILE = 'Human Immune_data.csv'
# EMBS_FILE = 'Human Immune_embs.pkl'

'''
UCI adult data
'''
WORK_FOLDER = '../data/uci adult'
DATA_SET_NAME = 'UCI data'
DATA_FILE = 'adult_clean.csv'
EMBS_FILE = 'uci_adult_emb.pkl'


tic = time.time()
# paths to data and embeddings
path_to_data_file = join(WORK_FOLDER, DATA_FILE)
path_to_embs_file = join(WORK_FOLDER, EMBS_FILE)

# df_data, df_data_scaled = load_data(path_to_data_file)
embs_data = from_cache(path_to_embs_file)
#  load data
df_data, df_data_scaled, lenBinary = load_data(path_to_data_file)

# for each embedding, store si-clustering and si-explanation
for EMB_NAME in embs_data.keys():
    # #  do clustering
    # print(EMB_NAME)
    # embedding = embs_data.get(EMB_NAME)
    # optimiser = ExclusOptimiser(df_data, df_data_scaled, lenBinary, embedding,
    #                             name=DATA_SET_NAME, emb_name=EMB_NAME, work_folder = WORK_FOLDER)
    # optimiser.optimise()
    # toc = time.time()
    # print(f'Overall time: {toc - tic} s')
    # if EMB_NAME == "tSNE_5":
    #     #  load data
    #     df_data, df_data_scaled, lenBinary = load_data(path_to_data_file)
    #     #  do clustering
    #     print(EMB_NAME)
    #     embedding = embs_data.get(EMB_NAME)
    #     optimiser = ExclusOptimiser(df_data, df_data_scaled, lenBinary, embedding,
    #                                 name=DATA_SET_NAME, emb_name=EMB_NAME, work_folder = WORK_FOLDER)
    #     optimiser.optimise()
    #     toc = time.time()
    #     print(f'Overall time: {toc - tic} s')
    if EMB_NAME == "Y":
        # load data
        df_data, df_data_scaled, lenBinary = load_data_sample(path_to_data_file)
        print(EMB_NAME)
        embedding = embs_data.get(EMB_NAME)
        optimiser = ExclusOptimiser(df_data, df_data_scaled, lenBinary, embedding, name=DATA_SET_NAME, emb_name=EMB_NAME,
                                    work_folder=WORK_FOLDER)
        optimiser.optimise()
        toc = time.time()
        print(f'Overall time: {toc - tic} s')












