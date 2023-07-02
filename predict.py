import argparse
import os
from random import random
import pandas as pd
import numpy
import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve,roc_curve
import matplotlib.pyplot as plt
import matplotlib
# import tensorflow.keras.layers
# from tensorflow.keras import Model
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical

from Utils.DataUtils import GetDataAndLabelsFromFiles, CreateModelFileNameFromArgs, SaveResults, DatasetOptions, \
    add_bool_arg
from Model import NBeddingModel
from NGramSequenceTransformer import NBeddingTransformer, CharacterLevelTransformer, WeightInitializer

Model = tf.keras.Model
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
Adam = tf.keras.optimizers.Adam
Dropout = tf.keras.layers.Dropout
Dense = tf.keras.layers.Dense
Attention = tf.keras.layers.Attention
EarlyStopping = tf.keras.callbacks.EarlyStopping

##################### TO MAKE MORE DETERMINISTIC EXPERIMENTS #########################
SEED = 43

# Function to initialize seeds for all libraries which might have stochastic behavior

import random


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # tf.random.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

# Activate Tensorflow deterministic behavior
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    # Below code is not supported in Windows
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


# Call the above function with seed value
set_global_determinism(seed=SEED)

######################################################################################


train_file = ''
val_file = ''
# dataset_name = 'URLsCLS'  # grambeddings
out_dir = 'outputs'
CHAR_EMBEDDING_DIM = 95
loss = "sparse_categorical_crossentropy"
loss2 = 'binary_crossentropy'
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
optimizer = "Adam"
LABEL_PHISH = 1
LABEL_LEGIT = 0

"""
    # Example usages: 
    python .\train.py dataset=grambeddings --ngram_1=3 --ngram_2=4 --ngram_3=5 --max_seq_len=128 --attn_width=10 --embed_dim=15 --max_features=160000 --max_df=0.7 --min_df=1e-06 --rnn_cell_size=256
    python .\train.py dataset=ebubekir --ngram_1=4 --ngram_2=5 --ngram_3=6 --max_seq_len=16 --attn_width=5 --embed_dim=20 --max_features=1200 --max_df=0.9 --min_df=1e-06 --rnn_cell_size=256
    # To enable warm_start just specify the related argument:
    python .\train.py warm_start -> warm_start is enabled
    python .\train.py            -> warm_start is disabled

    # To enable case_insensitive just specify the related argument:
    python .\train.py case_insensitive -> case_insensitive is enabled
    python .\train.py                  -> case_insensitive is disabled

"""


def get_args():
    parser = argparse.ArgumentParser(
        """Extracting Top-K Selected NGrams according tp selected scoring method.""")

    parser.add_argument("-d", "--dataset", type=DatasetOptions, default=DatasetOptions.grambeddings,
                        choices=list(DatasetOptions), help="dataset name")
    parser.add_argument("-o", "--output", type=str, default=out_dir,
                        help="The output directory where scores will be stored")

    parser.add_argument("-mn", "--model_name", type=str, default='asdas'
                        , help="Model filename, if it is None then automatically named from given arguments.")

    # Input ngram selections
    parser.add_argument("-n1", "--ngram_1", type=int, default=2, help="Ngram value of first   ngram embedding layer")
    parser.add_argument("-n2", "--ngram_2", type=int, default=2, help="Ngram value of second  ngram embedding layer")
    parser.add_argument("-n3", "--ngram_3", type=int, default=2, help="Ngram value of third   ngram embedding layer")
    # Feature Selection Parameters
    parser.add_argument("-maxf", "--max_features", type=int, default=160000, help="Maximum number of features")
    parser.add_argument("-madf", "--max_df", type=float, default=0.7, help="Embedding dimension for Embedding Layer")
    parser.add_argument("-midf", "--min_df", type=float, default=1e-06, help="Embedding dimension for Embedding Layer")
    parser.add_argument("-msl", "--max_seq_len", type=int, default=128,
                        help="The maximum sequence length to trim our transformed sequences")
    add_bool_arg(parser, 'case_insensitive', False)
    add_bool_arg(parser, 'warm_start', False)
    parser.add_argument("-wm", "--warm_mode", type=WeightInitializer, default=WeightInitializer.randomly_initialize,
                        choices=list(WeightInitializer), help="The selected Embedding Layer weight initializing "
                                                              "method. Only matters when warm_start is set True")

    parser.add_argument("-ed", "--embed_dim", type=int, default=15, help="Embedding dimension for Embedding Layer")
    parser.add_argument("-aw", "--attn_width", type=int, default=10, help="The attention layer width")
    parser.add_argument("-rnn", "--rnn_cell_size", type=int, default=128, help="The recurrent size")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epoch to train our model")

    parser.add_argument("-dp", "--save_deep_features", type=int, default=0,
                        help="Whether save or not logits. 0 False, True Otherwise")
    parser.add_argument("-num_class", "--num_class", type=int, default=13,
                        help="the num of class")
    parser.add_argument("-file_path", "--file_path", type=str,
                        default=r"data/urlData.csv",
                        help="csv file path of the data")
    parser.add_argument("-multi_task", "--multi_task", type=bool,
                        default=False,
                        help="multi_task")
    parser.add_argument("-pre_ft", "--pre_ft", type=bool,
                        default=False,
                        help="pretrain and fine-tune")
    parser.add_argument("-pretrained_model_dir", "--pretrained_model_dir", type=str,
                        default="outputs/training/models/attnasdasall2.h5",
                        help="pretrained_model_dir")
    parser.add_argument("-pred_file_path", "--pred_file_path", type=str,
                        default=r"D:\Fly\URLsClassification\data\test(unlabeled).csv",
                        help="csv file path of the data")
    args = parser.parse_args()
    return args


def Process(args):
    print(args)
    ####################################### Loading Dataset  #######################################
    print('####################################### Loading Dataset  #######################################')
    # train_file = 'data/' + args.dataset.value + '/train.csv'
    # val_file = 'data/' + args.dataset.value + '/test.csv'
    # train_samples, train_labels = GetDataAndLabelsFromFiles(train_file)
    # val_samples, val_labels = GetDataAndLabelsFromFiles(val_file)

    df = pd.read_csv(args.pred_file_path, header=None)
    df.columns = ['url']
    preds_urls = df.url.tolist()

    df2 = pd.read_csv(args.file_path)
    if "urlData" in args.file_path:
        data_name = 'all'
    else:
        data_name = 'access'

    if args.num_class == 2:
        df2['label'] = df2['label'].map(lambda x: 1 if x != 0 else 0)  # 把标签映射为两类，做二分类

    labels = df2.label.values
    urls = df2.url.tolist()


    from collections import Counter
    print(Counter(labels))

    train_samples, test_samples, train_labels, test_labels = \
        train_test_split(urls, labels, test_size=0.2, shuffle=True, random_state=1999)
    train_samples, val_samples, train_labels, val_labels = \
        train_test_split(train_samples, train_labels, test_size=0.1, shuffle=True, random_state=42)

    from train import map_label
    if args.multi_task:
        train_labels2 = map_label(train_labels)
        test_labels2 = map_label(test_labels)
        val_labels2 = map_label(val_labels)

    print('Completed')
    ################################ Character Level Transformation ################################
    print('################################ Character Level Transformation ################################')
    transformer_char = CharacterLevelTransformer(args.max_seq_len, embedding_dim=CHAR_EMBEDDING_DIM,
                                                 case_insensitive=args.case_insensitive)
    char_vocab_size, char_embedding_matrix = transformer_char.Fit()
    # train_sequences_char = transformer_char.Transform(train_samples)
    # val_sequences_char = transformer_char.Transform(val_samples)
    test_sequences_char = transformer_char.Transform(preds_urls)
    print('Completed')

    ############################### First NGram Input Transformation ###############################
    print('############################### First NGram Input Transformation ###############################')
    transformer_1 = NBeddingTransformer(
        ngram_value=args.ngram_1,
        max_num_features=args.max_features,
        max_document_length=args.max_seq_len,
        min_df=args.min_df,
        max_df=args.max_df,
        embedding_dim=args.embed_dim,
        case_insensitive=args.case_insensitive,
        weight_mode=args.warm_mode.value,
    )
    print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_1)
    selected_ngrams_1, selected_ngram_scores_1, weight_matrix_1, vocab_size_1, idf_dict_1 = transformer_1.Fit(
        train_samples, train_labels)
    # print("Starting convert train texts to train sequences for n = ", args.ngram_1)
    # train_sequences_1 = transformer_1.Transform(train_samples)
    # print("Starting convert validation texts to validation sequences for n = ", args.ngram_1)
    # val_sequences_1 = transformer_1.Transform(val_samples)
    print("Starting convert test texts to validation sequences for n = ", args.ngram_1)
    test_sequences_1 = transformer_1.Transform(preds_urls)
    # print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
    #       args.ngram_1)
    # train_sequences_1 = np.array(train_sequences_1, dtype='float32')
    # val_sequences_1 = np.array(val_sequences_1, dtype='float32')
    test_sequences_1 = np.array(test_sequences_1, dtype='float32')
    print('Completed')

    ################################ 2nd NGram Input Transformation ################################
    print('################################ 2nd NGram Input Transformation ################################')
    transformer_2 = NBeddingTransformer(
        ngram_value=args.ngram_2,
        max_num_features=args.max_features,
        max_document_length=args.max_seq_len,
        min_df=args.min_df,
        max_df=args.max_df,
        embedding_dim=args.embed_dim,
        case_insensitive=args.case_insensitive,
        weight_mode=args.warm_mode.value,
    )
    print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_2)
    selected_ngrams_2, selected_ngram_scores_2, weight_matrix_2, vocab_size_2, idf_dict_2 = transformer_2.Fit(
        train_samples, train_labels)
    # print("Starting convert train texts to train sequences for n = ", args.ngram_2)
    # train_sequences_2 = transformer_2.Transform(train_samples)
    # print("Starting convert validation texts to validation sequences for n = ", args.ngram_2)
    # val_sequences_2 = transformer_2.Transform(val_samples)
    print("Starting convert test texts to validation sequences for n = ", args.ngram_2)
    test_sequences_2 = transformer_2.Transform(preds_urls)
    # print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
    #       args.ngram_2)
    # train_sequences_2 = np.array(train_sequences_2, dtype='float32')
    # val_sequences_2 = np.array(val_sequences_2, dtype='float32')
    test_sequences_2 = np.array(test_sequences_2, dtype='float32')
    print('Completed')

    ################################ 3rd NGram Input Transformation ################################
    print('################################ 3rd NGram Input Transformation ################################')
    transformer_3 = NBeddingTransformer(
        ngram_value=args.ngram_3,
        max_num_features=args.max_features,
        max_document_length=args.max_seq_len,
        min_df=args.min_df,
        max_df=args.max_df,
        embedding_dim=args.embed_dim,
        case_insensitive=args.case_insensitive,
        weight_mode=args.warm_mode.value,
    )
    print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_3)
    selected_ngrams_3, selected_ngram_scores_3, weight_matrix_3, vocab_size_3, idf_dict_3 = transformer_3.Fit(
        train_samples, train_labels)
    # print("Starting convert train texts to train sequences for n = ", args.ngram_3)
    # train_sequences_3 = transformer_3.Transform(train_samples)
    # print("Starting convert validation texts to validation sequences for n = ", args.ngram_3)
    # val_sequences_3 = transformer_3.Transform(val_samples)
    print("Starting convert test texts to validation sequences for n = ", args.ngram_3)
    test_sequences_3 = transformer_3.Transform(preds_urls)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_3)
    # train_sequences_3 = np.array(train_sequences_3, dtype='float32')
    # val_sequences_3 = np.array(val_sequences_3, dtype='float32')
    test_sequences_3 = np.array(test_sequences_3, dtype='float32')
    print('Completed')

    # load model
    data_name = 'all'  # access
    if args.model_name is None:
        model_name = CreateModelFileNameFromArgs(opt=args)
    else:
        model_name = args.model_name

    print("load model......")
    if args.pre_ft:
        checkpoint_path = "outputs/training/models/attn" + model_name +"_pre_ft_"+ data_name + str(args.num_class) + ".h5"
    elif args.multi_task:
        checkpoint_path = "outputs/training/models/attn" + model_name +"_multi_task_"+ data_name + str(args.num_class) + ".h5"
    else:
        checkpoint_path = "outputs/training/models/attn" + model_name + data_name + str(args.num_class) + ".h5"

    model = tf.keras.models.load_model(checkpoint_path)
    print("loaded model form {}".format(checkpoint_path))
    model.summary()

    if args.multi_task:
        pred_scores, pred_scores2 = model.predict([test_sequences_char, test_sequences_1, test_sequences_2, test_sequences_3])
    else:
        pred_scores = model.predict([test_sequences_char, test_sequences_1, test_sequences_2, test_sequences_3])

    pred_labels = np.argmax(pred_scores, -1)
    from collections import Counter
    print(Counter(pred_labels))

if __name__ == "__main__":
    opt = get_args()
    Process(opt)
