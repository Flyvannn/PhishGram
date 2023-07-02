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
from pr import pr

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
# loss = "sparse_categorical_crossentropy"
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
    parser.add_argument("-n1", "--ngram_1", type=int, default=3, help="Ngram value of first   ngram embedding layer")
    parser.add_argument("-n2", "--ngram_2", type=int, default=4, help="Ngram value of second  ngram embedding layer")
    parser.add_argument("-n3", "--ngram_3", type=int, default=5, help="Ngram value of third   ngram embedding layer")
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
                        default="outputs/training/models/attnasdasaccess2_new.h5",
                        help="pretrained_model_dir")
    parser.add_argument("-pred_file_path", "--pred_file_path", type=str,
                        default=r"data/test(unlabeled).csv",
                        help="csv file path of the data")
    args = parser.parse_args()
    return args

def map_label(labels):
    new_labels = []
    labels = labels.tolist()
    for l in labels:
        if l == 0:
            new_labels.append(0)
        # elif l == 1:
        #     new_labels.append(1)
        # elif l == 2:
        #     new_labels.append(2)
        else:
            new_labels.append(1)
    return np.array(new_labels, dtype='int64')

def Process(args):
    print(args)
    ####################################### Loading Dataset  #######################################
    print('####################################### Loading Dataset  #######################################')
    # train_file = 'data/' + args.dataset.value + '/train.csv'
    # val_file = 'data/' + args.dataset.value + '/test.csv'
    # train_samples, train_labels = GetDataAndLabelsFromFiles(train_file)
    # val_samples, val_labels = GetDataAndLabelsFromFiles(val_file)

    df = pd.read_csv(args.file_path)
    df['len'] = df['url'].map(len)
    df = df[df['len'] > 5]

    if "urlData" in args.file_path:
        data_name = 'all'
    else:
        data_name = 'access'

    if args.num_class == 2:
        df['label'] = df['label'].map(lambda x: 1 if x!=0 else 0)  # 把标签映射为两类，做二分类

    labels = df.label.values
    urls = df.url.tolist()

    # 将信贷理财和刷单诈骗的标签改为1和2，用多任务
    # if args.multi_task:
    #     # labels = [0 if la == 0 else 1 for la in labels]
    #     for i, l in enumerate(labels):
    #         if l == 10 or l == 11:
    #             labels[i] = l-9
    #         elif l == 1 or l == 2:
    #             labels[i] = l+9

    from collections import Counter
    print(Counter(labels))

    train_samples, test_samples, train_labels, test_labels = \
        train_test_split(urls, labels, test_size=0.2, shuffle=True, random_state=1999)
    train_samples, val_samples, train_labels, val_labels = \
        train_test_split(train_samples, train_labels, test_size=0.1, shuffle=True, random_state=42)

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
    train_sequences_char = transformer_char.Transform(train_samples)
    val_sequences_char = transformer_char.Transform(val_samples)
    test_sequences_char = transformer_char.Transform(test_samples)
    # pred_sequences_char = transformer_char.Transform(pred_urls)
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
    print("Starting convert train texts to train sequences for n = ", args.ngram_1)
    train_sequences_1 = transformer_1.Transform(train_samples)
    print("Starting convert validation texts to validation sequences for n = ", args.ngram_1)
    val_sequences_1 = transformer_1.Transform(val_samples)
    print("Starting convert test texts to validation sequences for n = ", args.ngram_1)
    test_sequences_1 = transformer_1.Transform(test_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_1)
    train_sequences_1 = np.array(train_sequences_1, dtype='float32')
    val_sequences_1 = np.array(val_sequences_1, dtype='float32')
    test_sequences_1 = np.array(test_sequences_1, dtype='float32')

    # pred_sequences_1 = transformer_1.Transform(pred_urls)
    # pred_sequences_1 = np.array(pred_sequences_1, dtype='float32')

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
    print("Starting convert train texts to train sequences for n = ", args.ngram_2)
    train_sequences_2 = transformer_2.Transform(train_samples)
    print("Starting convert validation texts to validation sequences for n = ", args.ngram_2)
    val_sequences_2 = transformer_2.Transform(val_samples)
    print("Starting convert test texts to validation sequences for n = ", args.ngram_2)
    test_sequences_2 = transformer_2.Transform(test_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_2)
    train_sequences_2 = np.array(train_sequences_2, dtype='float32')
    val_sequences_2 = np.array(val_sequences_2, dtype='float32')
    test_sequences_2 = np.array(test_sequences_2, dtype='float32')

    # pred_sequences_2 = transformer_2.Transform(pred_urls)
    # pred_sequences_2 = np.array(pred_sequences_2, dtype='float32')

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
    print("Starting convert train texts to train sequences for n = ", args.ngram_3)
    train_sequences_3 = transformer_3.Transform(train_samples)
    print("Starting convert validation texts to validation sequences for n = ", args.ngram_3)
    val_sequences_3 = transformer_3.Transform(val_samples)
    print("Starting convert test texts to validation sequences for n = ", args.ngram_3)
    test_sequences_3 = transformer_3.Transform(test_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_3)
    train_sequences_3 = np.array(train_sequences_3, dtype='float32')
    val_sequences_3 = np.array(val_sequences_3, dtype='float32')
    test_sequences_3 = np.array(test_sequences_3, dtype='float32')

    # pred_sequences_3 = transformer_3.Transform(pred_urls)
    # pred_sequences_3 = np.array(pred_sequences_3, dtype='float32')

    print('Completed')

    ############################## Initializing the Parallel Networks ##############################
    print('############################## Initializing the Parallel Networks ##############################')
    char_model = NBeddingModel(
        vocab_size=char_vocab_size,
        embedding_dim=CHAR_EMBEDDING_DIM,
        max_seq_length=args.max_seq_len,
        embedding_matrix=char_embedding_matrix,
        rnn_cell_size=args.rnn_cell_size,
        attention_width=args.attn_width,
        warm_start=args.warm_start
    )

    signal_model_1 = NBeddingModel(
        vocab_size=vocab_size_1,
        embedding_dim=args.embed_dim,
        max_seq_length=args.max_seq_len,
        embedding_matrix=weight_matrix_1,
        rnn_cell_size=args.rnn_cell_size,
        attention_width=args.attn_width,
        warm_start=args.warm_start
    )

    signal_model_2 = NBeddingModel(
        vocab_size=vocab_size_2,
        embedding_dim=args.embed_dim,
        max_seq_length=args.max_seq_len,
        embedding_matrix=weight_matrix_2,
        rnn_cell_size=args.rnn_cell_size,
        attention_width=args.attn_width,
        warm_start=args.warm_start
    )

    signal_model_3 = NBeddingModel(
        vocab_size=vocab_size_3,
        embedding_dim=args.embed_dim,
        max_seq_length=args.max_seq_len,
        embedding_matrix=weight_matrix_3,
        rnn_cell_size=args.rnn_cell_size,
        attention_width=args.attn_width,
        warm_start=args.warm_start
    )

    print('Completed')

    ################################ Merging the Parallel Networks #################################
    print('################################ Merging the Parallel Networks #################################')
    last_layer_char, input_layer_char = char_model.CreateModel(embedding_layer_name="embed_char")
    last_layer_1, input_layer_1 = signal_model_1.CreateModel(embedding_layer_name="embed_ngram_1")
    last_layer_2, input_layer_2 = signal_model_2.CreateModel(embedding_layer_name="embed_ngram_2")
    last_layer_3, input_layer_3 = signal_model_3.CreateModel(embedding_layer_name="embed_ngram_3")

    # Merging whole ngram layers
    # attn
    # embedded_concats = tf.keras.layers.Concatenate()(
    #     [last_layer_char, last_layer_1, last_layer_2, last_layer_3])
    # dense1 = Dense(2 * args.rnn_cell_size, activation="relu", name='deep_features')(embedded_concats)

    embedded_concats = tf.stack([last_layer_char, last_layer_1, last_layer_2, last_layer_3], axis=1)
    embedded_concats2 = Dense(embedded_concats.shape[-1], name='attn_vec')(embedded_concats)
    attn = Attention(name="attn")([embedded_concats, embedded_concats2])
    attn = tf.reduce_mean(attn, axis=1)
    dense1 = Dense(2*args.rnn_cell_size, activation="relu", name='deep_features')(attn)
    dropout = Dropout(0.2)(dense1)
    if args.multi_task:
        predictions2 = Dense(2, activation="softmax", name="2cls_out")(dropout)  # 二分类
        dense2 = Dense(args.rnn_cell_size, activation="relu", name='deep_features2')(dropout)
        # dense3 = Dense(args.rnn_cell_size, activation="relu", name='deep_features3')(dense2)
        dropout = Dropout(0.2)(dense2)
    predictions = Dense(args.num_class, activation="softmax", name=f"{args.num_class}cls_out")(dropout)

    # Build and compile model
    model_name = ''
    if args.model_name is None:
        model_name = CreateModelFileNameFromArgs(opt=args)
    else:
        model_name = args.model_name

    log_dir = "outputs/tensorboard/" + model_name
    if args.multi_task:
        model = Model(inputs=[input_layer_char, input_layer_1, input_layer_2, input_layer_3],
                      outputs=[predictions, predictions2])
        model.compile(optimizer=Adam(), loss=loss, loss_weights=[4, 1], metrics=['accuracy'])
    elif args.pre_ft:
        # pretrained_model = tf.keras.models.load_model(args.pretrained_model_dir)
        # outputs = Dense(13, activation="relu", name="dense_2")(pretrained_model.layers[-1].input)
        # model = Model(inputs=pretrained_model.input, outputs=outputs, name='pretrained_model')

        model = Model(inputs=[input_layer_char, input_layer_1, input_layer_2, input_layer_3],
                      outputs=predictions)
        model.load_weights(args.pretrained_model_dir, by_name=True, skip_mismatch=True)
        model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
    else:
        model = Model(inputs=[input_layer_char, input_layer_1, input_layer_2, input_layer_3],
                      outputs=predictions)
        model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
    # metrics=[
    # tf.keras.metrics.TruePositives(name='tp'),
    # tf.keras.metrics.FalsePositives(name='fp'),
    # tf.keras.metrics.TrueNegatives(name='tn'),
    # tf.keras.metrics.FalseNegatives(name='fn'),
    # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    # tf.keras.metrics.Precision(name='precision'),
    # tf.keras.metrics.Recall(name='recall'),
    # tf.keras.metrics.AUC(name='auc'),]
    model.summary()

    # train_labels = train_labels.reshape(-1, 1)
    # val_labels = val_labels.reshape(-1, 1)
    # test_labels = test_labels.reshape(-1, 1)

    train_labels = to_categorical(train_labels, num_classes=args.num_class)
    val_labels = to_categorical(val_labels, num_classes=args.num_class)
    test_labels = to_categorical(test_labels, num_classes=args.num_class)

    if args.multi_task:
        # train_labels2 = train_labels2.reshape(-1, 1)
        # val_labels2 = val_labels2.reshape(-1, 1)
        # test_labels2 = test_labels2.reshape(-1, 1)

        train_labels2 = to_categorical(train_labels2, num_classes=2)
        val_labels2 = to_categorical(val_labels2, num_classes=2)
        test_labels2 = to_categorical(test_labels2, num_classes=2)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print("Tensorbaord is activated url : ", log_dir)

    if args.pre_ft:
        checkpoint_path = "outputs/training/models/attn" + model_name +"_pre_ft_"+ data_name + str(args.num_class) + ".h5"
    elif args.multi_task:
        checkpoint_path = "outputs/training/models/attn" + model_name +"_multi_task_"+ data_name + str(args.num_class) + ".h5"
    else:
        checkpoint_path = "outputs/training/models/attn" + model_name + data_name + str(args.num_class) + "_new.h5"

    import time
    start = time.time()
    if args.multi_task:
        history = model.fit(
            x=[train_sequences_char, train_sequences_1, train_sequences_2, train_sequences_3],
            y=[train_labels, train_labels2],
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            shuffle=True,
            validation_data=(
            [val_sequences_char, val_sequences_1, val_sequences_2, val_sequences_3], [val_labels, val_labels2]),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min', restore_best_weights=True),
                ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True,
                                verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=5, verbose=1, min_delta=1e-4, mode='min'),
                tensorboard_callback
            ]
        )
    else:
        history = model.fit(
            x=[train_sequences_char, train_sequences_1, train_sequences_2, train_sequences_3],
            y=train_labels,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            # shuffle=True,
            validation_data=([val_sequences_char, val_sequences_1, val_sequences_2, val_sequences_3], val_labels),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min', restore_best_weights=True),
                ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True,
                                verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=5, verbose=1, min_delta=1e-4, mode='min'),
                tensorboard_callback
            ]
        )

    # tf.keras.models.save_model(model, "outputs/training/models/best_model")
    print('Training is Completed')
    end = time.time()
    ############################ Extracting Best Epoch and it's scores #############################
    print("############################ Extracting Best Epoch and it's scores #############################")
    if args.multi_task:
        best_epoch_index = history.history['val_13cls_out_accuracy'].index(max(history.history['val_13cls_out_accuracy']))
        best_train_accu = history.history['13cls_out_accuracy'][best_epoch_index]
        best_train_loss = history.history['loss'][best_epoch_index]
        best_valid_accu = history.history['val_13cls_out_accuracy'][best_epoch_index]
        best_valid_loss = history.history['val_loss'][best_epoch_index]
    else:
        best_epoch_index = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
        best_train_accu = history.history['accuracy'][best_epoch_index]
        best_train_loss = history.history['loss'][best_epoch_index]
        best_valid_accu = history.history['val_accuracy'][best_epoch_index]
        best_valid_loss = history.history['val_loss'][best_epoch_index]

    # best_tp = history.history['val_tp'][best_epoch_index]
    # best_fp = history.history['val_fp'][best_epoch_index]
    # best_tn = history.history['val_tn'][best_epoch_index]
    # best_fn = history.history['val_fn'][best_epoch_index]
    # # Calculating the TPR  ==> tpr = tp / (tp+fn)
    # best_tpr = best_tp / (best_tp + best_fn)
    # # Calculating the FPR  ==> fpr = fp / (tn+fp)
    # best_fpr = best_fp / (best_tn + best_fp)
    #
    # best_precision = history.history['val_precision'][best_epoch_index]
    # best_recall = history.history['val_recall'][best_epoch_index]
    # best_auc = history.history['val_auc'][best_epoch_index]

    elapsed_time = end - start

    ######################################## Saving Results ########################################
    print("######################################## Saving Results ########################################")
    print(best_epoch_index + 1, best_train_accu, best_train_loss, best_valid_accu, best_valid_loss, opt, elapsed_time)
    # SaveResults(best_epoch_index + 1, best_train_accu, best_train_loss, best_valid_accu, best_valid_loss,
    #             best_tp, best_fp, best_tn, best_fn, best_tpr, best_fpr, best_precision, best_recall, best_auc,
    #             opt, elapsed_time)

    # embed_weights_1 = GetSpecifiedLayerWeightsByName('embed_ngram_1', model)
    # CreateAndSaveClassDistributionFromEmbeddingMatrix(opt.ngram_1, embed_weights_1, selected_ngrams_1)
    #
    # embed_weights_2 = GetSpecifiedLayerWeightsByName('embed_ngram_2', model)
    # CreateAndSaveClassDistributionFromEmbeddingMatrix(opt.ngram_2, embed_weights_2, selected_ngrams_2)
    #
    # embed_weights_3 = GetSpecifiedLayerWeightsByName('embed_ngram_3', model)
    # CreateAndSaveClassDistributionFromEmbeddingMatrix(opt.ngram_3, embed_weights_3, selected_ngrams_3)

    # deep_model_output =  GetSpecifiedLayerOutputByName('deep_features', model)
    # deep_model_inputs  = model.input
    # deep_model = Model(deep_model_inputs, deep_model_output)
    # deep_features_train = deep_model.predict(x=[train_sequences_char, train_sequences_1, train_sequences_2, train_sequences_3], verbose=1)
    # deep_features_valid = deep_model.predict(x=[val_sequences_char, val_sequences_1, val_sequences_2 , val_sequences_3])

    print("######################################## Test Results ########################################")
    if args.pre_ft:
        res_dir = "outputs/{}/attn_gram_embed_pre_ft_{}cls.txt".format(data_name, args.num_class)
    elif args.multi_task:
        res_dir = "outputs/{}/attn_gram_embed_multi_task_{}cls.txt".format(data_name, args.num_class)
    else:
        res_dir = "outputs/{}/attn_gram_embed_{}cls_new.txt".format(data_name, args.num_class)

    if args.multi_task:
        results = model.evaluate([test_sequences_char, test_sequences_1, test_sequences_2, test_sequences_3], [test_labels, test_labels2])
        print(results)
        pred_scores, pred_scores2 = model.predict([test_sequences_char, test_sequences_1, test_sequences_2, test_sequences_3])
        # pre_scores, pre_scores2 = model.predict([pred_sequences_char, pred_sequences_1, pred_sequences_2, pred_sequences_3])

        # test_labels2 = test_labels2.reshape(-1)
        # test_labels2 = np.squeeze(test_labels2)
        test_labels2 = np.argmax(test_labels2, -1)
        pred_labels2 = np.argmax(pred_scores2, -1)

        from collections import Counter
        print(Counter(test_labels2))

        cr2 = classification_report(test_labels2, pred_labels2, digits=5)
        cm2 = confusion_matrix(test_labels2, pred_labels2)
        print(cr2)
        print(cm2)

    else:
        results = model.evaluate([test_sequences_char, test_sequences_1, test_sequences_2, test_sequences_3],
                                 test_labels)
        print(results)
        pred_scores = model.predict([test_sequences_char, test_sequences_1, test_sequences_2, test_sequences_3])
        # pre_scores = model.predict([pred_sequences_char, pred_sequences_1, pred_sequences_2, pred_sequences_3])

    pr(args.num_class, test_labels, pred_scores)

    # test_labels = test_labels.reshape(-1)
    # test_labels = np.squeeze(test_labels)
    test_labels = np.argmax(test_labels, -1)
    pred_labels = np.argmax(pred_scores, -1)

    # pred_scores = pred_scores[:,1]

    # # roc
    # plt.figure()
    # plt.title('PR Curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.grid()
    # precision, recall, thresholds = precision_recall_curve(pred_labels, pred_scores)
    # plt.plot(recall, precision)
    # plt.show()
    #
    # # pr
    # plt.figure()
    # plt.grid()
    # plt.title('Roc Curve')
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # fpr, tpr, thresholds = roc_curve(pred_labels, pred_scores)
    #
    # from sklearn.metrics import auc
    # auc = auc(fpr, tpr)  # AUC计算
    # plt.plot(fpr, tpr, label='roc_curve(AUC=%0.2f)' % auc)
    # plt.legend()
    # plt.show()

    cr = classification_report(test_labels, pred_labels, digits=5)
    cm = confusion_matrix(test_labels, pred_labels)
    print(cr)
    print(cm)

    with open(res_dir, "w") as f:
        f.write(cr)
        f.write("\nconfusion matrix"+"\n")
        f.write(str(cm)+"\n")
        if args.multi_task:
            f.write(cr2)
            f.write("\nconfusion matrix" + "\n")
            f.write(str(cm2) + "\n")


    # pre_labels = np.argmax(pre_scores, -1)
    # print(pre_labels)

if __name__ == "__main__":
    opt = get_args()
    Process(opt)
