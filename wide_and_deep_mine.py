# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, concatenate, Embedding, Reshape
from keras.layers import Flatten, merge, Lambda, Dropout, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2, l1_l2


def maybe_download(train_data,test_data):
    """if adult data "train.csv" and "test.csv" are not in your directory,
    download them.
    """

    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

    if not os.path.exists(train_data):
        print("downloading training data...")
        df_train = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data",
            names=COLUMNS, skipinitialspace=True)
    else:
        df_train = pd.read_csv("train.csv",names=COLUMNS, skipinitialspace=True)

    if not os.path.exists(test_data):
        print("downloading testing data...")
        df_test = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test",
            names=COLUMNS, skipinitialspace=True, skiprows=1)
    else:
        df_test = pd.read_csv("test.csv",names=COLUMNS, skipinitialspace=True, skiprows=1)

    return df_train, df_test

def cross_columns(x_cols):
    cross_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        cross_columns[cname] = x_c
    return cross_columns

def wide(df_train, df_test, wide_cols, x_cols, target, model_type, method):

    df_train["IS_TRAIN"] = 1
    df_test["IS_TRAIN"] = 0
    df_wide = pd.concat([df_train, df_test])

    crossed_columns_d = cross_columns(x_cols)
    categorical_columns = list(
        df_wide.select_dtypes(include=['object']).columns
    )

    wide_cols += crossed_columns_d.keys()
    for k, v in crossed_columns_d.items():
        df_wide[k] = df_wide[v].apply(lambda x: "-".join(x), axis = 1)

    df_wide = df_wide[wide_cols + [target] + ["IS_TRAIN"]]

    dummy_cols = [
        c for c in wide_cols if c in categorical_columns or crossed_columns_d.keys()
    ]

    df_wide = pd.get_dummies(df_wide, columns=[x for x in dummy_cols])

    train = df_wide[df_wide.IS_TRAIN == 1].drop("IS_TRAIN", axis=1)
    test  = df_wide[df_wide.IS_TRAIN == 0].drop("IS_TRAIN", axis=1)

    cols = [target] + [c for c in train.columns if c != target]
    train = train[cols]
    test  = test[cols]

    X_train = train.values[:, 1:]
    y_train = train.values[:, 0].reshape(-1, 1)

    X_test  = test.values[:, 1:]
    y_test  = test.values[:, 0].reshape(-1, 1)

    if method == "multiclass":
        y_train = onehot(y_train)
        y_test  = onehot(y_test)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)

    if model_type == "wide":

        activation, loss, metrics = fit_param[method]

        if metrics:
            metrics = [metrics]

        wide_inp = Input(shape=X_train.shape[1], dtype="float32", name="wide_inp")
        w = Dense(y_train.shape[1], activation=activation)(wide_inp)
        wide = Model(wide_inp, w)
        wide.compile(Adam(0.01), loss=loss, metrics=metrics)
        wide.fit(X_train, y_train, nb_epoch=10, batch_size=64)
        result = wide.evaluate(X_test, y_test)

        print("\n", result)

    else:

        return X_train, y_train, X_test, y_test


def deep(df_train, df_test, embedding_cols, cont_cols, target, model_type, method):

    df_train["IS_TRAIN"] = 1
    df_test["IS_TRAIN"]  = 0
    df_deep = pd.concat([df_train, df_test])

    deep_cols = embedding_cols + cont_cols

    df_deep, unique_vals = val2idx(df_deep, embedding_cols)

    train = df_deep[df_deep.IS_TRAIN == 1].drop("IS_TRAIN", axis = 1)
    test  = df_deep[df_deep.IS_TRAIN == 0].drop("IS_TRAIN", axis = 1)

    embedding_tensors = []
    n_factors = 8
    reg = 1e-3
    for ec in embedding_cols:
        layer_name = ec + "_inp"
        t_inp, t_build = embedding_input(
            layer_name, unique_vals[ec], n_factors, reg
        )
        embedding_tensors.append((t_inp, t_build))
        del(t_inp, t_build)

    continuous_tensors = []
    for cc in cont_cols:
        layer_name = cc + "_in"
        t_inp, t_build = continous_input(layer_name)
        continuous_tensors.append((t_inp, t_build))
        del(t_inp, t_build)

    X_train = [train[c] for c in deep_cols]
    y_train = np.array(train[target].values).reshape(-1, 1)
    X_test = [test[c] for c in deep_cols]
    y_test = np.array(test[target].values).reshape(-1,1)

    if method == "multiclass":
        y_train = onehot(y_train)
        y_test  = onehot(y_test)

    inp_layer =  [et[0] for et in embedding_tensors]
    inp_layer += [ct[0] for ct in continuous_tensors]

    inp_embed =  [et[1] for et in embedding_tensors]
    inp_embed += [ct[1] for ct in continuous_tensors]

    if model_type == "deep":
        activation, loss, metrics = fit_param[method]
        if metrics:
            metrics = [metrics]

        d = Concatenate()(inp_embed)
        d = Flatten()(d)
        d = BatchNormalization()(d)
        d = Dense(100, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(d)
        d = Dense(50, activation="relu")(d)
        d = Dense(y_train.shape[1], activation=activation)(d)
        deep = Model(inp_layer, d)
        deep.compile(Adam(0.01), loss=loss, metrics= metrics)
        deep.fit(X_train, y_train, batch_size=64, nb_epoch=10)
        results = deep.evaluate(X_test, y_test)

        print("\n", results)
    else:
        return X_train, y_train, X_test, y_test, inp_embed, inp_layer

def wide_deep(df_train, df_test, wide_cols, x_cols, embedding_cols, cont_cols, method):

    X_train_wide, y_train_wide, X_test_wide, y_test_wide = \
        wide(df_train, df_test, wide_cols, x_cols, target, model_type, method)

    X_train_deep, y_train_deep, X_test_deep, y_test_deep, deep_inp_embed, deep_inp_layer = \
        deep(df_train, df_test, embedding_cols, cont_cols, target, model_type, method)

    X_tr_wd = [X_train_wide] + X_train_deep
    Y_tr_wd = y_train_deep
    X_te_wd = [X_test_wide] + X_test_deep
    Y_te_wd = y_test_deep

    activation, loss, metrics = fit_param[method]
    if metrics:
        metrics = [metrics]

    # WIDE
    w = Input(shape=(X_train_wide.shape[1],), dtype="float32", name="wide")

    # DEEP
    d = Concatenate()(deep_inp_embed)
    d = Flatten()(d)
    d = BatchNormalization()(d)
    d = Dense(100, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(d)
    d = Dense(50, activation="relu", name="deep")(d)

    # WIDE + DEEP
    wd_inp = concatenate([w, d])
    wd_out = Dense(Y_tr_wd.shape[1], activation=activation, name="wide_deep")(wd_inp)
    wide_deep = Model(inputs=[w] + deep_inp_layer, outputs= wd_out)
    wide_deep.summary()
    wide_deep.compile(optimizer=Adam(lr=0.01), loss=loss, metrics=metrics)
    wide_deep.fit(X_tr_wd, Y_tr_wd, nb_epoch=10, batch_size=128)

    results = wide_deep.evaluate(X_te_wd, Y_te_wd)
    print("\n", results)


def continous_input(name):
    inp = Input(shape=(1,), dtype="float32", name=name)
    return inp, Reshape((1,1))(inp)

def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype="int64", name=name)
    return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)

def val2idx(df, cols):

    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    unique_vals = dict()
    for c in cols:
        unique_vals[c] = df[c].nunique()

    return df, unique_vals

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="logistic", help="fitting method")
    ap.add_argument("--model_type", type=str, default="wide_deep", help="wide, deep or both")
    ap.add_argument("--train_data", type=str, default="train.csv")
    ap.add_argument("--test_data", type=str, default="test.csv")

    args = vars(ap.parse_args())
    method = args["method"]
    model_type = args["model_type"]
    train_data = args["train_data"]
    test_data  = args["test_data"]

    fit_param = dict()
    fit_param['logistic'] = ("sigmoid", "binary_crossentropy", "accuracy")
    fit_param['regression'] = (None, "mse", None)
    fit_param['multiclass'] = ("softmax", "categorical_crossentropy", "accuracy")

    df_train, df_test = maybe_download(train_data, test_data)

    df_train['income_label'] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test['income_label'] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    age_groups = [0, 25, 65, 90]
    age_labels = range(len(age_groups) - 1)
    df_train['age_group'] = pd.cut(
        df_train['age'], age_groups, labels=age_labels)
    df_test['age_group'] = pd.cut(
        df_test['age'], age_groups, labels=age_labels)

    # columns for wide model
    wide_cols = ['age', 'hours_per_week', 'education', 'relationship', 'workclass',
                 'occupation', 'native_country', 'gender']
    x_cols = (['education', 'occupation'], ['native_country', 'occupation'])

    # columns for deep model
    embedding_cols = ['education', 'relationship', 'workclass', 'occupation',
                      'native_country']
    cont_cols = ["age", "hours_per_week"]

    # target for logistic
    target = 'income_label'

    if model_type == 'wide':
        wide(df_train, df_test, wide_cols, x_cols, target, model_type, method)
    elif model_type == 'deep':
        deep(df_train, df_test, embedding_cols, cont_cols, target, model_type, method)
    else:
        wide_deep(df_train, df_test, wide_cols, x_cols, embedding_cols, cont_cols, method)


