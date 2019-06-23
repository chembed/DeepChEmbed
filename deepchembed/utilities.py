import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def clean_out_of_bound(bio_deg):
    """
    clean the bio degradation part, if negative, treat as 0
    if above 100, treat as 100
    ----
    Args:
    bio_deg: pd.series or list or numpy.ndarray
    ----
    Return:
    cleaned pd.series
    """
    cleaned_bio = []
    for i in bio_deg:
        if i < 0:
            i = 0
        elif i > 100:
            i = 100
        cleaned_bio.append(i)
    return pd.Series(cleaned_bio)


def bi_class(raw, boundary):
    """
    divide raw input into two classes, based on selected boundary
    """
    bi_class = pd.Series([0 if i < boundary else 1 for i in raw])
    return bi_class



def check_dtypes_count(df):
    """
    Quickly check for unique data types in a dataframe and return
    counts for each type.
    ----
    Args:
        pd.dataframe
    ----
    Return
        dtypes and counts: tuple of two arrays
    """
    return np.unique(df.dtypes, return_counts=True)

def dedup_input_cols(df):
    """
    return columns that has distinct input
    """
    return df.loc[:,df.nunique()>1]

def assign_class(num, cuts):
    """
    num: int/float target to be assigned to classes
    cuts: list(of float/int) or np.ndarray to be used as cut-edges between
        classes, no start/end value
    """
    assert len(cuts) >0; "cuts can not be empty"

    for i in range(len(cuts)):
        if num <=cuts[i]:
            return i

    return i+1


def divide_classes(lst, cuts):
    """
    lst: pd.dataframe(int/float)
    cuts: list(of float/int) or np.ndarray to be used as cut-edges between
        classes, no start/end value
    """
    cls = [assign_class(num,cuts) for num in lst]
    return pd.DataFrame(cls)

def tsne_2d_visulization(input_feat, plot_labels, ax=None, labels=None,
                         figsaveto=None, verbose=1, perplexity=40,
                         n_iter=500, alpha=1):
    """
    """
    df = pd.DataFrame()

    if len(plot_labels) == len(input_feat):
        plot_labels = [plot_labels]
    else:
        assert len(plot_labels[0]) == len(input_feat)

    nplots = len(plot_labels)
    nbins = []

    for i in range(nplots):
        df['Labels_' + str(i)] = plot_labels[i]
        if labels is not None:
            for j in pd.value_counts(df['Labels_' + str(i)]).keys():
                df['Labels_' + str(i)] = df['Labels_' + str(i)].replace(j,labels[j])
        nbins.append(len(pd.value_counts(df['Labels_' + str(i)])))

    tsne = TSNE(n_components=2, verbose=verbose,
                perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(input_feat)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    if ax is None:
        fig, ax = plt.subplots(nplots,1,figsize=(8*nplots,5))
        if not (isinstance(ax,list) or isinstance(ax, np.ndarray)): ax = [ax]
    else:
        if not (isinstance(ax,list) or isinstance(ax, np.ndarray)): ax = [ax]
        assert len(ax) == nplots

    for i in range(nplots):
        sns.scatterplot(x='tsne-2d-one', y='tsne-2d-two',
                        hue='Labels_' + str(i),
                        palette=sns.color_palette("hls", nbins[i]),
                        data=df,legend="full",alpha=alpha, ax=ax[i],)

    if figsaveto is not None:
        plt.savefig(figsaveto, bbox_inches='tight')

    return

def tsne_2d_visulization_test_and_train(
        train_feat, train_labels, test_feat, test_labels,
        labels=None,ax=None,figsaveto=None, verbose=1, perplexity=40,
        n_iter=500, alpha=1):
    """
    """
    assert len(train_feat[0]) == len(test_feat[0])

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    input_feat = np.concatenate((train_feat, test_feat), axis=0)
    n_train = len(train_feat)
    df_train['Training Labels'] = train_labels
    df_test['Testing Labels'] = test_labels
    assert len(pd.value_counts(df_train['Training Labels'])) ==\
        len(pd.value_counts(df_test['Testing Labels']))
    nbins = len(pd.value_counts(df_train['Training Labels']))

    if labels is not None:
        assert isinstance(labels,list) or isinstance(labels,dict)
        for i in pd.value_counts(train_labels).keys():
            df_train['Training Labels'] = df_train['Training Labels'].replace(i,labels[i])
            df_test['Testing Labels'] = df_test['Testing Labels'].replace(i,labels[i])

    tsne = TSNE(n_components=2, verbose=verbose,
                perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(input_feat)

    df_train['tsne-2d-one'] = tsne_results[:n_train,0]
    df_train['tsne-2d-two'] = tsne_results[:n_train,1]
    df_test['tsne-2d-one']  = tsne_results[n_train:,0]
    df_test['tsne-2d-two']  = tsne_results[n_train:,1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    else:
        assert isinstance(ax, plt.Axes)

    sns.scatterplot(x='tsne-2d-one', y='tsne-2d-two',
                    hue='Training Labels',
                    marker='o',
                    palette=sns.color_palette("hls", nbins),
                    data=df_train,legend="full",alpha=alpha*0.7, ax=ax)
    markers={}
    for i in pd.value_counts(df_test['Testing Labels']).keys():
        markers[i] = 'x'

    sns.scatterplot(x='tsne-2d-one', y='tsne-2d-two',
                    hue='Testing Labels', style='Testing Labels',
                    markers=markers, s=50, linewidth=2,
                    palette=sns.color_palette("hls", nbins),
                    data=df_test,legend='full',alpha=alpha, ax=ax)

    if figsaveto is not None:
        plt.savefig(figsaveto, bbox_inches='tight')

    return
