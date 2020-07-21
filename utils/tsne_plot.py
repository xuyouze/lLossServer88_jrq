# coding:utf-8
# @Time         : 2019/9/24 
# @Author       : xuyouze
# @File Name    : tsne_plot.py

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_plot(feature, label, n_components):
    from sklearn.decomposition import PCA

    model = TSNE(n_components=n_components, random_state=0, learning_rate=100)
    import seaborn as sns

    print('fit TSNE model...')
    #    from sklearn.decomposition import PCA
    #    x = PCA(n_components=10).fit_transform(x)
    color_mapping = {0: sns.xkcd_rgb['red'], 1: sns.xkcd_rgb['blue']}
    colors = list(map(lambda x: color_mapping[x], label[:, 0]))

    tsne = model.fit_transform(feature)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=colors)
    plt.savefig("attr 0.jpg")
