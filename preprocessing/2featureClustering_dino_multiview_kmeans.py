import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import mvlearn
from mvlearn.cluster import MultiviewKMeans, MultiviewSpectralClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from skfuzzy.cluster import cmeans
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import glob
import random
import json
import torch
import torch.nn as nn
from tqdm import tqdm


parser = argparse.ArgumentParser(description='K-means for json file')
# parser.add_argument('--giga_feat_path', default='/hdd_data/yangzd/vision_transformer/orthogonal_features_training/gigapath_features', metavar='WSI_PATH', type=str,
#                     help='Path to the input WSI file')
# parser.add_argument('--uni_feat_path', default='/hdd_data/yangzd/vision_transformer/orthogonal_features_training/uni_features', metavar='WSI_PATH', type=str,
#                     help='Path to the input WSI file')
# parser.add_argument('--conch_feat_path', default='/hdd_data/yangzd/vision_transformer/orthogonal_features_training/conch_features', metavar='WSI_PATH', type=str,
#                     help='Path to the input WSI file')
# parser.add_argument('--position_path', default='/hdd_data/yangzd/vision_transformer/orthogonal_features_training/conch_features', metavar='WSI_PATH', type=str,
#                     help='Path to the input WSI file')
# parser.add_argument('--save_txt_path', default='/hdd_data/yangzd/vision_transformer/orthogonal_features_training/conch_features', metavar='TXT_PATH', type=str,
#                     help='Path to the input WSI file')
# parser.add_argument('--txt_path', default='/hdd_data/yangzd/vision_transformer/orthogonal_features_training/train_direct_concat', metavar='TXT_PATH', type=str,
#                     help='Path to the input WSI file')
parser.add_argument('--giga_feat_path', default='/hdd_data/yangzd/TCGA_features_analysis/gigapath_features', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--uni_feat_path', default='/hdd_data/yangzd/TCGA_features_analysis/uni_features', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--conch_feat_path', default='/hdd_data/yangzd/TCGA_features_analysis/conch_features', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--position_path', default='/hdd_data/yangzd/TCGA_features_analysis/conch_features', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--save_txt_path', default='/hdd_data/yangzd/TCGA_features_analysis/conch_features', metavar='TXT_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--txt_path', default='/hdd_data/yangzd/TCGA_features_analysis/multiview_kmeans_feat_clustering', metavar='TXT_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--class_num', default=50, metavar='CLASS_NUM', type=int, help='Clustering Number Class')
parser.add_argument('--scale_ratio', default=512, metavar='SCALE_RATIO', type=int, help='Coordinate reduction factor')

random = np.random.RandomState(0)


class OrthogonalLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.projector = torch.nn.utils.parametrizations.orthogonal(nn.Linear(in_dim, out_dim, bias=False))
        self.weight = nn.Parameter(torch.ones(out_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, x):
        x = self.projector(x)
        x = self.weight * x
        if self.bias is not None:
            x = x + self.bias
        return x


def main():
    args = parser.parse_args()
    #run2(args)      #将torch的特征形式写为WSI的txt格式
    run1(args)      #特征聚类
    # print(111)


def randomcolor(class_num):
    colorList = []
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    for num in range(class_num):
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0,14)]
        colorList.append("#"+color)
    return colorList


def loadDataSet(fileName):
    with open(fileName, 'r')as fp:
        json_data = fp.readlines()
        featName = []
        feats = []
        for line in json_data:
            lineList = line.split(':')
            featName.append(lineList[0][2:-1])
            feats.append(list(map(float, lineList[1][2:-3].split(', '))))
    return featName, feats


def load_position(position_path):
    wsi_count = []
    with open(position_path, "r") as f:
        content = f.read()
        path_data = json.loads(content)
    wsi_name_list= list(map(lambda x: os.path.split(os.path.dirname(x))[-1], path_data))
    patch_name_list = list(map(lambda x: os.path.basename(x)[:-4], path_data))
    patch_position_list = list(map(lambda x: x[x.find('_') + 1:], patch_name_list))
    temp_wsi_name = wsi_name_list[0]
    for patch_idx, wsi_name in enumerate(wsi_name_list):
        if temp_wsi_name != wsi_name_list[patch_idx]:
            wsi_count.append(patch_idx)
            temp_wsi_name = wsi_name_list[patch_idx]
    wsi_count.append(len(wsi_name_list))
    return wsi_count, wsi_name_list, patch_position_list
    

def showClass(args, json_name, feat_name, features, labels, color_list, ext=''):
    png_file = os.path.join(args.txt_path, json_name.replace('.txt', ext + '.png'))
    plt.rcParams['axes.unicode_minus'] = False
    for i, coords in enumerate(feat_name):
        coord = coords.split(',')
        plt.plot(int(coord[0][1:]), int(coord[1][1:-1]), color=color_list[labels[i]], marker='.', markersize=1)

    plt.savefig(png_file, bbox_inches='tight', dpi=1000)
    plt.clf()


def showEvaluate(args, feat_name, features, labels, json_name, ext):
    data = (list(zip(feat_name, labels)))
    data = pd.DataFrame(data)
    json_name = json_name.replace('.txt', ext + '_cls.txt')
    txt_path = os.path.join(args.txt_path, json_name)
    data.to_csv(txt_path, sep='\t', index=0, header=0)


# K-Means clustering
def kmeans_train(features, clusters):
    return KMeans(n_clusters=clusters, n_init='auto').fit(features)


# Multiview K-Means clustering
def mv_kmeans_train(features, clusters):
    return MultiviewKMeans(n_clusters=clusters, n_init=100, random_state=3232).fit_predict(features)


# Multiview Spectral clustering
def mv_spectral_train(features, clusters):
    return MultiviewSpectralClustering(n_clusters=clusters, affinity='nearest_neighbors', random_state=clusters, n_init=clusters).fit_predict(features)


def run1(args):
    giga_paths = glob.glob(os.path.join(args.giga_feat_path, '*.txt'))
    uni_paths = glob.glob(os.path.join(args.uni_feat_path, '*.txt'))
    conch_paths = glob.glob(os.path.join(args.conch_feat_path, '*.txt'))
    color_list = randomcolor(args.class_num)
    giga_paths.sort()
    giga_paths.reverse()
    uni_paths.sort()
    uni_paths.reverse()
    conch_paths.sort()
    conch_paths.reverse()

    for giga_path, uni_path, conch_path in zip(giga_paths, uni_paths, conch_paths):
        json_name = os.path.basename(giga_path)
        print(json_name)
        # exit(0)
        if os.path.exists(os.path.join(args.txt_path, json_name.replace('.txt', 'kmeans.png'))):
            continue
        feat_name, Xgiga = loadDataSet(giga_path)
        Xgiga_np = np.array(Xgiga)
        _, Xuni = loadDataSet(uni_path)
        Xuni_np = np.array(Xuni)
        _, Xconch = loadDataSet(conch_path)
        Xconch_np = np.array(Xconch)
        features = [Xgiga,Xuni,Xconch]

        if len(features) < 1:
            continue
        # cls_nums = args.class_num if len(features) >= args.class_num else len(features)
        # print(cls_nums)
        cls_nums = 50
        print('Clustering....')
        kmeans_labels_ = mv_spectral_train(features, cls_nums)
        print('Writing...')
        showEvaluate(args, feat_name, features, kmeans_labels_, json_name, '_mvkmeans')
        print('Ploting...')
        showClass(args, json_name, feat_name, features, kmeans_labels_,color_list, '_mvkmeans')
        exit(0)


def run2(args):
    wsi_count_list, wsi_name_list, patch_position_list = load_position(args.position_path)
    x_position_list = list(map(lambda x: int(x[:x.find('_')]), patch_position_list))
    y_position_list = list(map(lambda x: int(x[x.find('_') + 1:]), patch_position_list))
    patch_features = torch.load(args.feat_path)
    start_count = 0
    for wsi_idx, wsi_count in enumerate(wsi_count_list):
        print('{}/{} : {}, {}'.format(str(wsi_idx + 1), str(len(wsi_count_list)), wsi_name_list[start_count], str(wsi_count)))
        wsi_txt_path = os.path.join(args.save_txt_path, wsi_name_list[start_count] + '.txt')
        with open(wsi_txt_path, "w") as f:
            idList = list(zip(x_position_list[start_count:wsi_count], y_position_list[start_count:wsi_count]))
            featureList = list(patch_features[start_count:wsi_count].tolist())
            for idNum,idName in tqdm(enumerate(idList)):
                featureDict = {}
                featureDict['{}'.format(str(idName))]=featureList[idNum]
                json.dump(featureDict, f)
                f.write('\n')
        start_count = wsi_count


if __name__ == '__main__':
    main()

