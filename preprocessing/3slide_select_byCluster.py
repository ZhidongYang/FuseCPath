import pandas as pd
import numpy as np
import argparse
import glob
import os
import random
import openslide
import cv2
import math

parser = argparse.ArgumentParser(description='K-menas for json file')
parser.add_argument('--txt_path', default='./clustering_and_patch_selection/dino_feat_clustering', metavar='TXT_PATH', type=str,
                    help='INPUT: Path to the input WSI file')
parser.add_argument('--txt_res_path', default='./clustering_and_patch_selection/res_select_txt', metavar='TXT_RES_PATH', type=str,
                    help='OUTPUT: Path to the input WSI file')
parser.add_argument('--num_per_cls', default='10', metavar='NUM_PER_CLASS', type=int,help='The count of patch for every clustering class')

random = np.random.RandomState(0)

def run(args):
    txt_res_path = args.txt_res_path
    paths = glob.glob(os.path.join(args.txt_path, '*.txt'))
    for path in paths:
        print(path)
        basename = os.path.basename(path)
        png_name = basename[:basename.find('kmeans')]
        file_object = open(path)
        try:
            file_content = file_object.read()
        finally:
            file_object.close()
        a = {}
        results = file_content.split('\n')
        results = list(filter(None, results))
        for i in range(len(results)):
            coord_cls = results[i].split('\t')
            if int(coord_cls[1]) in a:
                a[int(coord_cls[1])].append(coord_cls[0])
            else:
                a[int(coord_cls[1])] = [coord_cls[0]]

        if len(a) < 50:
            print('The effective tissue area of the current image is too small and can be ignored--{}'.format(basename))
            continue

        max_cluster_count = max(len(a[idx]) for idx in range(50))   # the maximum of the items in a cluster
        max_cluster_iters = math.ceil(max_cluster_count/args.num_per_cls)
        for key, values in a.items():
            random.shuffle(a[key])

        for key, value in a.items():
            if len(value) < (max_cluster_iters) * 60:
                repeat_count = math.ceil((max_cluster_iters) * 60 / len(value))
                a[key] = value * repeat_count
                assert len(a[key]) >= (max_cluster_iters * 60)

        for key, values in a.items():
            random.shuffle(a[key])

        for i in range(50):
            coords = []
            for key, value_long in a.items():
                lower_bound = i*10
                upper_bound = (i+1)*10
                if upper_bound >= len(value_long) or lower_bound >= len(value_long):
                    choice_coord = np.random.choice(value_long, size=10, replace=False)
                else:
                    choice_coord = value_long[i*10:(i+1)*10]
                coords.append(choice_coord)
            data = pd.DataFrame(coords)
            data.to_csv(os.path.join(txt_res_path, os.path.basename(path).replace('.txt', '_{}.txt'.format(str(i)))), sep='\t', index=0, header=0)

        # exit(0)

def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()



