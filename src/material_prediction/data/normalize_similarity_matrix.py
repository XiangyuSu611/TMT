import os
import pandas
import numpy as np
from collections import defaultdict


def main():
    simi_matrix_path = './data/training_data/material_prediction/total_similarity_matrix_sqrt.csv'
    save_path = './data/training_data/material_prediction/'  

    # read the.csv file as ndarray and drop the index value of the first column
    simi_matrix_data_frame = pandas.read_csv(simi_matrix_path)
    simi_matrix_ori = np.array(simi_matrix_data_frame, dtype=float)
    simi_matrix = simi_matrix_ori[:, 1:simi_matrix_ori.shape[1]]

    # normalize the similarity distance matrix according to maximum of every row
    simi_matrix_norm2 = simi_matrix.copy()
    for row in range(simi_matrix_norm2.shape[0]):
        row_maximum = simi_matrix_norm2[row].max()
        simi_matrix_norm2[row] = simi_matrix_norm2[row] / row_maximum

    simi_matrix_norm2 = pandas.DataFrame(simi_matrix_norm2)
    simi_matrix_norm2.columns = list(simi_matrix_ori[:,0].astype('int'))
    simi_matrix_norm2.index = list(simi_matrix_ori[:,0].astype('int'))

    # save
    simi_matrix_norm2.to_csv(save_path + 'similarity_matrix_norm_row.csv')


if __name__ == '__main__':
    main()
