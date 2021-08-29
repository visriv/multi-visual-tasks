import argparse
import pickle

import numpy as np
import torch
from mvt.cores.metric_ops import LpDistance


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate emdedding')
    parser.add_argument('reference', help='reference embedding file path')
    parser.add_argument('query', help='query embedding file path')
    args = parser.parse_args()

    return args


def load_embedding(file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data['embeddings'], data['labels']


def print_info(embeddings, labels):
    print('Number of samples: ', embeddings.shape[0])
    print('Number of classes: ', np.unique(labels).size)


def main():

    args = parse_args()
    ref_path = args.reference
    qry_path = args.query

    ref_emb, ref_labels = load_embedding(ref_path)
    print('Loaded reference embeddings')
    print_info(ref_emb, ref_labels)
    qry_emb, qry_labels = load_embedding(qry_path)
    print('Loaded query embeddings')
    print_info(qry_emb, qry_labels)

    valid_inds = []
    valid_labels = np.unique(ref_labels)
    for i, label in enumerate(qry_labels):
        if label in valid_labels:
            valid_inds.append(i)

    qry_emb = qry_emb[valid_inds, :]
    qry_labels = qry_labels[valid_inds, :]
    print('Valid query embeddings')
    print_info(qry_emb, qry_labels)

    dist_func = LpDistance()
    ref_emb = torch.from_numpy(ref_emb).cuda()
    qry_emb = torch.from_numpy(qry_emb).cuda()

    mat = dist_func(qry_emb, ref_emb)
    mat_inds = torch.argsort(mat, dim=1).data.cpu().numpy()

    ref_labels = ref_labels.reshape((ref_labels.shape[0],))
    assigned_labels = ref_labels[mat_inds]
    
    acc = np.sum(assigned_labels[:, 0] == qry_labels[:,0]) / qry_labels.shape[0]
    print('Rank 1 accuracy = {:.2f}%'.format(acc * 100.0))

    for k in [10, 20, 50, 100, 200]:
        print('Assign label by frequency from top {} predictions'.format(k))
        tp = 0
        for i in range(assigned_labels.shape[0]):
            pred = np.argmax(np.bincount(assigned_labels[i, :k]))
            if pred == qry_labels[i,0]:
                tp += 1

        acc = (100.0 * tp) / qry_labels.shape[0]
        print('Accuracy = {:.2f}%'.format(acc))

if __name__ == '__main__':
    main()
