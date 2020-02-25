import numpy as np
import os
from tqdm.autonotebook import tqdm
from edflow.data.believers.meta_util import store_label_mmap


def generate_labels(data_root, label_root):
    ids = np.load(os.path.join(data_root, 'identities.npy'))
    sizes = np.load(os.path.join(data_root, 'sizes.npy'))
    trajs = np.load(os.path.join(data_root, 'trajectory_seeds.npy'))
    health = np.load(os.path.join(data_root, 'health_levels.npy'))

    labels_ = [
            'base_trajectory',
            'finl_trajectory',
            'identity',
            'image_path',
            'sick_trajectory',
            'speed',
            'trajectory_seed',
            'uniq_trajectory',
            'unique_frequencey',
            'unique_speed',
            'health_level',
            'size'
            ]


    path_proto = os.path.join(data_root,
                              'id:{:0>3d}/s:{:0>3d}/t:{:0>4d}/h:{:0.3f}/',
                              '{}.npy')

    labels = {l: [] for l in labels_}
    labels['fid'] = []
    for i in tqdm(ids, desc='i'):
        for s in tqdm(sizes, desc='s'):
            for t in tqdm(trajs, desc='t'):
                for h in health:
                    for l in labels_[:1]:
                        label_data = np.load(path_proto.format(i, s, t, h, l))
                        labels[l] += [label_data]
                    # labels['fid'] += [np.arange(len(label_data))]
                    labels['size'] += [[s] * len(label_data)]

    for k, v in labels.items():
        if k != 'size':
            continue
        v = np.concatenate(v)

        store_label_mmap(v, label_root, k)


def generate_view(view_root, seq_len=100, **bmkwargs):
    from bemoji import BEmoji
    from edflow.data.believers.sequence import get_sequence_view

    B = BEmoji(bmkwargs)
    frame_ids = B.labels['fid']

    view = get_sequence_view(frame_ids, seq_len, step=1, strategy="raise")
    n_seq = len(view)

    view_path = os.path.join(view_root, 'labels')
    os.makedirs(view_path, exist_ok=True)

    store_label_mmap(view, view_path, 'seq_view')

    with open(os.path.join(view_root, 'meta.yaml'), 'w+') as mf:
        mf.write(f'''
description: |
    # Sequence View with length {seq_len}

    Contains {n_seq} sequences of the base dataset

base_dset: bemoji.bemoji.BEmoji
base_kwargs: {{config: {{}}}}

views:
    seq_view
    ''')


if __name__ == '__main__':
    # generate_labels(
    #         '/export/scratch/jhaux/Data/BEmoji/Data',
    #         '/export/scratch/jhaux/Data/BEmoji/meta_dset/labels'
    #         )

    seq_len = 100
    view_root = f'/home/jhaux/Dr_J/Data/BEmoji/seq_{seq_len}'
    base_root = f'/home/jhaux/Dr_J/Data/BEmoji/meta_dset'
    os.makedirs(view_root, exist_ok=True)
    generate_view(view_root, seq_len, root=base_root)
