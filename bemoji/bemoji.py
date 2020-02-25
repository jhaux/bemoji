from edflow.data.believers.meta import MetaDataset
from edflow.util import retrieve, contains_key
import os


BEMOJI_ROOT = os.environ.get('BEMOJIROOT',
                             '/export/scratch/jhaux/Data/BEmoji/meta_dset/')


class BEmoji(MetaDataset):
    def __init__(self, config):
        if contains_key(config, 'data/base/root'):
            BEMOJI_ROOT = retrieve(config, 'data/base/root')
        super().__init__(BEMOJI_ROOT)


if __name__ == '__main__':
    B = BEmoji({})

    B.show()
