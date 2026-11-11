from training.train_upstream import train_upstream
from utils.features import UPSTREAM_MODELS

if __name__ == '__main__':
    train_upstream(
        name='misc',
        model_file='misc',
        features=UPSTREAM_MODELS['misc'][1],
        class_prefix='5'
    )
