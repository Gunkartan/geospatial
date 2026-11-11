from training.train_upstream import train_upstream
from utils.features import UPSTREAM_MODELS

if __name__ == '__main__':
    train_upstream(
        name='building',
        model_file='buildings',
        features=UPSTREAM_MODELS['building'][1],
        class_prefix='1'
    )
