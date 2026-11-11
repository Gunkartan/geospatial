from training.train_upstream import train_upstream
from utils.features import UPSTREAM_MODELS

if __name__ == '__main__':
    train_upstream(
        name='water',
        model_file='water',
        features=UPSTREAM_MODELS['water'][1],
        class_prefix='4'
    )
