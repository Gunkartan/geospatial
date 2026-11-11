from training.train_upstream import train_upstream
from utils.features import UPSTREAM_MODELS

if __name__ == '__main__':
    train_upstream(
        name='forest',
        model_file='forest',
        features=UPSTREAM_MODELS['forest'][1],
        class_prefix='3'
    )
