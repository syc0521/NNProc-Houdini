from enum import Enum

class TrainingMode(Enum):
    train = 'train'
    test = 'test'

training_mode = TrainingMode.train
batch_size = 64