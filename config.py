import json

class Configuration(object):
    
    def __init__(self):

        # Network Architecture
        self.im_size = [192, 192]
        self.multiplier = 1
        self.has_sigmoid = False

        # Data augmentation
        self.random_scale = [0.8, 1.2] # None # (0.8,1.2)
        self.random_displacement = 0.2 # None # 0.1
        self.random_flip = True

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.weight_decay = 2e-6
        self.epochs = 50
        self.optimizer = 'sgd'
        self.momentum = 0.9
        self.pos_weight = 5.0

        # File management
        self.epochs_per_save = 1

        # Evaluation
        self.pos_threshold = 0.5

        # Visualization
        self.batches_per_print = 10

    def save(self, filename):
        config = {}
        # Network Architecture
        config['im_size'] = self.im_size
        config['multiplier'] = self.multiplier
        config['has_sigmoid'] = self.has_sigmoid

        # Data augmentation
        config['random_scale'] = self.random_scale
        config['random_displacement'] = self.random_displacement
        config['random_flip'] = self.random_flip

        # Training parameters
        config['batch_size'] = self.batch_size
        config['learning_rate'] = self.learning_rate
        config['weight_decay'] = self.weight_decay
        config['epochs'] = self.epochs
        config['optimizer'] = self.optimizer
        config['momentum'] = self.momentum
        config['pos_weight'] = self.pos_weight

        # File management
        config['epochs_per_save'] = self.epochs_per_save

        # Evaluation
        config['pos_threshold'] = self.pos_threshold

        # Visualization
        config['batches_per_print'] = self.batches_per_print
        with open(filename, 'w') as fp:
            json.dump(config, fp)

    def load(self, filename):
        with open(filename, 'r') as jf:
            config = json.loads(jf.read())

        # Network Architecture
        self.im_size = config['im_size']
        self.multiplier = config['multiplier']
        self.has_sigmoid = config['has_sigmoid']

        # Data augmentation
        self.random_scale = config['random_scale']
        self.random_displacement = config['random_displacement']
        self.random_flip = config['random_flip']

        # Training parameters
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.epochs = config['epochs']
        self.optimizer = config['optimizer']
        self.momentum = config['momentum']
        self.pos_weight = config['pos_weight']

        # File management
        self.epochs_per_save = config['epochs_per_save']

        # Evaluation
        self.pos_threshold = config['pos_threshold']

        # Visualization
        self.batches_per_print = config['batches_per_print']

