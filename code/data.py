import random
import json
import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse

from pfa import ProbablisticFiniteAutomata
from distribution import sample_distribution
from utils import sample_cfg

class DatasetGenerator:
    """
    Base class for dataset generation.
    """
    def __init__(self, 
                dataset_name="default",
                num_seq=16,
                seq_len=4,
                train_examples=1000,
                val_examples=10,
                seed=42,
                symbols=None):
        """
        Initialize dataset generator.
        
        Args:
            dataset_name: Name of the dataset
            num_seq: Number of sequences per example
            seq_len: Length of each sequence
            train_examples: Number of training examples
            val_examples: Number of validation examples
            seed: Random seed
            symbols: List of symbols to use (default: A-H)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.dataset_name = dataset_name
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.train_examples = train_examples
        self.val_examples = val_examples
        
        self.symbols = symbols or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        # Create dataset directory
        os.makedirs(f'data/{self.dataset_name}', exist_ok=True)
        
        # Initialize files
        self.train_file = open(f'data/{self.dataset_name}/train.txt', 'w+')
        self.val_file = open(f'data/{self.dataset_name}/val.txt', 'w+')
        self.train_pfa_data_file = open(f'data/{self.dataset_name}/train_pfa_data.pkl', 'wb')
        self.val_pfa_data_file = open(f'data/{self.dataset_name}/val_pfa_data.pkl', 'wb')
        self.config_file = open(f'data/{self.dataset_name}/config.json', 'w+')
        
        # Initialize alignment
        self.alignment = list(range(len(self.symbols)))
        
    def setup_alignment(self, alignment_type="sequential", param=None):
        """
        Set up alignment for the dataset.
        
        Args:
            alignment_type: Type of alignment (sequential, permuted, distribution)
            param: Parameter for distribution alignment (lambda value)
        """
        alignment_file = open(f'data/{self.dataset_name}/alignment.txt', 'w')
        
        if alignment_type == "permuted":
            self.alignment = random.sample(range(len(self.symbols)), len(self.symbols))
        elif alignment_type == "distribution":
            self.alignment = sample_distribution(len(self.symbols), len(self.symbols), 0, float(param))
            self.alignment = [x-1 for x in self.alignment]
        else:  # sequential
            self.alignment = list(range(len(self.symbols)))
            
        alignment_file.write(','.join([str(x) for x in self.alignment]))
        alignment_file.close()
        
    def generate_sequence(self, num_examples, outfile, pfa_data_file, ood=False):
        """
        Generate sequences for the dataset.
        
        Args:
            num_examples: Number of examples to generate
            outfile: Output file to write examples to
            pfa_data_file: Output file to write PFA data to
            ood: Whether to generate out-of-distribution examples
        """
        pfa_data = []

        for _ in tqdm(range(num_examples)):
            V = [chr(i) for i in range(97, 97+26)]
            random.shuffle(V)
            pfas = {x: ProbablisticFiniteAutomata(V) for x in self.symbols}

            example_sequence = ''
            for _ in range(self.num_seq):
                if ood:
                    shuffled_X = random.sample(self.symbols, len(self.symbols))
                else:
                    shuffled_X = sample_cfg('Z')

                permuted_X = [shuffled_X[self.alignment[j]] for j in range(len(shuffled_X))]
                
                example_sequence += ''.join([f'!' + x for _, x in enumerate(shuffled_X)]) + ' : '
                for _, x in enumerate(permuted_X):
                    example_sequence += f'!' + f'!'.join(pfas[x].generate_sequence(length=1))
                example_sequence += ' '

            print(example_sequence, file=outfile)

            shuffled_pfas = [pfas[x] for x in permuted_X]
            pfa_data.append(shuffled_pfas)

        config = {
            'num_seq': self.num_seq,
            'seq_len': self.seq_len,
            'train_examples': self.train_examples,
            'val_examples': self.val_examples,
            'seed': self.seed,
            'symbols': self.symbols,
            'alignment': self.alignment
        }
        json.dump(config, self.config_file)
        self.config_file.close()

        pickle.dump(pfa_data, pfa_data_file)
        
    def generate_dataset(self):
        """
        Generate the full dataset (train and validation).
        """
        self.generate_sequence(self.train_examples, self.train_file, self.train_pfa_data_file)
        self.generate_sequence(self.val_examples, self.val_file, self.val_pfa_data_file)
        
        self.train_file.close()
        self.val_file.close()
        
    def cleanup(self):
        """
        Close all open files.
        """
        if not self.train_file.closed:
            self.train_file.close()
        if not self.val_file.closed:
            self.val_file.close()


class DistributionDatasetGenerator(DatasetGenerator):
    """
    Generator for distribution-based datasets.
    """
    def __init__(self, lambda_param=16, **kwargs):
        """
        Initialize distribution dataset generator.
        
        Args:
            lambda_param: Lambda parameter for distribution
            **kwargs: Additional arguments for base class
        """
        dataset_name = kwargs.get('dataset_name', f'distribution-{lambda_param}')
        super().__init__(dataset_name=dataset_name, **kwargs)
        self.setup_alignment(alignment_type="distribution", param=lambda_param)


class PermutedDatasetGenerator(DatasetGenerator):
    """
    Generator for permuted datasets.
    """
    def __init__(self, **kwargs):
        """
        Initialize permuted dataset generator.
        """
        dataset_name = kwargs.get('dataset_name', 'permuted')
        super().__init__(dataset_name=dataset_name, **kwargs)
        self.setup_alignment(alignment_type="permuted")


def generate_sequential_dataset(train_examples=1000, val_examples=10):
    """
    Generate a sequential dataset.
    
    Args:
        train_examples: Number of training examples
        val_examples: Number of validation examples
    """
    generator = DatasetGenerator(
        dataset_name='sequential',
        train_examples=train_examples,
        val_examples=val_examples
    )
    generator.setup_alignment(alignment_type="sequential")
    generator.generate_dataset()
    generator.cleanup()


def generate_distribution_dataset(lambda_param=16, train_examples=1000, val_examples=10):
    """
    Generate a distribution-based dataset.
    
    Args:
        lambda_param: Lambda parameter for distribution
        train_examples: Number of training examples
        val_examples: Number of validation examples
    """
    generator = DistributionDatasetGenerator(
        lambda_param=lambda_param,
        train_examples=train_examples,
        val_examples=val_examples
    )
    generator.generate_dataset()
    generator.cleanup()
    
def generate_permuted_dataset(train_examples=1000, val_examples=10):
    """
    Generate a permuted dataset.
    
    Args:
        train_examples: Number of training examples
        val_examples: Number of validation examples
    """
    generator = PermutedDatasetGenerator(
        train_examples=train_examples,
        val_examples=val_examples
    )
    generator.generate_dataset()
    generator.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets for training and evaluation")
    parser.add_argument('--type', type=str, default='distribution', choices=['distribution', 'permuted', 'sequential'],
                        help='Type of dataset to generate')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=16,
                        help='Lambda parameter for distribution dataset')
    parser.add_argument('--train_examples', type=int, default=1000,
                        help='Number of training examples to generate')
    parser.add_argument('--val_examples', type=int, default=10,
                        help='Number of validation examples to generate')
    
    args = parser.parse_args()
    
    if args.type == 'distribution':
        generate_distribution_dataset(
            lambda_param=args.lambda_param,
            train_examples=args.train_examples,
            val_examples=args.val_examples
        )
    elif args.type == 'permuted':
        generate_permuted_dataset(
            train_examples=args.train_examples,
            val_examples=args.val_examples
        )
    else:
        generate_sequential_dataset(
            train_examples=args.train_examples,
            val_examples=args.val_examples
        )