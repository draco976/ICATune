import random
from typing import List
import numpy as np

class ProbablisticFiniteAutomata:
    def __init__(self, V=None) -> None:
        """
        Initialize a random PFA based on the given vocabulary V.
        
        Args:
            V: Vocabulary to sample alphabet from. If None, uses lowercase letters.
        """
        if V is None:
            V = [chr(i) for i in range(97, 123)]  # lowercase letters
            
        self.num_states = random.randint(4, 12)
        self.alphabet_size = random.randint(4, 18)
        self.alphabet = random.sample(V, self.alphabet_size)
        self.transitions = {}
        self.symbol_transition = {}

        self.initial_state = 0

        for i in range(self.num_states+1):
            state = i
            self.transitions[state] = []
            self.symbol_transition[state] = {}

            num_transitions = random.randint(1, 4)

            next_states = random.sample(range(1, self.num_states+1), num_transitions)
            letters = random.sample(self.alphabet, num_transitions)

            for i in range(num_transitions):
                next_state = next_states[i]
                letter = letters[i]
                self.transitions[state].append((next_state, letter))
                self.symbol_transition[state][letter] = next_state

    def generate_sequence(self, length: int) -> str:
        """
        Generate a sequence of given length from the PFA.
        
        Args:
            length: Length of sequence to generate
            
        Returns:
            Generated sequence string
        """
        current_state = self.initial_state
        sequence = ""

        for _ in range(length):
            next_state, letter = random.choice(self.transitions[current_state])
            sequence += letter
            current_state = next_state

        return sequence
    
    def check_sequence(self, sequence: str) -> bool:
        """
        Check if the given sequence is accepted by the PFA.
        
        Args:
            sequence: Sequence to check
            
        Returns:
            True if sequence is accepted, False otherwise
        """
        current_state = self.initial_state

        for letter in sequence:
            next_state = self.symbol_transition[current_state].get(letter, None)
            if next_state is None:
                return False
            current_state = next_state

        return True

    def evaluate_sequence(self, actual: str, prediction: str):
        """
        Evaluate the given prediction based on the actual sequence.
        
        Args:
            actual: Ground truth sequence
            prediction: Predicted sequence
            
        Returns:
            Evaluation scores for each position in prediction
        """
        evaluation = np.zeros(len(prediction))

        for idx, x in enumerate(prediction):
            evaluation[idx] = float(self.check_sequence(actual[:idx]+x))

        return evaluation