from collections import deque
from random import sample as random_sample

class Memory:

    def __init__(self, max_capacity=1e5):
        """
        Wraps collectons.deque to serve a "memory" or database to store experiences. 

        Attributes:
            - self.max_capacity=1e5 (int): Maximum number of memories that can
            be stored
            - self.memory (collections.deque): Stores the memories
        """
        self.max_capacity = int(max_capacity)
        self.memory = deque(maxlen=self.max_capacity)

    def __len__(self):
        """ Returns number of transitions stored in memory """
        return len(self.memory)

    def __getitem__(self, key):
        """ Direct indexing of deque memory object """
        return self.memory[key]

    def remember(self, *args):
        """ Store transition """
        # The appened function of a deque object will overwrite the oldest entries
        # after the deque object's maxlen is reached.
        self.memory.append(args)

    def sample(self, n_batches):
        """ Sample minibatch """
        return random_sample(self.memory, n_batches)
    
    def clear(self):
        """ Clear all elements from the memory """ 
        self.memory.clear()
