from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def choose_action():
        """ Return the agent's action for the next time-step """
        pass

    @abstractmethod
    def remember():
        """ Store trajectory to in the agent's experience replay memory """
        pass

    @abstractmethod
    def learn():
        """ Train the agent """
        pass
