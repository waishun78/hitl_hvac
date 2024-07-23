class BaseAgent:
    def __init__(self):
        pass

    def set_exploring(self, is_exploring):
        """Set the learning mode of the agent."""
        self.is_exploring = is_exploring

    def select_action(self, state):
        """Select an action based on the current state"""
        pass

    def reset_replay_memory(self):
        """Reset replay memory used to optimize the model"""
        pass

    def optimize_model(self):
        """Optimize the model by sampling from the step performed performing a gradient descent step."""
        pass

    def memorize(self, state, action, next_state, reward):
        """Store a transition in the replay memory."""
        pass

    def save_model(self, filepath):
        """Save the model parameters to a file."""
        pass

    def load_model(self, filepath):
        """Load the model parameters from a file."""
        pass