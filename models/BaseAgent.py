class BaseAgent:
    def __init__(self):
        pass

    def set_learning(self, is_learning):
        """Set the learning mode of the agent."""
        self.learning = is_learning

    def select_action(self, state):
        """Select an action based on the current state"""
        pass

    def optimize_model(self):
        """Optimize the model by sampling from the replay memory and performing a gradient descent step."""
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