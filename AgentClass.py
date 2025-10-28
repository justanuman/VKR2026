import math

class Agent:
    def __init__(self, q_value, k_level, private_signal=None):
        """
        Initialize the agent.
        
        Args:
            q_value (float): Probability of private signal being correct (between 0 and 1)
            k_level (int): The agent's K-level (0, 1, or 2)
            private_signal (bool, optional): The agent's private signal. If None, randomly generated.
        """
        self.q = q_value
        self.k_level = k_level
        self.observations = []
        
        # Set private signal
        if private_signal is None:
            # Randomly generate private signal (50/50 chance)
            self.private_signal = random.choice([True, False])
        else:
            self.private_signal = private_signal
            
    def observation(self, observations_list):
        """
        Set observations from function input.
        
        Args:
            observations_list (list of bool): List of Boolean observations from other agents
        """
        self.observations = observations_list.copy()
    
    def action(self):
        """
        Determine the agent's action based on K-level.
        
        Returns:
            bool: The agent's decision (True or False)
        """
        if self.k_level == 0:
            # K-level 0: Use private signal (or reverse if q < 0.5)
            if self.q >= 0.5:
                return self.private_signal
            else:
                return not self.private_signal
        
        elif self.k_level == 1:
            # K-level 1: Use Bayesian updating with observations
            if not self.observations:
                # If no observations, use private signal
                if self.q >= 0.5:
                    return self.private_signal
                else:
                    return not self.private_signal
            
            # Calculate probability of true state being True using Bayesian updating
            prob_true = self._bayesian_update()
            
            # Return True if probability > 0.5, False otherwise
            return prob_true > 0.5
        
        elif self.k_level == 2:
            # K-level 2: Respond recursively to others
            if not self.observations:
                # If no observations, use private signal
                if self.q >= 0.5:
                    return self.private_signal
                else:
                    return not self.private_signal
            
            # For k-level 2, we consider what lower-level agents would do
            # This is a simplified implementation - in practice, this would be more complex
            # based on the specific recursive reasoning model
            
            # Count True and False actions in observations
            true_count = sum(self.observations)
            false_count = len(self.observations) - true_count
            
            # Simple heuristic: if majority of observations are True, return True
            # In a more sophisticated implementation, this would involve
            # modeling the reasoning of lower-level agents
            if true_count > false_count:
                return True
            elif false_count > true_count:
                return False
            else:
                # Tie breaker: use private signal
                if self.q >= 0.5:
                    return self.private_signal
                else:
                    return not self.private_signal
        
        else:
            raise ValueError(f"Unsupported K-level: {self.k_level}")
    
    def _bayesian_update(self):
        """
        Calculate probability of true state being True using Bayesian updating.
        
        Returns:
            float: Probability that the true state is True
        """
        if not self.observations:
            # If no observations, prior is 0.5
            prior = 0.5
            # Update with private signal
            if self.private_signal:
                # P(state=True | signal=True) = P(signal=True | state=True) * P(state=True) / P(signal=True)
                # P(signal=True) = P(signal=True | state=True) * P(state=True) + P(signal=True | state=False) * P(state=False)
                #               = q * 0.5 + (1-q) * 0.5 = 0.5
                posterior = (self.q * prior) / 0.5
            else:
                # P(state=True | signal=False) = P(signal=False | state=True) * P(state=True) / P(signal=False)
                # P(signal=False) = 0.5
                posterior = ((1 - self.q) * prior) / 0.5
            return posterior
        
        # Start with uniform prior
        prob_true = 0.5
        
        # Update probability based on each observation
        for obs in self.observations:
            if obs:
                # If observation is True
                # P(state=True | obs=True) = P(obs=True | state=True) * P(state=True) / P(obs=True)
                # We assume the observation comes from an agent with q=0.6 (typical assumption)
                # This could be parameterized if needed
                q_other = 0.6
                prob_obs_given_true = q_other
                prob_obs_given_false = 1 - q_other
                
                prob_obs = prob_obs_given_true * prob_true + prob_obs_given_false * (1 - prob_true)
                prob_true = (prob_obs_given_true * prob_true) / prob_obs
            else:
                # If observation is False
                q_other = 0.6
                prob_obs_given_true = 1 - q_other
                prob_obs_given_false = q_other
                
                prob_obs = prob_obs_given_true * prob_true + prob_obs_given_false * (1 - prob_true)
                prob_true = (prob_obs_given_true * prob_true) / prob_obs
        
        # Finally, update with private signal
        if self.private_signal:
            prob_signal_given_true = self.q
            prob_signal_given_false = 1 - self.q
            
            prob_signal = prob_signal_given_true * prob_true + prob_signal_given_false * (1 - prob_true)
            prob_true = (prob_signal_given_true * prob_true) / prob_signal
        else:
            prob_signal_given_true = 1 - self.q
            prob_signal_given_false = self.q
            
            prob_signal = prob_signal_given_true * prob_true + prob_signal_given_false * (1 - prob_true)
            prob_true = (prob_signal_given_true * prob_true) / prob_signal
        
        return prob_true
    
    def __str__(self):
        """String representation of the agent."""
        return f"Agent(q={self.q}, k_level={self.k_level}, private_signal={self.private_signal})"