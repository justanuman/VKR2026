import numpy as np

class BayesianAgent:
    def __init__(self, private_signal, k_level, priors=None):
        self.private_signal = private_signal
        self.k_level = k_level
        self.observations = []
        
        # Set priors - default to uniform if not provided
        if priors is not None:
            self.priors = priors.copy()
        else:
            self.priors = [0.5, 0.5]  # [P(True), P(False)]
        
        # Normalize priors to sum to 1
        total = sum(self.priors)
        self.priors = [p / total for p in self.priors]
    
    def observation(self, observations_list):
        self.observations.extend(observations_list)
    
    def bayesian_update(self, signal, likelihood_true=0.5, likelihood_false=0.5):
        """
        Apply Bayesian update to priors given a signal.
        
        Args:
            signal (bool): The signal to incorporate
            likelihood_true (float): P(signal=True | True state)
            likelihood_false (float): P(signal=False | False state)
            
        Returns:
            list: Updated posterior probabilities
        """
        # P(True) and P(False) from priors
        p_true, p_false = self.priors
        
        if signal:
            # If signal is True, use likelihoods for True state
            posterior_true = p_true * likelihood_true
            posterior_false = p_false * (1 - likelihood_false)
        else:
            # If signal is False, use likelihoods for False state
            posterior_true = p_true * (1 - likelihood_true)
            posterior_false = p_false * likelihood_false
        
        # Normalize
        total = posterior_true + posterior_false
        if total > 0:
            return [posterior_true / total, posterior_false / total]
        else:
            return [0.5, 0.5]  # Default if division by zero
    
    def action(self):
        """
        Determine the supposed state of the world based on K-level reasoning.
        Uses Bayesian updating with private signal and observations.
        
        Returns:
            bool: The agent's belief about the true state of the world
        """
        current_belief = self.priors.copy()
        
        # K-level 0: Use only private signal
        if self.k_level == 0:
            current_belief = self.bayesian_update(self.private_signal,0.5,0.5)
            decision = current_belief[0] > current_belief[1]
            
        # K-level 1: Use private signal and think others use only private signals
        elif self.k_level == 1:
            # Start with private signal
            current_belief = self.bayesian_update(self.private_signal,0.5,0.5)
            
            # Update with observations (assuming they are raw signals)
            for obs in self.observations:
                current_belief = self.bayesian_update(obs)
                self.priors = current_belief  # Update priors for sequential updating
            
            decision = current_belief[0] > current_belief[1]
            
        # K-level 2: Think that others are level-1 thinkers
        elif self.k_level == 2:
            # First, what would level-1 agents believe?
            level1_beliefs = []
            for obs in self.observations:
                # Simulate what a level-1 agent would believe given this observation
                level1_agent = BayesianAgent(obs, 1, self.priors.copy())
                level1_belief = level1_agent.bayesian_update(obs,0.5,0.5)
                level1_beliefs.append(level1_belief[0] > 0.5)  # Their decision
            
            # Update belief using level-1 agents' decisions as sophisticated signals
            current_belief = self.bayesian_update(self.private_signal,0.5,0.5)
            for l1_decision in level1_beliefs:
                current_belief = self.bayesian_update(l1_decision, 0.6, 0.6)  # Weaker signal since it's processed
                self.priors = current_belief
                
            decision = current_belief[0] > current_belief[1]
            
        # Higher K-levels: Recursive thinking
        else:
            # Start with private signal
            current_belief = self.bayesian_update(self.private_signal)
            
            # For each observation, simulate what lower-level agents would believe
            for obs in self.observations:
                # Create agent with k_level-1
                lower_agent = BayesianAgent(obs, self.k_level-1, self.priors.copy())
                lower_belief = lower_agent.bayesian_update(obs)
                lower_decision = lower_belief[0] > lower_belief[1]
                
                # Update based on the lower-level agent's decision
                current_belief = self.bayesian_update(lower_decision, 0.55, 0.55)  # Even weaker signal
                self.priors = current_belief
                
            decision = current_belief[0] > current_belief[1]
        
        return decision
    
    def get_beliefs(self):
        """
        Returns the current beliefs of the agent.
        
        Returns:
            list: Current probabilities [P(True), P(False)]
        """
        return self.priors.copy()