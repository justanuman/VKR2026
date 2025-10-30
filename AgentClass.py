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
    

    def infer_private_signals_count(self):
        """
        Infer private signals by counting consistency with observed actions.
        
        For each observation, checks if the action is consistent with having 
        a True or False private signal given previous observations.
        
        Args:
            observations (list): List of boolean actions/decisions
            q_agent (float): Agent's signal accuracy 
            q_other (float): Assumed accuracy of other agents
        
        Returns:
            tuple: (inferred_signal, true_count, false_count, details)
        """
        observations=self.observations
        q_agent=0.6
        q_other=0.6
        true_count = 0
        false_count = 0
        inference_details = []
        print("HEREEEEEEEEEE",self.observations)
        for i in range(len(self.observations)):
            if i == 0:
                # First agent - no previous observations
                # Check what private signal would lead to this action
                current_action = observations[i]
                
                # For first agent with no observations:
                # If q >= 0.5: action = private_signal
                # If q < 0.5: action = not private_signal
                if q_agent >= 0.5:
                    # Action matches private signal
                    inferred_signal = current_action
                else:
                    # Action is opposite of private signal  
                    inferred_signal = not current_action
                
                if inferred_signal:
                    true_count += 1
                else:
                    false_count += 1
                    
                inference_details.append({
                    'agent_index': i,
                    'action': current_action,
                    'inferred_signal': inferred_signal,
                    'previous_obs_count': 0,
                    'method': 'first_agent'
                })
                
            else:
                # Subsequent agents - use previous observations
                previous_obs = observations[:i]  # Observations before current one
                current_action = observations[i]
                
                # Calculate belief after previous observations
                belief_after_prev = self.calculate_belief_after_observations(previous_obs, q_other)
                
                # Calculate what action would be taken for each possible private signal
                action_if_true = self.calculate_expected_action(belief_after_prev, True, q_agent)
                action_if_false = self.calculate_expected_action(belief_after_prev, False, q_agent)
                
                # Count consistency
                if current_action == action_if_true:
                    true_count += 1
                    inferred_signal = True
                else:
                    inferred_signal = False  # Default if not consistent with True
                    
                if current_action == action_if_false:
                    false_count += 1
                    # If also consistent with False, we already counted True above
                    # This handles cases where both signals lead to same action
                
                inference_details.append({
                    'agent_index': i,
                    'action': current_action,
                    'inferred_signal': inferred_signal,
                    'previous_obs_count': len(previous_obs),
                    'belief_after_prev': belief_after_prev,
                    'action_if_true': action_if_true,
                    'action_if_false': action_if_false,
                    'consistent_with_true': (current_action == action_if_true),
                    'consistent_with_false': (current_action == action_if_false)
                })
        
        # Determine overall inferred signal
        if true_count > false_count:
            overall_signal = True
        elif false_count > true_count:
            overall_signal = False
        else:
            overall_signal = None  # Tie
        
        return overall_signal, true_count, false_count, inference_details


    def calculate_belief_after_observations(self,observations, q_other=0.6):
        """
        Calculate Bayesian belief after seeing a list of observations.
        """
        belief = 0.5  # Uniform prior
        
        for obs in observations:
            if obs:  # Observation is True
                likelihood_true = q_other
                likelihood_false = 1 - q_other
            else:    # Observation is False  
                likelihood_true = 1 - q_other
                likelihood_false = q_other
            
            numerator = likelihood_true * belief
            denominator = numerator + likelihood_false * (1 - belief)
            
            if denominator > 0:
                belief = numerator / denominator
            # If denominator is 0, keep current belief (edge case)
        
        return belief


    def calculate_expected_action(self,belief_after_observations, private_signal, q_agent):
        """
        Calculate what action a Bayesian agent would take given:
        - belief after observations
        - private signal
        - their q value
        """
        # Update belief with private signal
        if private_signal:
            likelihood_true = q_agent
            likelihood_false = 1 - q_agent
        else:
            likelihood_true = 1 - q_agent
            likelihood_false = q_agent
        
        belief = belief_after_observations
        numerator = likelihood_true * belief
        denominator = numerator + likelihood_false * (1 - belief)
        
        if denominator > 0:
            final_belief = numerator / denominator
        else:
            final_belief = belief
        
        # Decision rule: action = True if final_belief > 0.5, else False
        return final_belief > 0.5


    # Alternative simpler version that just counts direct consistency
    def infer_private_signals_simple(observations, q_agent=0.7, q_other=0.6):
        """
        Simplified version that only counts when action uniquely determines signal.
        """
        true_count = 0
        false_count = 0
        
        for i in range(len(observations)):
            if i == 0:
                # First agent logic
                if q_agent >= 0.5:
                    inferred_signal = observations[i]
                else:
                    inferred_signal = not observations[i]
                    
                if inferred_signal:
                    true_count += 1
                else:
                    false_count += 1
            else:
                previous_obs = observations[:i]
                current_action = observations[i]
                
                belief_after_prev = calculate_belief_after_observations(previous_obs, q_other)
                
                action_if_true = calculate_expected_action(belief_after_prev, True, q_agent)
                action_if_false = calculate_expected_action(belief_after_prev, False, q_agent)
                
                # Only count if action uniquely determines the signal
                if action_if_true and not action_if_false:
                    if current_action:  # Action is True
                        true_count += 1
                elif action_if_false and not action_if_true:
                    if not current_action:  # Action is False  
                        false_count += 1
                # If both signals lead to same action, we can't infer - don't count
                # If action doesn't match expected for either signal, don't count
        
        if true_count > false_count:
            return True, true_count, false_count
        elif false_count > true_count:
            return False, true_count, false_count
        else:
            return None, true_count, false_count


        
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
    def action(self,observations):
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
            print("THEREEEEEEEEEEEE",self.observations)
            overall_signal, true_count, false_count, details = self.infer_private_signals_count()
            return overall_signal
        
        else:
            raise ValueError(f"Unsupported K-level: {self.k_level}")
    def __str__(self):
        """String representation of the agent."""
        return f"Agent(q={self.q}, k_level={self.k_level}, private_signal={self.private_signal})"