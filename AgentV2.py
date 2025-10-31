'''
fields:
private signal
observation 
decision 
Prior (tuple)
Signal strength 
k level 

'''

class AgentV2:
    """
    An agent that makes decisions based on private signals, observations, and k-level reasoning.
    
    The agent uses Bayesian updating to process information and make decisions,
    with different levels of strategic reasoning based on the k-level.
    """
    
    def __init__(self, private_signal: bool, signal_strength: float, k_level: int, prior: tuple = (0.5, 0.5)):
        """
        Initialize the AgentV2 with basic properties.
        
        Args:
            private_signal: The agent's private signal (True/False)
            signal_strength: Probability that private signal is correct (q value)
            k_level: Strategic reasoning level (0, 1, or 2)
            prior: Prior probabilities for Bayesian updating (P(True), P(False))
        
        Raises:
            ValueError: If signal_strength is not in [0,1] or k_level is not 0,1,2
        """
        if not 0 <= signal_strength <= 1:
            raise ValueError("signal_strength must be between 0 and 1")
        if k_level not in [0, 1, 2]:
            raise ValueError("k_level must be 0, 1, or 2")
        
        self.private_signal = private_signal
        self.signal_strength = signal_strength
        self.k_level = k_level
        self.prior = prior
        self.observations = []
        self.decision = None
    
    def observe(self, observations: list) -> None:
        """
        Set the agent's observations from previous agents' decisions.
        
        Args:
            observations: List of boolean decisions from previous agents
        """
        self.observations = observations.copy()
    
    def aux_lvl0(self) -> bool:
        """
        Level 0 reasoning: Direct use of private signal.
        
        Returns:
            The private signal if signal_strength >= 0.5, otherwise its negation
        """
        if self.signal_strength >= 0.5:
            return self.private_signal
        else:
            return not self.private_signal
    
    def aux_lvl1(self) -> bool:
        """
        Level 1 reasoning: Bayesian updating with observations and private signal.
        
        Returns:
            True if posterior probability > 0.5, False otherwise
        """
        posterior_prob = self._bayesian_update(self.observations, self.private_signal)
        return posterior_prob > 0.5
    
    def aux_lvl2(self) -> bool:
        """
        Level 2 reasoning: Infer private signals of previous agents and update.
        
        Returns:
            Decision based on inferred signals and Bayesian updating
        """
        if not self.observations:
            return self.aux_lvl1()
        
        # Count inferred private signals from observations
        true_signals, false_signals = self._infer_private_signals()
        
        # Use inferred signals for Bayesian updating
        posterior_prob = self._bayesian_update_with_inferred_signals(
            true_signals, false_signals, self.private_signal
        )
        
        return posterior_prob > 0.5
    
    def action(self) -> bool:
        """
        Execute the agent's decision based on its k-level.
        
        Returns:
            The agent's final decision
        """
        if self.k_level == 0:
            self.decision = self.aux_lvl0()
        elif self.k_level == 1:
            self.decision = self.aux_lvl1()
        elif self.k_level == 2:
            self.decision = self.aux_lvl2()
        
        return self.decision
    
    def _bayesian_update(self, observations: list, private_signal: bool) -> float:
        """
        Perform Bayesian updating given observations and private signal.
        
        Args:
            observations: List of boolean observations
            private_signal: The agent's private signal
            
        Returns:
            Posterior probability that true state is True
        """
        prob_true = self.prior[0]  # Start with prior P(True)
        
        # Update with each observation
        for observation in observations:
            prob_true = self._update_belief(prob_true, observation, is_observation=True)
        
        # Update with private signal
        prob_true = self._update_belief(prob_true, private_signal, is_observation=False)
        
        return prob_true
    
    def _update_belief(self, current_prob: float, signal: bool, is_observation: bool = True) -> float:
        """
        Update belief using Bayes' rule.
        
        Args:
            current_prob: Current probability of true state being True
            signal: The signal to incorporate (observation or private signal)
            is_observation: Whether this is an observation (vs private signal)
            
        Returns:
            Updated probability
        """
        # Use signal_strength for private signals, assumed 0.6 for observations
        q = self.signal_strength if not is_observation else 0.6
        
        if signal:  # Signal is True
            likelihood_true = q
            likelihood_false = 1 - q
        else:  # Signal is False
            likelihood_true = 1 - q
            likelihood_false = q
        
        numerator = likelihood_true * current_prob
        denominator = numerator + likelihood_false * (1 - current_prob)
        
        # Avoid division by zero
        if denominator == 0:
            return current_prob
        
        return numerator / denominator
    
    def _infer_private_signals(self) -> tuple:
        """
        Infer private signals from observations using level 1 reasoning.
        
        Returns:
            Tuple of (true_signals_count, false_signals_count)
        """
        true_count = 0
        false_count = 0
        
        for i, observation in enumerate(self.observations):
            # Use all previous observations as context for this agent
            previous_observations = self.observations[:i]
            
            # Calculate what this agent would decide with True vs False private signal
            decision_if_true = self._simulate_level1_agent(previous_observations, True)
            decision_if_false = self._simulate_level1_agent(previous_observations, False)
            
            # Infer private signal based on which is more consistent with the actual observation
            if observation == decision_if_true:
                true_count += 1
            elif observation == decision_if_false:
                false_count += 1
            else:
                # Default to True if probabilities are equal or no clear match
                true_count += 1
        
        return true_count, false_count
    
    def _simulate_level1_agent(self, observations: list, private_signal: bool) -> bool:
        """
        Simulate what a level 1 agent would decide given observations and private signal.
        
        Args:
            observations: Observations available to the simulated agent
            private_signal: Private signal of the simulated agent
            
        Returns:
            Decision of the simulated level 1 agent
        """
        posterior_prob = self._bayesian_update(observations, private_signal)
        return posterior_prob > 0.5
    
    def _bayesian_update_with_inferred_signals(self, true_count: int, false_count: int, 
                                             private_signal: bool) -> float:
        """
        Perform Bayesian updating using inferred signals and private signal.
        
        Args:
            true_count: Number of inferred True signals
            false_count: Number of inferred False signals
            private_signal: Agent's own private signal
            
        Returns:
            Posterior probability that true state is True
        """
        prob_true = self.prior[0]
        
        # Update with each inferred True signal
        for _ in range(true_count):
            prob_true = self._update_belief(prob_true, True, is_observation=True)
        
        # Update with each inferred False signal  
        for _ in range(false_count):
            prob_true = self._update_belief(prob_true, False, is_observation=True)
        
        # Update with own private signal
        prob_true = self._update_belief(prob_true, private_signal, is_observation=False)
        
        return prob_true
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return (f"AgentV2(signal={self.private_signal}, "
                f"strength={self.signal_strength}, "
                f"k_level={self.k_level}, "
                f"decision={self.decision})")