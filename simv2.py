"""
Demonstration module for AgentV2 class showcasing informational cascades.

This module provides comprehensive simulations of agent behavior
with different k-levels and analysis of cascade formation.
"""

import random
from typing import List, Tuple, Dict, Any
from AgentV2 import AgentV2  # Assuming the class is in agentv2.py


class CascadeSimulator:
    """
    Simulates informational cascades with agents of varying k-levels.
    
    Provides methods to run simulations, analyze results, and generate
    comprehensive reports on agent behavior and cascade formation.
    """
    
    def __init__(self, true_state: bool = True, q_value: float = 0.7, 
                 num_agents: int = 10, random_seed: int = None):
        """
        Initialize the cascade simulator.
        
        Args:
            true_state: The actual state of the world (True/False)
            q_value: Probability that private signals are correct
            num_agents: Number of agents in the simulation
            random_seed: Random seed for reproducible results
        """
        self.true_state = true_state
        self.q_value = q_value
        self.num_agents = num_agents
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def generate_private_signals(self) -> List[bool]:
        """
        Generate private signals based on true state and q_value.
        
        Returns:
            List of private signals where each signal is correct with probability q_value
        """
        signals = []
        for _ in range(self.num_agents):
            if random.random() < self.q_value:
                signals.append(self.true_state)  # Correct signal
            else:
                signals.append(not self.true_state)  # Incorrect signal
        return signals
    
    def simulate_k_level_cascade(self, k_level: int) -> Dict[str, Any]:
        """
        Simulate a cascade with all agents at the specified k-level.
        
        Args:
            k_level: The k-level for all agents (0, 1, or 2)
            
        Returns:
            Dictionary containing simulation results and analysis
        """
        private_signals = self.generate_private_signals()
        agents = []
        decisions = []
        observations_history = []
        
        print(f"\n{'='*60}")
        print(f"K-LEVEL {k_level} CASCADE SIMULATION")
        print(f"{'='*60}")
        print(f"True State: {self.true_state}")
        print(f"Q Value: {self.q_value}")
        print(f"Number of Agents: {self.num_agents}")
        print(f"Private Signals: {private_signals}")
        print(f"{'-'*60}")
        
        for i in range(self.num_agents):
            # Create agent with current private signal and specified k-level
            agent = AgentV2(
                private_signal=private_signals[i],
                signal_strength=self.q_value,
                k_level=k_level
            )
            agents.append(agent)
            
            # Provide previous decisions as observations
            if decisions:
                agent.observe(decisions)
                observations_history.append(decisions.copy())
            
            # Agent makes decision
            decision = agent.action()
            decisions.append(decision)
            
            # Calculate statistics
            correct = decision == self.true_state
            signal_correct = private_signals[i] == self.true_state
            
            print(f"Agent {i:2d}: Signal={private_signals[i]}, "
                  f"Decision={decision}, Correct={correct}, "
                  f"Signal Correct={signal_correct}")
        
        # Analyze results
        analysis = self._analyze_cascade(decisions, private_signals)
        
        print(f"{'-'*60}")
        print(f"RESULTS: {analysis['correct_decisions']}/{self.num_agents} "
              f"correct decisions ({analysis['accuracy']:.1%})")
        print(f"Cascade detected at agent: {analysis['cascade_start']}")
        print(f"Cascade type: {analysis['cascade_type']}")
        
        return {
            'agents': agents,
            'decisions': decisions,
            'private_signals': private_signals,
            'analysis': analysis,
            'observations_history': observations_history
        }
    
    def simulate_mixed_k_levels(self, k_levels: List[int]) -> Dict[str, Any]:
        """
        Simulate a cascade with agents having different k-levels.
        
        Args:
            k_levels: List of k-levels for each agent
            
        Returns:
            Dictionary containing simulation results and analysis
        """
        if len(k_levels) != self.num_agents:
            raise ValueError(f"Number of k-levels ({len(k_levels)}) must match number of agents ({self.num_agents})")
        
        private_signals = self.generate_private_signals()
        agents = []
        decisions = []
        observations_history = []
        
        print(f"\n{'='*60}")
        print(f"MIXED K-LEVELS CASCADE SIMULATION")
        print(f"{'='*60}")
        print(f"True State: {self.true_state}")
        print(f"Q Value: {self.q_value}")
        print(f"Number of Agents: {self.num_agents}")
        print(f"K-Levels: {k_levels}")
        print(f"Private Signals: {private_signals}")
        print(f"{'-'*60}")
        
        for i in range(self.num_agents):
            # Create agent with current private signal and specified k-level
            agent = AgentV2(
                private_signal=private_signals[i],
                signal_strength=self.q_value,
                k_level=k_levels[i]
            )
            agents.append(agent)
            
            # Provide previous decisions as observations
            if decisions:
                agent.observe(decisions)
                observations_history.append(decisions.copy())
            
            # Agent makes decision
            decision = agent.action()
            decisions.append(decision)
            
            # Calculate statistics
            correct = decision == self.true_state
            signal_correct = private_signals[i] == self.true_state
            
            print(f"Agent {i:2d}: K-Level={k_levels[i]}, "
                  f"Signal={private_signals[i]}, Decision={decision}, "
                  f"Correct={correct}, Signal Correct={signal_correct}")
        
        # Analyze results
        analysis = self._analyze_cascade(decisions, private_signals)
        
        print(f"{'-'*60}")
        print(f"RESULTS: {analysis['correct_decisions']}/{self.num_agents} "
              f"correct decisions ({analysis['accuracy']:.1%})")
        print(f"Cascade detected at agent: {analysis['cascade_start']}")
        print(f"Cascade type: {analysis['cascade_type']}")
        
        # Additional analysis by k-level
        k_level_analysis = self._analyze_by_k_level(agents, decisions, k_levels)
        print(f"\nK-Level Performance:")
        for k_level, stats in k_level_analysis.items():
            print(f"  K-Level {k_level}: {stats['correct']}/{stats['total']} "
                  f"correct ({stats['accuracy']:.1%})")
        
        return {
            'agents': agents,
            'decisions': decisions,
            'private_signals': private_signals,
            'k_levels': k_levels,
            'analysis': analysis,
            'k_level_analysis': k_level_analysis,
            'observations_history': observations_history
        }
    
    def _analyze_cascade(self, decisions: List[bool], private_signals: List[bool]) -> Dict[str, Any]:
        """
        Analyze cascade formation and performance metrics.
        
        Args:
            decisions: List of agent decisions
            private_signals: List of private signals
            
        Returns:
            Dictionary with cascade analysis
        """
        correct_decisions = sum(1 for d in decisions if d == self.true_state)
        accuracy = correct_decisions / len(decisions) if decisions else 0
        
        # Detect cascade start (when 3 consecutive same decisions occur)
        cascade_start = -1
        for i in range(2, len(decisions)):
            if decisions[i] == decisions[i-1] == decisions[i-2]:
                cascade_start = i - 2
                break
        
        # Determine cascade type
        if cascade_start != -1:
            cascade_decision = decisions[cascade_start]
            cascade_type = "CORRECT" if cascade_decision == self.true_state else "INCORRECT"
        else:
            cascade_type = "NO CASCADE"
        
        # Calculate signal utilization (how often agents followed their private signals)
        signal_followed = sum(1 for i, (d, s) in enumerate(zip(decisions, private_signals)) 
                             if d == s)
        signal_utilization = signal_followed / len(decisions) if decisions else 0
        
        return {
            'correct_decisions': correct_decisions,
            'accuracy': accuracy,
            'cascade_start': cascade_start,
            'cascade_type': cascade_type,
            'signal_utilization': signal_utilization,
            'total_agents': len(decisions)
        }
    
    def _analyze_by_k_level(self, agents: List[AgentV2], decisions: List[bool], 
                           k_levels: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Analyze performance by k-level.
        
        Args:
            agents: List of AgentV2 instances
            decisions: List of agent decisions
            k_levels: List of k-levels for each agent
            
        Returns:
            Dictionary with performance statistics by k-level
        """
        analysis = {}
        
        for i, (agent, decision, k_level) in enumerate(zip(agents, decisions, k_levels)):
            if k_level not in analysis:
                analysis[k_level] = {'total': 0, 'correct': 0}
            
            analysis[k_level]['total'] += 1
            if decision == self.true_state:
                analysis[k_level]['correct'] += 1
        
        # Calculate accuracy for each k-level
        for k_level in analysis:
            analysis[k_level]['accuracy'] = (
                analysis[k_level]['correct'] / analysis[k_level]['total']
            )
        
        return analysis
    
    def run_comprehensive_demo(self):
        """
        Run a comprehensive demonstration with multiple simulation scenarios.
        """
        print("INFORMATIONAL CASCADE DEMONSTRATION")
        print("=" * 60)
        
        # Scenario 1: All agents at k-level 1
        results_k1 = self.simulate_k_level_cascade(k_level=1)
        
        # Scenario 2: All agents at k-level 2  
        results_k2 = self.simulate_k_level_cascade(k_level=2)
        
        # Scenario 3: Mixed k-levels
        mixed_k_levels = [0, 1, 2, 1, 2, 0, 1, 2, 1, 0][:self.num_agents]
        results_mixed = self.simulate_mixed_k_levels(mixed_k_levels)
        
        # Comparative analysis
        self._print_comparative_analysis([results_k1, results_k2, results_mixed])
    
    def _print_comparative_analysis(self, results_list: List[Dict[str, Any]]):
        """
        Print comparative analysis of multiple simulation runs.
        
        Args:
            results_list: List of simulation results dictionaries
        """
        print(f"\n{'='*60}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*60}")
        
        scenarios = ["K-Level 1", "K-Level 2", "Mixed K-Levels"]
        
        print(f"{'Scenario':<15} {'Accuracy':<10} {'Cascade Start':<15} {'Cascade Type':<15}")
        print(f"{'-'*60}")
        
        for scenario, results in zip(scenarios, results_list):
            analysis = results['analysis']
            print(f"{scenario:<15} {analysis['accuracy']:<10.1%} "
                  f"{analysis['cascade_start']:<15} {analysis['cascade_type']:<15}")


def main():
    """
    Main demonstration function with configurable parameters.
    """
    # Configuration
    config = {
        'true_state': True,
        'q_value': 0.6,  # Signal accuracy
        'num_agents': 10,
        'random_seed': 42  # For reproducible results
    }
    
    # Create simulator instance
    simulator = CascadeSimulator(**config)
    
    # Run comprehensive demonstration
    simulator.run_comprehensive_demo()
    
    # Additional focused simulations
    print(f"\n{'='*60}")
    print("ADDITIONAL FOCUSED SIMULATIONS")
    print(f"{'='*60}")
    
    # Test with different q values
    for q in [0.55, 0.7, 0.8]:
        print(f"\nTesting with q = {q}:")
        test_simulator = CascadeSimulator(
            true_state=True,
            q_value=q,
            num_agents=8,
            random_seed=42
        )
        test_simulator.simulate_k_level_cascade(k_level=1)


if __name__ == "__main__":
    main()