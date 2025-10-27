import numpy as np
import matplotlib.pyplot as plt
from AgentClass import BayesianAgent

def demonstrate_k_level_reasoning():
    """Demonstrate different K-level agents with the same information"""
    print("=== Demonstrating K-level Reasoning ===")
    
    # Same scenario for all agents
    true_state = True
    private_signal = True  # Correct signal
    observations = [True, False, True, True, False]  # Mixed signals (3 True, 2 False)
    
    print(f"True state: {true_state}")
    print(f"Private signal: {private_signal}")
    print(f"Observations: {observations}")
    print()
    
    # Test different K-level agents
    results = []
    for k in range(4):
        print(f"--- Testing K-level {k} Agent ---")
        agent = BayesianAgent(private_signal, k)
        agent.observation(observations)
        decision = agent.action()
        beliefs = agent.get_beliefs()
        accuracy = "CORRECT" if decision == true_state else "WRONG"
        
        results.append({
            'k_level': k,
            'decision': decision,
            'belief_true': beliefs[0],
            'accuracy': accuracy
        })
        
        print(f"K-level {k} agent decision: {decision} ({accuracy})")
        print(f"Final belief - P(True): {beliefs[0]:.3f}, P(False): {beliefs[1]:.3f}")
        print("-" * 40)
        print()
    
    return results

def simulate_information_cascade():
    """Simulate how different K-level agents might create information cascades"""
    print("\n=== Simulating Information Cascades ===")
    
    true_state = True
    # First few agents get correct signals, then noisy signals
    early_signals = [True, True, True]  # Strong early consensus
    late_signals = [False, False, True, False]  # Contradictory later signals
    
    print(f"True state: {true_state}")
    print(f"Early signals: {early_signals}")
    print(f"Late signals: {late_signals}")
    print()
    
    cascade_results = []
    for k in [0, 1, 2]:
        print(f"K-level {k} agent sequence:")
        agent = BayesianAgent(True, k)  # Correct private signal
        
        # Process early signals
        agent.observation(early_signals)
        early_decision = agent.action()
        early_beliefs = agent.get_beliefs()
        
        # Process late signals  
        agent.observation(late_signals)
        final_decision = agent.action()
        final_beliefs = agent.get_beliefs()
        
        early_acc = "CORRECT" if early_decision == true_state else "WRONG"
        final_acc = "CORRECT" if final_decision == true_state else "WRONG"
        cascade_effect = early_decision == final_decision
        
        cascade_results.append({
            'k_level': k,
            'early_decision': early_decision,
            'final_decision': final_decision,
            'early_belief_true': early_beliefs[0],
            'final_belief_true': final_beliefs[0],
            'cascade_effect': cascade_effect
        })
        
        print(f"Early decision: {early_decision} ({early_acc}) - Belief: {early_beliefs[0]:.3f}")
        print(f"Final decision: {final_decision} ({final_acc}) - Belief: {final_beliefs[0]:.3f}")
        print(f"Cascade effect: {'YES' if cascade_effect else 'NO'}")
        print("-" * 40)
    
    return cascade_results

def sequential_cascade_simulation():
    """Simulate a sequential cascade where agents observe previous agents' decisions"""
    print("\n=== Sequential Cascade Simulation ===")
    
    true_state = True
    num_agents = 10
    signal_accuracy = 0.7  # Probability of receiving correct private signal
    
    print(f"True state: {true_state}")
    print(f"Number of agents: {num_agents}")
    print(f"Signal accuracy: {signal_accuracy}")
    print()
    
    # Generate private signals for each agent
    private_signals = []
    for i in range(num_agents):
        if np.random.random() < signal_accuracy:
            private_signals.append(true_state)  # Correct signal
        else:
            private_signals.append(not true_state)  # Incorrect signal
    
    print(f"Private signals: {private_signals}")
    print(f"Correct signals: {sum(1 for s in private_signals if s == true_state)}/{num_agents}")
    print()
    
    # Simulate sequential decision making for different K-levels
    k_levels = [0, 1, 2]
    sequential_results = {k: {'decisions': [], 'beliefs': []} for k in k_levels}
    
    for k in k_levels:
        print(f"--- Sequential decisions for K-level {k} ---")
        decisions_so_far = []
        
        for i in range(num_agents):
            agent = BayesianAgent(private_signals[i], k)
            
            # Agent observes all previous decisions
            if decisions_so_far:
                agent.observation(decisions_so_far)
            
            decision = agent.action()
            beliefs = agent.get_beliefs()
            decisions_so_far.append(decision)
            
            sequential_results[k]['decisions'].append(decision)
            sequential_results[k]['beliefs'].append(beliefs[0])
            
            accuracy = "✓" if decision == true_state else "✗"
            print(f"Agent {i+1}: Signal={private_signals[i]}, Decision={decision} {accuracy}, Belief={beliefs[0]:.3f}")
        
        correct_decisions = sum(1 for d in sequential_results[k]['decisions'] if d == true_state)
        print(f"Total correct: {correct_decisions}/{num_agents}")
        print()
    
    return sequential_results, private_signals, true_state

def plot_results(k_level_results, cascade_results, sequential_results, private_signals, true_state):
    """Plot the simulation results"""
    
    # Plot 1: K-level comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    k_levels = [r['k_level'] for r in k_level_results]
    beliefs = [r['belief_true'] for r in k_level_results]
    colors = ['green' if r['decision'] == true_state else 'red' for r in k_level_results]
    
    bars = plt.bar(k_levels, beliefs, color=colors, alpha=0.7)
    plt.title('Belief in True State by K-level')
    plt.xlabel('K-level')
    plt.ylabel('P(True)')
    plt.xticks(k_levels)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, belief in zip(bars, beliefs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{belief:.3f}', ha='center', va='bottom')
    
    # Plot 2: Cascade effects
    plt.subplot(2, 2, 2)
    k_levels_cascade = [r['k_level'] for r in cascade_results]
    early_beliefs = [r['early_belief_true'] for r in cascade_results]
    final_beliefs = [r['final_belief_true'] for r in cascade_results]
    
    x = np.arange(len(k_levels_cascade))
    width = 0.35
    
    plt.bar(x - width/2, early_beliefs, width, label='After Early Signals', alpha=0.7)
    plt.bar(x + width/2, final_beliefs, width, label='After All Signals', alpha=0.7)
    
    plt.title('Cascade Effect on Beliefs')
    plt.xlabel('K-level')
    plt.ylabel('P(True)')
    plt.xticks(x, k_levels_cascade)
    plt.legend()
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Sequential decisions
    plt.subplot(2, 2, 3)
    for k in [0, 1, 2]:
        beliefs_evolution = sequential_results[k]['beliefs']
        decisions = sequential_results[k]['decisions']
        plt.plot(range(1, len(beliefs_evolution) + 1), beliefs_evolution, 
                marker='o', label=f'K-level {k}', linewidth=2)
    
    plt.title('Belief Evolution in Sequential Cascade')
    plt.xlabel('Agent Sequence')
    plt.ylabel('P(True)')
    plt.legend()
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Decision accuracy by position
    plt.subplot(2, 2, 4)
    positions = range(1, 11)
    
    for k in [0, 1, 2]:
        decisions = sequential_results[k]['decisions']
        accuracy_by_position = []
        for i in range(len(decisions)):
            accuracy = sum(1 for d in decisions[:i+1] if d == true_state) / (i + 1)
            accuracy_by_position.append(accuracy)
        
        plt.plot(positions, accuracy_by_position, marker='s', label=f'K-level {k}', linewidth=2)
    
    plt.title('Cumulative Accuracy by Position')
    plt.xlabel('Number of Agents')
    plt.ylabel('Cumulative Accuracy')
    plt.legend()
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run all simulations"""
    print("Bayesian Agent Cascade Simulation")
    print("=" * 50)
    
    # Run all simulations
    k_level_results = demonstrate_k_level_reasoning()
    cascade_results = simulate_information_cascade()
    sequential_results, private_signals, true_state = sequential_cascade_simulation()
    
    # Plot results
    plot_results(k_level_results, cascade_results, sequential_results, private_signals, true_state)
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    for k in [0, 1, 2]:
        decisions = sequential_results[k]['decisions']
        correct = sum(1 for d in decisions if d == true_state)
        accuracy = correct / len(decisions)
        final_belief = sequential_results[k]['beliefs'][-1]
        
        print(f"K-level {k}:")
        print(f"  Final accuracy: {accuracy:.1%} ({correct}/{len(decisions)})")
        print(f"  Final belief in true state: {final_belief:.3f}")
        print(f"  Cascade strength: {sum(sequential_results[k]['beliefs']) / len(sequential_results[k]['beliefs']):.3f}")

if __name__ == "__main__":
    main()