import random
from AgentClass import Agent

def generate_private_signals(true_state, q, num_agents):
    """
    Generate private signals for agents based on true state and q.
    
    Args:
        true_state (bool): The true state of the world
        q (float): Probability that private signal is correct
        num_agents (int): Number of agents to generate signals for
    
    Returns:
        list: List of private signals (booleans)
    """
    signals = []
    for _ in range(num_agents):
        if random.random() < q:
            # Signal is correct
            signals.append(true_state)
        else:
            # Signal is incorrect
            signals.append(not true_state)
    return signals

def simulate_cascade(true_state, q, k_level, num_agents=10):
    """
    Simulate an informational cascade with sequential decision making.
    
    Args:
        true_state (bool): The true state of the world
        q (float): Probability that private signal is correct
        k_level (int): K-level for all agents
        num_agents (int): Number of agents in the simulation
    
    Returns:
        tuple: (agents, decisions, correct_count)
    """
    # Generate private signals for all agents
    private_signals = generate_private_signals(true_state, q, num_agents)
    
    # Create agents
    agents = []
    for i in range(num_agents):
        agent = Agent(q_value=q, k_level=k_level, private_signal=private_signals[i])
        agents.append(agent)
    
    # Simulate sequential decision making
    decisions = []
    correct_count = 0
    
    print(f"True state: {true_state}")
    print(f"Private signals: {private_signals}")
    print(f"K-level: {k_level}, q: {q}")
    print("-" * 50)
    
    for i, agent in enumerate(agents):
        # Give previous decisions as observations to current agent
        if i > 0:
            agent.observation(decisions)
        
        # Agent makes decision
        decision = agent.action()
        decisions.append(decision)
        
        # Check if decision is correct
        is_correct = (decision == true_state)
        if is_correct:
            correct_count += 1
        
        print(f"Agent {i}: private_signal={agent.private_signal}, decision={decision}, correct={is_correct}")
    
    print("-" * 50)
    print(f"Final results: {correct_count}/{num_agents} correct decisions")
    print(f"Decisions: {decisions}")
    
    return agents, decisions, correct_count

def compare_k_levels(true_state=True, q=0.6, num_agents=10):
    """
    Compare different K-levels in the same simulation setup.
    """
    print("=" * 60)
    print("COMPARING DIFFERENT K-LEVELS")
    print("=" * 60)
    
    results = {}
    
    for k_level in [0, 1, 2]:
        print(f"\nK-Level {k_level} Simulation:")
        print("-" * 40)
        
        agents, decisions, correct_count = simulate_cascade(
            true_state=true_state, 
            q=q, 
            k_level=k_level, 
            num_agents=num_agents
        )
        
        results[k_level] = {
            'decisions': decisions,
            'correct_count': correct_count,
            'cascade_start': detect_cascade_start(decisions)
        }
    
    return results

def detect_cascade_start(decisions):
    """
    Detect when an informational cascade starts (when decisions become uniform).
    
    Args:
        decisions (list): List of boolean decisions
    
    Returns:
        int: Index where cascade starts, or -1 if no clear cascade
    """
    if len(decisions) < 3:
        return -1
    
    for i in range(2, len(decisions)):
        # Check if last 3 decisions are the same
        if decisions[i] == decisions[i-1] == decisions[i-2]:
            return i-2
    
    return -1

def analyze_cascade_pattern(decisions, true_state):
    """
    Analyze the pattern of decisions in a cascade.
    """
    if not decisions:
        return "No decisions"
    
    correct_decision = true_state
    correct_count = sum(1 for d in decisions if d == correct_decision)
    accuracy = correct_count / len(decisions)
    
    # Check for cascade
    cascade_start = detect_cascade_start(decisions)
    
    analysis = {
        'total_agents': len(decisions),
        'correct_decisions': correct_count,
        'accuracy': accuracy,
        'cascade_start': cascade_start,
        'cascade_occurred': cascade_start != -1,
        'final_decision': decisions[-1] if decisions else None,
        'correct_final_decision': decisions[-1] == correct_decision if decisions else None
    }
    
    return analysis

def main():
    """
    Main demonstration function.
    """
    # Set simulation parameters
    TRUE_STATE = True
    Q = 0.6  # Probability signal is correct
    NUM_AGENTS = 15
    
    print("INFORMATIONAL CASCADE SIMULATION")
    print("=" * 50)
    
    # Run simulation for each K-level
    results = compare_k_levels(
        true_state=TRUE_STATE, 
        q=Q, 
        num_agents=NUM_AGENTS
    )
    
    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    for k_level, result in results.items():
        analysis = analyze_cascade_pattern(result['decisions'], TRUE_STATE)
        
        print(f"\nK-Level {k_level}:")
        print(f"  Accuracy: {analysis['accuracy']:.1%} ({analysis['correct_decisions']}/{analysis['total_agents']})")
        print(f"  Cascade started at agent: {analysis['cascade_start']}" if analysis['cascade_occurred'] else "  No clear cascade detected")
        print(f"  Final decision: {analysis['final_decision']} (correct: {analysis['correct_final_decision']})")
    
    # Run multiple simulations to show variability
    print("\n" + "=" * 60)
    print("MULTIPLE SIMULATION RUNS (K-Level 1)")
    print("=" * 60)
    
    num_simulations = 3
    for i in range(num_simulations):
        print(f"\nSimulation Run {i+1}:")
        simulate_cascade(true_state=TRUE_STATE, q=Q, k_level=1, num_agents=NUM_AGENTS)

if __name__ == "__main__":
    main()