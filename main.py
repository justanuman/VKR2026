import numpy as np
import matplotlib.pyplot as plt
from AgentClass import BayesianAgent

def main():
    # Create agents with different K-levels
    agent0 = BayesianAgent(0.7, k_level=0)
    agent1 = BayesianAgent(0.7, k_level=1)
    agent2 = BayesianAgent(0.7, k_level=2)
    
    print("Initial agents:")
    print(agent0)
    print(agent1)
    print(agent2)
    print()
    
    # Set some observations for the higher-level agents
    observations = [True, False, True, True, False]
    agent1.observation(observations)
    agent2.observation(observations)
    
    print("After setting observations:")
    print(f"Agent 0 action: {agent0.action()}")
    print(f"Agent 1 action: {agent1.action()}")
    print(f"Agent 2 action: {agent2.action()}")
    print()
    
    # Test with different q values
    print("Testing with different q values:")
    for q in [0.3, 0.5, 0.8]:
        agent = BayesianAgent(q, k_level=0)
        print(f"Agent with q={q}, private_signal={agent.private_signal}, action: {agent.action()}")

if __name__ == "__main__":
    main()