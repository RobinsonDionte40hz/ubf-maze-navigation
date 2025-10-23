"""
Universal Behavioral Framework - Test Scenarios

Implements the three test scenarios: solo runs, experienced+new group, 
and all-new group with comprehensive data logging and analysis.
"""

import json
import time
import random
import sys
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import Agent
from simulation.maze_environment import MazeEnvironment
from core.consciousness_update import create_maze_exploration_event, create_goal_completion_event


class UBFTestScenarios:
    """
    Manages the three test scenarios for the UBF system:
    1. Solo Phase: 1 agent, 3 runs (reset position, keep coords/memories)
    2. Group 1: Experienced + 2 new agents, 1 run (share avg coords post-run)
    3. Group 2: 3 new agents, 1 run
    """
    
    def __init__(self, maze_size: Tuple[int, int] = (10, 10), max_steps: int = 500):
        """
        Initialize test scenarios.
        
        Args:
            maze_size: (width, height) of maze
            max_steps: Maximum steps per run
        """
        self.maze_width, self.maze_height = maze_size
        self.max_steps = max_steps
        self.results = {
            'solo_phase': [],
            'group_1': [],
            'group_2': [],
            'metadata': {
                'maze_size': maze_size,
                'max_steps': max_steps,
                'timestamp': time.time(),
                'random_seed': None
            }
        }
        
    def run_all_scenarios(self, random_seed: int = None) -> Dict[str, Any]:
        """
        Run all three test scenarios and collect results.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Complete results dictionary
        """
        if random_seed is not None:
            random.seed(random_seed)
            self.results['metadata']['random_seed'] = random_seed
        
        print("Starting UBF Test Scenarios...")
        print(f"Maze size: {self.maze_width}x{self.maze_height}, Max steps: {self.max_steps}")
        
        # Scenario 1: Solo Phase
        print("\n=== Scenario 1: Solo Phase ===")
        solo_agent = self._create_agent("solo_agent")
        self.results['solo_phase'] = self._run_solo_phase(solo_agent)
        
        # Scenario 2: Experienced + 2 New
        print("\n=== Scenario 2: Experienced + 2 New ===")
        experienced_agent = solo_agent  # Use the experienced agent from solo phase
        new_agents = [self._create_agent(f"new_agent_{i}") for i in range(2)]
        self.results['group_1'] = self._run_group_scenario([experienced_agent] + new_agents, "experienced_group")
        
        # Scenario 3: 3 New Agents
        print("\n=== Scenario 3: 3 New Agents ===")
        fresh_agents = [self._create_agent(f"fresh_agent_{i}") for i in range(3)]
        self.results['group_2'] = self._run_group_scenario(fresh_agents, "naive_group")
        
        # Add final analysis
        self.results['analysis'] = self._analyze_results()
        
        print("\nAll scenarios completed!")
        return self.results
    
    def _create_agent(self, agent_id: str, experienced: bool = False) -> Agent:
        """
        Create agent with initial consciousness state.
        
        Args:
            agent_id: Unique identifier
            experienced: Whether to use experienced consciousness coordinates
            
        Returns:
            New Agent instance
        """
        if experienced:
            # Experienced agents start with slightly higher frequency and coherence
            freq = 8.0 + random.uniform(-0.5, 0.5)
            coh = 0.75 + random.uniform(-0.1, 0.1)
        else:
            # New agents start with default coordinates plus small variation
            freq = 7.5 + random.uniform(-1.0, 1.0)
            coh = 0.7 + random.uniform(-0.2, 0.2)
        
        return Agent(agent_id=agent_id, initial_frequency=freq, initial_coherence=coh)
    
    def _run_solo_phase(self, agent: Agent) -> List[Dict[str, Any]]:
        """
        Run solo phase: 3 runs with the same agent.
        Reset position between runs but keep consciousness and memories.
        """
        solo_results = []
        
        for run_num in range(3):
            print(f"\nSolo Run {run_num + 1}/3")
            
            # Create new maze for each run
            maze = MazeEnvironment(self.maze_width, self.maze_height)
            
            # Reset agent position but keep consciousness and memories
            agent.reset_for_new_run(keep_memories=True, keep_consciousness=True)
            
            # Add agent to maze
            maze.add_agent(agent)
            
            # Run simulation
            run_result = self._run_single_simulation(maze, [agent], f"solo_run_{run_num + 1}")
            run_result['run_number'] = run_num + 1
            run_result['agent_type'] = 'experienced' if run_num > 0 else 'initial'
            
            solo_results.append(run_result)
            
            # Print run summary
            self._print_run_summary(run_result, f"Solo Run {run_num + 1}")
        
        return solo_results
    
    def _run_group_scenario(self, agents: List[Agent], scenario_name: str) -> Dict[str, Any]:
        """
        Run group scenario with multiple agents.
        
        Args:
            agents: List of agents to run
            scenario_name: Name for logging
            
        Returns:
            Group scenario results
        """
        print(f"\nRunning {scenario_name} with {len(agents)} agents")
        
        # Create maze
        maze = MazeEnvironment(self.maze_width, self.maze_height)
        
        # Reset all agents
        for agent in agents:
            agent.reset_for_new_run(keep_memories=True, keep_consciousness=True)
            maze.add_agent(agent)
        
        # Run simulation
        group_result = self._run_single_simulation(maze, agents, scenario_name)
        
        # Share experience between agents (average consciousness coordinates)
        if len(agents) > 1:
            experience_sharing = self._share_group_experience(agents)
            group_result['experience_sharing'] = experience_sharing
        
        # Print group summary
        self._print_group_summary(group_result, scenario_name)
        
        return group_result
    
    def _run_single_simulation(self, maze: MazeEnvironment, agents: List[Agent], 
                             simulation_name: str) -> Dict[str, Any]:
        """
        Run a single simulation with given maze and agents.
        
        Args:
            maze: Maze environment
            agents: List of agents
            simulation_name: Name for logging
            
        Returns:
            Simulation results
        """
        start_time = time.time()
        step_logs = []
        
        # Initial state
        initial_states = {agent.agent_id: agent.get_status_summary() for agent in agents}
        
        # Run simulation steps
        while not maze._check_simulation_complete(agents):
            step_result = maze.step(agents)
            step_logs.append(step_result)
            
            # Optional: Print progress every 50 steps
            if step_result['step'] % 50 == 0:
                print(f"  Step {step_result['step']}: {len(maze.agents_finished)}/{len(agents)} finished")
        
        # Final state
        final_states = {agent.agent_id: agent.get_status_summary() for agent in agents}
        
        # Simulation metrics
        simulation_time = time.time() - start_time
        
        return {
            'simulation_name': simulation_name,
            'maze_info': maze.get_maze_summary(),
            'initial_states': initial_states,
            'final_states': final_states,
            'step_logs': step_logs,
            'simulation_metrics': {
                'total_steps': maze.step_count,
                'agents_completed': len(maze.agents_finished),
                'completion_rate': len(maze.agents_finished) / len(agents),
                'simulation_time': simulation_time,
                'steps_per_second': maze.step_count / simulation_time if simulation_time > 0 else 0
            },
            'agent_details': {agent.agent_id: agent.get_detailed_history() for agent in agents}
        }
    
    def _share_group_experience(self, agents: List[Agent]) -> Dict[str, Any]:
        """
        Share experience between agents by averaging consciousness coordinates.
        
        Args:
            agents: List of agents to share experience
            
        Returns:
            Experience sharing information
        """
        if len(agents) < 2:
            return {}
        
        # Calculate average consciousness coordinates
        avg_freq = sum(agent.consciousness.frequency for agent in agents) / len(agents)
        avg_coh = sum(agent.consciousness.coherence for agent in agents) / len(agents)
        
        # Store original coordinates
        original_coords = {
            agent.agent_id: (agent.consciousness.frequency, agent.consciousness.coherence)
            for agent in agents
        }
        
        # Move each agent 50% toward the average
        sharing_info = []
        for agent in agents:
            freq_delta = (avg_freq - agent.consciousness.frequency) * 0.5
            coh_delta = (avg_coh - agent.consciousness.coherence) * 0.5
            
            agent.consciousness.update_coordinates(freq_delta, coh_delta, noise_std=0.05)
            agent.behavioral_state = agent.behavioral_state.from_consciousness(agent.consciousness)
            
            sharing_info.append({
                'agent_id': agent.agent_id,
                'original_coords': original_coords[agent.agent_id],
                'new_coords': (agent.consciousness.frequency, agent.consciousness.coherence),
                'delta': (freq_delta, coh_delta)
            })
        
        return {
            'average_coordinates': (avg_freq, avg_coh),
            'agent_changes': sharing_info,
            'sharing_method': 'averaged_coordinates_50pct'
        }
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        Analyze results across all scenarios to identify learning patterns.
        
        Returns:
            Analysis summary
        """
        analysis = {
            'learning_progression': self._analyze_learning_progression(),
            'failure_adaptation': self._analyze_failure_adaptation(),
            'memory_patterns': self._analyze_memory_patterns(),
            'consciousness_evolution': self._analyze_consciousness_evolution(),
            'performance_comparison': self._analyze_performance_comparison()
        }
        
        return analysis
    
    def _analyze_learning_progression(self) -> Dict[str, Any]:
        """Analyze how performance improves across solo runs."""
        solo_runs = self.results['solo_phase']
        
        if len(solo_runs) < 2:
            return {}
        
        # Extract performance metrics
        completion_times = []
        success_rates = []
        step_counts = []
        
        for run in solo_runs:
            metrics = run['simulation_metrics']
            completion_times.append(metrics['total_steps'])
            success_rates.append(metrics['completion_rate'])
            step_counts.append(metrics['total_steps'])
        
        # Calculate improvement
        time_improvement = (completion_times[0] - completion_times[-1]) / completion_times[0] if completion_times[0] > 0 else 0
        
        return {
            'completion_times': completion_times,
            'success_rates': success_rates,
            'time_improvement_pct': time_improvement * 100,
            'learning_trend': 'improving' if time_improvement > 0.1 else 'stable' if abs(time_improvement) < 0.1 else 'declining'
        }
    
    def _analyze_failure_adaptation(self) -> Dict[str, Any]:
        """Analyze how agents adapt to failures."""
        all_agents = []
        
        # Collect all agent data
        for scenario in ['solo_phase', 'group_1', 'group_2']:
            scenario_data = self.results[scenario]
            if isinstance(scenario_data, list):
                for run in scenario_data:
                    all_agents.extend(run['agent_details'].values())
            else:
                all_agents.extend(scenario_data['agent_details'].values())
        
        failure_adaptations = []
        
        for agent_data in all_agents:
            history = agent_data['action_history']
            failure_responses = []
            
            # Look for failure -> creative response patterns
            for i in range(len(history) - 1):
                if history[i]['outcome'] == 'failure' or history[i]['outcome'] == 'collision':
                    next_action = history[i + 1]['action']
                    if next_action in ['creative_solution', 'investigate', 'turn_left', 'turn_right']:
                        failure_responses.append({
                            'failure_step': history[i]['step'],
                            'response_action': next_action,
                            'consciousness_before': history[i]['consciousness'],
                            'consciousness_after': history[i + 1]['consciousness']
                        })
            
            if failure_responses:
                failure_adaptations.append({
                    'agent_id': agent_data['agent_id'],
                    'failure_count': len(failure_responses),
                    'adaptation_types': [r['response_action'] for r in failure_responses],
                    'sample_adaptations': failure_responses[:3]  # First 3 for brevity
                })
        
        return {
            'agents_with_adaptations': len(failure_adaptations),
            'total_failure_adaptations': sum(fa['failure_count'] for fa in failure_adaptations),
            'common_adaptations': self._get_most_common([fa['adaptation_types'] for fa in failure_adaptations]),
            'adaptation_examples': failure_adaptations[:2]  # Sample examples
        }
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory formation and usage patterns."""
        memory_stats = []
        
        # Collect memory statistics from all agents
        for scenario in ['solo_phase', 'group_1', 'group_2']:
            scenario_data = self.results[scenario]
            if isinstance(scenario_data, list):
                for run in scenario_data:
                    for agent_data in run['agent_details'].values():
                        memory_stats.append(agent_data['memory_details']['stats'])
            else:
                for agent_data in scenario_data['agent_details'].values():
                    memory_stats.append(agent_data['memory_details']['stats'])
        
        if not memory_stats:
            return {}
        
        # Aggregate statistics
        total_memories = sum(stat['total_memories'] for stat in memory_stats if stat['total_memories'] > 0)
        memory_agents = [s for s in memory_stats if s['total_memories'] > 0]
        
        if memory_agents:
            avg_significance = sum(stat['avg_significance'] for stat in memory_agents) / len(memory_agents)
        else:
            avg_significance = 0.0
        
        # Memory type distribution
        all_by_type = {}
        for stat in memory_stats:
            for mem_type, count in stat.get('by_type', {}).items():
                all_by_type[mem_type] = all_by_type.get(mem_type, 0) + count
        
        return {
            'total_memories_formed': total_memories,
            'average_significance': avg_significance,
            'memory_types_distribution': all_by_type,
            'agents_with_memories': len([s for s in memory_stats if s['total_memories'] > 0])
        }
    
    def _analyze_consciousness_evolution(self) -> Dict[str, Any]:
        """Analyze how consciousness coordinates evolve over time."""
        consciousness_trajectories = []
        
        # Extract consciousness trajectories
        for scenario in ['solo_phase', 'group_1', 'group_2']:
            scenario_data = self.results[scenario]
            if isinstance(scenario_data, list):
                for run in scenario_data:
                    for agent_data in run['agent_details'].values():
                        trajectory = agent_data['consciousness_trajectory']
                        if trajectory:
                            consciousness_trajectories.append({
                                'agent_id': agent_data['agent_id'],
                                'scenario': scenario,
                                'trajectory': trajectory
                            })
            else:
                for agent_data in scenario_data['agent_details'].values():
                    trajectory = agent_data['consciousness_trajectory']
                    if trajectory:
                        consciousness_trajectories.append({
                            'agent_id': agent_data['agent_id'],
                            'scenario': scenario,
                            'trajectory': trajectory
                        })
        
        # Analyze evolution patterns
        evolution_patterns = []
        for traj_data in consciousness_trajectories:
            trajectory = traj_data['trajectory']
            if len(trajectory) > 1:
                start_freq = trajectory[0]['frequency']
                end_freq = trajectory[-1]['frequency']
                start_coh = trajectory[0]['coherence']
                end_coh = trajectory[-1]['coherence']
                
                evolution_patterns.append({
                    'agent_id': traj_data['agent_id'],
                    'scenario': traj_data['scenario'],
                    'frequency_change': end_freq - start_freq,
                    'coherence_change': end_coh - start_coh,
                    'total_steps': len(trajectory)
                })
        
        return {
            'trajectories_analyzed': len(evolution_patterns),
            'average_frequency_change': sum(p['frequency_change'] for p in evolution_patterns) / len(evolution_patterns) if evolution_patterns else 0.0,
            'average_coherence_change': sum(p['coherence_change'] for p in evolution_patterns) / len(evolution_patterns) if evolution_patterns else 0.0,
            'evolution_examples': evolution_patterns[:3]
        }
    
    def _analyze_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance across different scenarios."""
        scenario_performance = {}
        
        # Solo phase average
        if self.results['solo_phase']:
            solo_completion_rate = sum(run['simulation_metrics']['completion_rate'] 
                                     for run in self.results['solo_phase']) / len(self.results['solo_phase'])
            solo_avg_steps = sum(run['simulation_metrics']['total_steps'] 
                               for run in self.results['solo_phase']) / len(self.results['solo_phase'])
            scenario_performance['solo_phase'] = {
                'completion_rate': solo_completion_rate,
                'average_steps': solo_avg_steps
            }
        
        # Group scenarios
        for group_name in ['group_1', 'group_2']:
            if group_name in self.results and self.results[group_name]:
                group_data = self.results[group_name]
                scenario_performance[group_name] = {
                    'completion_rate': group_data['simulation_metrics']['completion_rate'],
                    'average_steps': group_data['simulation_metrics']['total_steps']
                }
        
        return scenario_performance
    
    def _get_most_common(self, list_of_lists: List[List[str]]) -> Dict[str, int]:
        """Get most common items across lists."""
        all_items = [item for sublist in list_of_lists for item in sublist]
        counts = {}
        for item in all_items:
            counts[item] = counts.get(item, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def _print_run_summary(self, run_result: Dict[str, Any], title: str):
        """Print summary of a single run."""
        metrics = run_result['simulation_metrics']
        print(f"\n{title} Summary:")
        print(f"  Success: {metrics['completion_rate']:.1%}")
        print(f"  Steps: {metrics['total_steps']}")
        print(f"  Time: {metrics['simulation_time']:.2f}s")
        
        # Agent consciousness evolution
        for agent_id, agent_data in run_result['agent_details'].items():
            initial = run_result['initial_states'][agent_id]['consciousness']
            final = run_result['final_states'][agent_id]['consciousness']
            print(f"  {agent_id}: {initial['frequency']:.2f}Hz/{initial['coherence']:.2f} → {final['frequency']:.2f}Hz/{final['coherence']:.2f}")
    
    def _print_group_summary(self, group_result: Dict[str, Any], title: str):
        """Print summary of group scenario."""
        metrics = group_result['simulation_metrics']
        print(f"\n{title} Summary:")
        print(f"  Success Rate: {metrics['completion_rate']:.1%}")
        print(f"  Total Steps: {metrics['total_steps']}")
        print(f"  Agents Completed: {metrics['agents_completed']}")
        
        # Individual agent performance
        for agent_id, agent_data in group_result['agent_details'].items():
            success = agent_data['final_status']['goal_achieved']
            steps = agent_data['final_status']['step_count']
            print(f"  {agent_id}: {'SUCCESS' if success else 'INCOMPLETE'} ({steps} steps)")
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        print(f"Results loaded from {filepath}")


# Utility function to run quick test
def run_ubf_test(maze_size: Tuple[int, int] = (10, 10), max_steps: int = 500, 
                 random_seed: int = None, save_file: str = None) -> Dict[str, Any]:
    """
    Quick function to run all UBF test scenarios.
    
    Args:
        maze_size: Size of maze (width, height)
        max_steps: Maximum steps per simulation
        random_seed: Random seed for reproducibility  
        save_file: Optional file to save results
        
    Returns:
        Complete test results
    """
    test_runner = UBFTestScenarios(maze_size, max_steps)
    results = test_runner.run_all_scenarios(random_seed)
    
    if save_file:
        test_runner.save_results(save_file)
    
    return results