"""
Universal Behavioral Framework - Performance Optimizations

SIMD batching hints and efficiency optimizations for scaling to swarms.
"""

import math
from typing import List, Dict, Any, Tuple
import time

from .agent import Agent
from .consciousness_state import ConsciousnessState, BehavioralState
from .decision_system import DecisionSystem


class BatchProcessor:
    """
    Batch processor for efficient computation of agent states and decisions.
    Implements SIMD-friendly algorithms for scaling to large swarms.
    """
    
    @staticmethod
    def batch_consciousness_update(consciousness_states: List[ConsciousnessState],
                                 freq_deltas: List[float],
                                 coh_deltas: List[float],
                                 noise_std: float = 0.1) -> List[ConsciousnessState]:
        """
        Batch update consciousness coordinates for multiple agents.
        SIMD-friendly implementation for vector operations.
        
        Args:
            consciousness_states: List of consciousness states
            freq_deltas: Frequency changes for each agent
            coh_deltas: Coherence changes for each agent
            noise_std: Standard deviation for noise injection
            
        Returns:
            Updated consciousness states
        """
        updated_states = []
        
        # Vectorized noise generation (SIMD hint)
        import random
        noise_pairs = [(random.gauss(0, noise_std), random.gauss(0, noise_std)) 
                      for _ in range(len(consciousness_states))]
        
        # Batch processing with SIMD-friendly operations
        for i, (state, freq_delta, coh_delta) in enumerate(zip(consciousness_states, freq_deltas, coh_deltas)):
            freq_noise, coh_noise = noise_pairs[i]
            
            # Apply updates (vectorizable)
            new_freq = state.frequency + freq_delta + freq_noise
            new_coh = state.coherence + coh_delta + coh_noise
            
            # Clamp values (vectorizable)
            new_freq = max(3.0, min(15.0, new_freq))
            new_coh = max(0.2, min(1.0, new_coh))
            
            # Create updated state
            updated_state = ConsciousnessState(new_freq, new_coh)
            updated_state.last_updated = state.last_updated
            updated_states.append(updated_state)
        
        return updated_states
    
    @staticmethod
    def batch_behavioral_state_generation(consciousness_states: List[ConsciousnessState]) -> List[BehavioralState]:
        """
        Generate behavioral states for multiple agents in batch.
        
        Args:
            consciousness_states: List of consciousness states
            
        Returns:
            List of behavioral states
        """
        behavioral_states = []
        
        # Extract coordinates for vectorized operations
        frequencies = [cs.frequency for cs in consciousness_states]
        coherences = [cs.coherence for cs in consciousness_states]
        
        # Vectorized calculations (SIMD-friendly)
        energies = [(freq - 3.0) / 12.0 for freq in frequencies]
        focuses = [(coh - 0.2) / 0.8 for coh in coherences]
        
        # Mood calculation (vectorized)
        moods = []
        for freq, coh in zip(frequencies, coherences):
            mood_base = (freq - 9.0) / 6.0
            mood_coherence_boost = (coh - 0.6) * 0.5
            mood = max(-1.0, min(1.0, mood_base + mood_coherence_boost))
            moods.append(mood)
        
        # Social drive (vectorized)
        social_drives = []
        for freq, coh in zip(frequencies, coherences):
            social_drive = max(0.0, min(1.0, (freq - 4.0) / 8.0))
            if coh < 0.4:
                social_drive *= 0.7
            social_drives.append(social_drive)
        
        # Risk tolerance (vectorized)
        risk_tolerances = []
        for freq, coh in zip(frequencies, coherences):
            risk_tolerance = max(0.0, min(1.0, (freq - 6.0) / 6.0))
            if coh > 0.8:
                risk_tolerance = min(1.0, risk_tolerance * 1.2)
            risk_tolerances.append(risk_tolerance)
        
        # Ambition (vectorized)
        ambitions = [max(0.0, min(1.0, coh * (freq / 10.0))) 
                    for freq, coh in zip(frequencies, coherences)]
        
        # Creativity (vectorized)
        creativities = []
        for freq, coh in zip(frequencies, coherences):
            creativity_coh = max(0.2, 1.0 - coh)
            creativity_energy = min(1.0, freq / 10.0)
            creativity = creativity_coh * creativity_energy
            creativities.append(creativity)
        
        # Adaptability (vectorized)
        adaptabilities = []
        for freq, coh in zip(frequencies, coherences):
            adapt_coh_factor = 1.0 - abs(coh - 0.6) / 0.4
            adapt_freq_factor = min(1.0, freq / 12.0)
            adaptability = adapt_coh_factor * adapt_freq_factor
            adaptabilities.append(adaptability)
        
        # Construct behavioral states
        for i in range(len(consciousness_states)):
            behavioral_state = BehavioralState(
                energy=energies[i],
                focus=focuses[i],
                mood=moods[i],
                social_drive=social_drives[i],
                risk_tolerance=risk_tolerances[i],
                ambition=ambitions[i],
                creativity=creativities[i],
                adaptability=adaptabilities[i],
                last_generated=consciousness_states[i].last_updated
            )
            behavioral_states.append(behavioral_state)
        
        return behavioral_states
    
    @staticmethod
    def batch_quantum_resonance(consciousness_states: List[ConsciousnessState],
                               action_energies: List[float],
                               gamma_freq: float = 40.0) -> List[float]:
        """
        Calculate quantum resonance for multiple agents in batch.
        
        Args:
            consciousness_states: List of consciousness states
            action_energies: Energy levels of actions being evaluated
            gamma_freq: Gamma frequency baseline
            
        Returns:
            List of resonance values
        """
        resonances = []
        
        # Vectorized resonance calculation
        for cs, action_energy in zip(consciousness_states, action_energies):
            character_energy = cs.frequency
            energy_diff = character_energy - action_energy
            
            # Quantum resonance formula (vectorizable)
            exponent = -math.pow(energy_diff - gamma_freq, 2) / (2 * gamma_freq)
            resonance = math.exp(exponent)
            
            # Coherence amplification
            coherence_amplifier = 0.5 + cs.coherence * 0.5
            final_resonance = resonance * coherence_amplifier
            
            resonances.append(final_resonance)
        
        return resonances
    
    @staticmethod
    def batch_memory_influence(agents: List[Agent], 
                             interaction_types: List[str]) -> List[float]:
        """
        Calculate memory influence for multiple agents in batch.
        
        Args:
            agents: List of agents
            interaction_types: Interaction types to evaluate
            
        Returns:
            List of memory influence multipliers
        """
        influences = []
        
        for agent, interaction_type in zip(agents, interaction_types):
            from ..core.memory_system import InteractionType
            try:
                int_type = InteractionType(interaction_type)
                influence = agent.memory_manager.calculate_memory_influence(int_type)
            except ValueError:
                influence = 1.0  # Neutral if unknown type
            influences.append(influence)
        
        return influences


class PerformanceProfiler:
    """
    Performance profiling and optimization recommendations for UBF systems.
    """
    
    def __init__(self):
        self.timings = {}
        self.operation_counts = {}
    
    def profile_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Profile a single operation."""
        start_time = time.perf_counter()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        
        if operation_name not in self.timings:
            self.timings[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        self.timings[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {}
        
        for operation, times in self.timings.items():
            count = len(times)
            total_time = sum(times)
            avg_time = total_time / count
            min_time = min(times)
            max_time = max(times)
            
            # Calculate operations per second
            ops_per_second = 1.0 / avg_time if avg_time > 0 else float('inf')
            
            report[operation] = {
                'count': count,
                'total_time': total_time,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'ops_per_second': ops_per_second,
                'microseconds_per_op': avg_time * 1_000_000
            }
        
        return report
    
    def print_performance_report(self):
        """Print formatted performance report."""
        report = self.get_performance_report()
        
        print("\n=== UBF Performance Report ===")
        print(f"{'Operation':<25} {'Count':<8} {'Avg Time':<12} {'Ops/Sec':<12} {'μs/Op':<10}")
        print("-" * 70)
        
        for operation, metrics in report.items():
            avg_time_str = f"{metrics['avg_time']:.6f}s"
            ops_per_sec_str = f"{metrics['ops_per_second']:.0f}"
            microseconds_str = f"{metrics['microseconds_per_op']:.1f}"
            
            print(f"{operation:<25} {metrics['count']:<8} {avg_time_str:<12} {ops_per_sec_str:<12} {microseconds_str:<10}")


class SwarmOptimizer:
    """
    Optimizations for large-scale swarm simulations.
    """
    
    @staticmethod
    def optimize_agent_update_order(agents: List[Agent]) -> List[Agent]:
        """
        Optimize agent update order for better cache locality.
        Groups agents by similar consciousness states.
        
        Args:
            agents: List of agents to reorder
            
        Returns:
            Reordered list for better performance
        """
        # Sort by consciousness coordinates for cache locality
        return sorted(agents, key=lambda a: (a.consciousness.frequency, a.consciousness.coherence))
    
    @staticmethod
    def batch_decision_weights(agents: List[Agent], 
                             available_actions: List[List],
                             environmental_contexts: List[Dict]) -> List[List[float]]:
        """
        Calculate decision weights for multiple agents in batch.
        
        Args:
            agents: List of agents
            available_actions: Actions available to each agent
            environmental_contexts: Environmental context for each agent
            
        Returns:
            Decision weights for each agent's actions
        """
        all_weights = []
        
        # Group agents with similar states for batch processing
        for agent, actions, context in zip(agents, available_actions, environmental_contexts):
            weights = []
            for action in actions:
                weight = agent.decision_system.calculate_interaction_weight(
                    consciousness=agent.consciousness,
                    behavioral_state=agent.behavioral_state,
                    memory_manager=agent.memory_manager,
                    action=action,
                    environmental_context=context
                )
                weights.append(weight)
            all_weights.append(weights)
        
        return all_weights
    
    @staticmethod
    def memory_consolidation_batch(agents: List[Agent], 
                                  consolidation_threshold: int = 20):
        """
        Perform memory consolidation for multiple agents when needed.
        
        Args:
            agents: List of agents
            consolidation_threshold: Minimum memories before consolidation
        """
        for agent in agents:
            if len(agent.memory_manager.memories) >= consolidation_threshold:
                agent.memory_manager.consolidate_similar_memories()


# Performance testing utilities
def benchmark_ubf_operations():
    """Benchmark core UBF operations for performance analysis."""
    profiler = PerformanceProfiler()
    
    print("Benchmarking UBF operations...")
    
    # Create test data
    test_agents = [Agent(f"test_{i}") for i in range(100)]
    test_consciousness = [agent.consciousness for agent in test_agents]
    
    # Benchmark consciousness updates
    freq_deltas = [0.1] * 100
    coh_deltas = [0.05] * 100
    
    for _ in range(10):
        profiler.profile_operation(
            "batch_consciousness_update",
            BatchProcessor.batch_consciousness_update,
            test_consciousness,
            freq_deltas,
            coh_deltas
        )
    
    # Benchmark behavioral state generation
    for _ in range(10):
        profiler.profile_operation(
            "batch_behavioral_state_generation",
            BatchProcessor.batch_behavioral_state_generation,
            test_consciousness
        )
    
    # Benchmark quantum resonance
    action_energies = [8.0] * 100
    for _ in range(10):
        profiler.profile_operation(
            "batch_quantum_resonance",
            BatchProcessor.batch_quantum_resonance,
            test_consciousness,
            action_energies
        )
    
    # Print results
    profiler.print_performance_report()
    
    return profiler.get_performance_report()


def estimate_scaling_performance(agent_count: int) -> Dict[str, Any]:
    """
    Estimate performance for different agent counts.
    
    Args:
        agent_count: Number of agents to simulate
        
    Returns:
        Performance estimates
    """
    # Base timings (microseconds per operation)
    base_timings = {
        'consciousness_update': 2.0,
        'behavioral_state_gen': 1.5,
        'decision_making': 8.0,
        'memory_operations': 3.0,
        'environment_step': 5.0
    }
    
    # Calculate total time per step
    total_time_per_step = sum(base_timings.values()) * agent_count
    
    # Estimate steps per second
    steps_per_second = 1_000_000 / total_time_per_step  # Convert μs to seconds
    
    # Memory estimation (bytes per agent)
    memory_per_agent = {
        'consciousness_state': 64,
        'behavioral_state': 128,
        'memories': 8 * 1024,  # 8KB for 50 memories
        'agent_overhead': 256
    }
    
    total_memory_per_agent = sum(memory_per_agent.values())
    total_memory_mb = (total_memory_per_agent * agent_count) / (1024 * 1024)
    
    return {
        'agent_count': agent_count,
        'estimated_steps_per_second': steps_per_second,
        'time_per_step_ms': total_time_per_step / 1000,
        'memory_usage_mb': total_memory_mb,
        'scalability_class': _classify_scalability(agent_count, steps_per_second),
        'recommendations': _get_scaling_recommendations(agent_count, steps_per_second)
    }


def _classify_scalability(agent_count: int, steps_per_second: float) -> str:
    """Classify scalability performance."""
    if steps_per_second > 1000:
        return "excellent"
    elif steps_per_second > 100:
        return "good"
    elif steps_per_second > 10:
        return "acceptable"
    else:
        return "optimization_needed"


def _get_scaling_recommendations(agent_count: int, steps_per_second: float) -> List[str]:
    """Get optimization recommendations based on performance."""
    recommendations = []
    
    if agent_count > 1000:
        recommendations.append("Consider SIMD batch processing for consciousness updates")
        recommendations.append("Implement memory consolidation scheduler")
        recommendations.append("Use spatial partitioning for environment interactions")
    
    if steps_per_second < 10:
        recommendations.append("Enable agent update order optimization")
        recommendations.append("Reduce memory system max_memories_per_agent")
        recommendations.append("Consider distributed processing for very large swarms")
    
    if agent_count > 10000:
        recommendations.append("Implement GPU acceleration for decision weight calculations")
        recommendations.append("Use approximate algorithms for less critical operations")
    
    return recommendations