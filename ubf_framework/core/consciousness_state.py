"""
Universal Behavioral Framework - Consciousness State

This module implements the core consciousness coordinates (frequency and coherence)
that drive all agent behavior in the UBF system.
"""

import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class ConsciousnessState:
    """
    Core consciousness coordinates that define an agent's behavioral state.
    
    Frequency (3-15 Hz): Energy, activation, drive
    - Low (3-6): Lethargic, passive, withdrawn
    - Moderate (6-10): Balanced, steady, functional  
    - High (10-15): Energetic, active, driven
    
    Coherence (0.2-1.0): Focus, clarity, stability
    - Low (0.2-0.4): Scattered, unfocused, chaotic
    - Moderate (0.4-0.8): Balanced, flexible
    - High (0.8-1.0): Focused, precise, stable
    """
    
    frequency: float  # 3.0-15.0 Hz
    coherence: float  # 0.2-1.0
    last_updated: float = 0.0  # Timestamp of last update
    
    def __post_init__(self):
        """Ensure coordinates stay within valid bounds."""
        self.clamp_values()
    
    def clamp_values(self):
        """Enforce coordinate boundaries."""
        self.frequency = max(3.0, min(15.0, self.frequency))
        self.coherence = max(0.2, min(1.0, self.coherence))
    
    def update_coordinates(self, freq_delta: float, coh_delta: float, noise_std: float = 0.1):
        """
        Update consciousness coordinates with experience-driven changes.
        
        Args:
            freq_delta: Change in frequency from experience outcome
            coh_delta: Change in coherence from experience outcome  
            noise_std: Standard deviation for Gaussian noise injection
        """
        # Add Gaussian noise for creativity (scaled by low coherence for scattered exploration)
        coherence_factor = max(0.5, 1.0 - self.coherence)  # More noise when scattered
        freq_noise = random.gauss(0, noise_std * coherence_factor)
        coh_noise = random.gauss(0, noise_std * coherence_factor)
        
        # Apply updates with noise
        self.frequency += freq_delta + freq_noise
        self.coherence += coh_delta + coh_noise
        
        # Enforce bounds
        self.clamp_values()
    
    def get_energy_level(self) -> str:
        """Get categorical energy level from frequency."""
        if self.frequency < 6.0:
            return "low"
        elif self.frequency < 10.0:
            return "moderate"
        else:
            return "high"
    
    def get_focus_level(self) -> str:
        """Get categorical focus level from coherence."""
        if self.coherence < 0.4:
            return "scattered"
        elif self.coherence < 0.8:
            return "balanced"
        else:
            return "focused"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'frequency': self.frequency,
            'coherence': self.coherence,
            'energy_level': self.get_energy_level(),
            'focus_level': self.get_focus_level(),
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsciousnessState':
        """Create instance from dictionary."""
        return cls(
            frequency=data['frequency'],
            coherence=data['coherence'],
            last_updated=data.get('last_updated', 0.0)
        )
    
    def __str__(self) -> str:
        return f"Consciousness(freq={self.frequency:.2f}Hz, coh={self.coherence:.2f}, {self.get_energy_level()}/{self.get_focus_level()})"


@dataclass 
class BehavioralState:
    """
    Cached behavioral dimensions derived from consciousness coordinates.
    Updated only when significant events occur for performance optimization.
    """
    
    energy: float          # 0.0-1.0: Physical/mental activation
    focus: float           # 0.0-1.0: Concentration ability
    mood: float           # -1.0 to +1.0: Emotional valence  
    social_drive: float   # 0.0-1.0: Tendency toward social interaction
    risk_tolerance: float # 0.0-1.0: Willingness to take risks
    ambition: float       # 0.0-1.0: Drive to achieve goals
    
    creativity: float     # 0.0-1.0: Tendency for novel solutions
    adaptability: float   # 0.0-1.0: Flexibility in changing plans
    
    last_generated: float = 0.0  # When this state was computed
    
    @classmethod
    def from_consciousness(cls, consciousness: ConsciousnessState) -> 'BehavioralState':
        """
        Generate behavioral state from consciousness coordinates.
        Uses simple deterministic formulas for fast computation.
        """
        freq = consciousness.frequency
        coh = consciousness.coherence
        
        # Energy: Direct mapping from frequency
        energy = (freq - 3.0) / 12.0  # Map 3-15 to 0-1
        
        # Focus: Direct mapping from coherence  
        focus = (coh - 0.2) / 0.8  # Map 0.2-1.0 to 0-1
        
        # Mood: Combined function of both coordinates
        # High freq + high coh = very positive
        # Low freq + low coh = very negative
        mood_base = (freq - 9.0) / 6.0  # Center around 9Hz
        mood_coherence_boost = (coh - 0.6) * 0.5  # Coherence stabilizes mood
        mood = max(-1.0, min(1.0, mood_base + mood_coherence_boost))
        
        # Social drive: Frequency-based with coherence modulation
        social_drive = max(0.0, min(1.0, (freq - 4.0) / 8.0))
        if coh < 0.4:  # Scattered state reduces social effectiveness
            social_drive *= 0.7
        
        # Risk tolerance: Frequency-driven, coherence-modulated
        risk_tolerance = max(0.0, min(1.0, (freq - 6.0) / 6.0))
        if coh > 0.8:  # High focus can enable calculated risks
            risk_tolerance = min(1.0, risk_tolerance * 1.2)
        
        # Ambition: Product of frequency and coherence (need both energy and focus)
        ambition = max(0.0, min(1.0, coh * (freq / 10.0)))
        
        # Creativity: Inverse relationship with coherence (scattered = creative)
        # But need some energy to act on creativity
        creativity_coh = max(0.2, 1.0 - coh)  # High when scattered
        creativity_energy = min(1.0, freq / 10.0)  # Need some energy
        creativity = creativity_coh * creativity_energy
        
        # Adaptability: Balance of moderate coherence and frequency
        # Peak around coh=0.6 (flexible but not scattered)
        adapt_coh_factor = 1.0 - abs(coh - 0.6) / 0.4  # Peak at 0.6
        adapt_freq_factor = min(1.0, freq / 12.0)
        adaptability = adapt_coh_factor * adapt_freq_factor
        
        return cls(
            energy=energy,
            focus=focus, 
            mood=mood,
            social_drive=social_drive,
            risk_tolerance=risk_tolerance,
            ambition=ambition,
            creativity=creativity,
            adaptability=adaptability,
            last_generated=consciousness.last_updated
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'energy': self.energy,
            'focus': self.focus,
            'mood': self.mood,
            'social_drive': self.social_drive,
            'risk_tolerance': self.risk_tolerance,
            'ambition': self.ambition,
            'creativity': self.creativity,
            'adaptability': self.adaptability,
            'last_generated': self.last_generated
        }
    
    def __str__(self) -> str:
        return (f"Behavioral(energy={self.energy:.2f}, focus={self.focus:.2f}, "
                f"mood={self.mood:+.2f}, social={self.social_drive:.2f}, "
                f"risk={self.risk_tolerance:.2f}, ambition={self.ambition:.2f})")