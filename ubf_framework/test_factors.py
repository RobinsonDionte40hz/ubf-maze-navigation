#!/usr/bin/env python3
"""
Test Factor Contributions
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tests.test_scenarios import run_ubf_test
import json

def test_factor_contributions():
    """Test if factor contributions are working"""
    print('Testing factor contributions...')

    # Run one quick test
    results = run_ubf_test(maze_size=(8, 8), max_steps=50, generate_summary=False)
    solo_run = results['solo_phase'][0]

    # Check the first few decision breakdowns
    for i, step in enumerate(solo_run['step_logs'][:3]):
        if 'agent_results' in step and 'solo_agent' in step['agent_results']:
            breakdown = step['agent_results']['solo_agent'].get('decision_breakdown', {})
            factors = breakdown.get('factor_contributions', {})
            print(f'Step {i+1}:')
            if factors:
                print(f'  Available action types: {list(factors.keys())}')
                # Check memory influence for each action
                for action_type, action_factors in factors.items():
                    memory_influence = action_factors.get('memory_influence', 'NOT FOUND')
                    print(f'    {action_type}: memory_influence = {memory_influence}')
                    if isinstance(memory_influence, (int, float)) and memory_influence != 1.0:
                        print(f'      MEMORY IS BEING USED! Factor: {memory_influence:.3f}')
            else:
                print('  No factor contributions found!')

    print('Factor contributions test complete')

if __name__ == '__main__':
    test_factor_contributions()