#!/usr/bin/env python3
"""
Regression Analysis Script
Runs multiple solo phase iterations to investigate learning regression
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tests.test_scenarios import UBFTestScenarios
import json

#!/usr/bin/env python3
"""
Regression Analysis Script
Runs multiple solo phase iterations to investigate learning regression
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tests.test_scenarios import run_ubf_test
import json

def run_regression_analysis():
    """Run solo phase multiple times and analyze results"""
    results = []

    for i in range(5):
        print(f'Running full test suite iteration {i+1}...')
        # Run full test suite and extract solo phase results
        full_results = run_ubf_test(maze_size=(8, 8), max_steps=1000, generate_summary=False)
        solo_results = full_results['solo_phase']
        results.append(solo_results)
        print(f'  Solo phase: 3 runs completed')

    # Save detailed results
    with open('regression_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('Regression analysis complete - check regression_analysis.json')

    # Analyze patterns across all iterations
    all_runs = []
    for iteration_results in results:
        for run in iteration_results:
            all_runs.append(run)

    # Group by run number (1, 2, 3)
    run_1_steps = [r['simulation_metrics']['total_steps'] for r in all_runs if r['run_number'] == 1]
    run_2_steps = [r['simulation_metrics']['total_steps'] for r in all_runs if r['run_number'] == 2]
    run_3_steps = [r['simulation_metrics']['total_steps'] for r in all_runs if r['run_number'] == 3]

    print(f'\nAnalysis across {len(results)} iterations:')
    print(f'Run 1 average steps: {sum(run_1_steps)/len(run_1_steps):.1f}')
    print(f'Run 2 average steps: {sum(run_2_steps)/len(run_2_steps):.1f}')
    print(f'Run 3 average steps: {sum(run_3_steps)/len(run_3_steps):.1f}')

    # Check for regression pattern
    if run_3_steps and run_2_steps:
        avg_run2 = sum(run_2_steps)/len(run_2_steps)
        avg_run3 = sum(run_3_steps)/len(run_3_steps)
        regression_pct = ((avg_run3 - avg_run2) / avg_run2) * 100

        if regression_pct > 50:  # Significant regression
            print(f'WARNING: Learning regression detected! Run 3 is {regression_pct:.1f}% worse than Run 2')
        elif regression_pct < -20:  # Significant improvement
            print(f'SUCCESS: Learning improvement! Run 3 is {abs(regression_pct):.1f}% better than Run 2')
        else:
            print(f'Learning appears stable (change: {regression_pct:+.1f}%)')

if __name__ == '__main__':
    run_regression_analysis()