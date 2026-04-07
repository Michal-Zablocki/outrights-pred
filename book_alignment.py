import json
import math
import os
from typing import Optional

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.optimize import minimize


load_dotenv()

API_TOKEN = os.getenv('X-RapidAPI-Key')
HFA = 0.045  # home field advantage; optimized
K_FACTOR = 20  # Elo rating system K-factor
NU = 1.65  # draw parameter; optimized
ODDS_FILEPATH = 'data/football-data/POL.csv'
SEASON = '2025/2026'


def load_book_odds(
    file_path: str, season: str, min_date: Optional[str] = None
) -> pd.DataFrame:
    """Load bookmaker odds from an Excel file for a given season. Vectorized."""
    odds_df = pd.read_csv(file_path)
    odds_df = odds_df[odds_df['Season'] == season]

    odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%d/%m/%Y')
    odds_df['Date'] = odds_df['Date'].dt.strftime('%Y-%m-%d')

    if min_date:
        odds_df = odds_df[odds_df['Date'] >= min_date]

    odds_df = odds_df[
        [
            'Country',
            'League',
            'Season',
            'Date',
            'Home',
            'Away',
            'HG',
            'AG',
            'Res',
            'PSCH',
            'PSCD',
            'PSCA',
            'B365CH',
            'B365CD',
            'B36CA',
        ]
    ]

    odds_df['HomeOdds'] = odds_df['PSCH'].fillna(odds_df['B365CH'])
    odds_df['DrawOdds'] = odds_df['PSCD'].fillna(odds_df['B365CD'])
    odds_df['AwayOdds'] = odds_df['PSCA'].fillna(odds_df['B36CA'])

    odds_df.drop(
        columns=['PSCH', 'PSCD', 'PSCA', 'B365CH', 'B365CD', 'B36CA'], inplace=True
    )
    odds_df.dropna(inplace=True)

    # Vectorized probability calculations
    total_inv_odds = (
        1 / odds_df['HomeOdds'] + 1 / odds_df['DrawOdds'] + 1 / odds_df['AwayOdds']
    )

    odds_df['ImpliedPProbH'] = (1 / odds_df['HomeOdds']) / total_inv_odds
    odds_df['ImpliedPProbD'] = (1 / odds_df['DrawOdds']) / total_inv_odds
    odds_df['ImpliedPProbA'] = (1 / odds_df['AwayOdds']) / total_inv_odds
    odds_df['PBookieSpread'] = total_inv_odds - 1

    print(odds_df.describe())

    return odds_df


def get_teams(df: pd.DataFrame) -> list:
    teams = set(df['Home']).union(set(df['Away']))
    teams = sorted(teams)
    print(teams)
    return teams


def get_odds(elo_home: float, elo_away: float) -> tuple[float, float, float]:
    """Calculate the odds for home win, draw, and away win based on Elo ratings."""

    elo_difference = elo_home * (1 + HFA) - elo_away

    p_win_base = 1 / (1 + math.pow(10, -elo_difference / 400))
    denom = 1 + NU * p_win_base * (1 - p_win_base)
    pH = p_win_base / denom
    pA = (1 - p_win_base) / denom
    pD = 1 - pH - pA

    return pH, pD, pA


def elo_error_function(df: pd.DataFrame, team_elo_map: dict) -> float:
    """
    Calculate error between bookmaker and simulated probabilities.
    Uses KL divergence which heavily penalizes differences in small probabilities.
    """
    df = df.copy()
    df['EloHome'] = df['Home'].map(team_elo_map)
    df['EloAway'] = df['Away'].map(team_elo_map)

    # Vectorized odds calculation
    elo_diff = df['EloHome'].values * (1 + HFA) - df['EloAway'].values
    p_win_base = 1 / (1 + np.power(10, -elo_diff / 400))
    denom = 1 + NU * p_win_base * (1 - p_win_base)

    df['SimPProbH'] = p_win_base / denom
    df['SimPProbA'] = (1 - p_win_base) / denom
    df['SimPProbD'] = 1 - df['SimPProbH'] - df['SimPProbA']

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10

    # Vectorized KL divergence calculation
    p_h = df['ImpliedPProbH'].values + epsilon
    p_d = df['ImpliedPProbD'].values + epsilon
    p_a = df['ImpliedPProbA'].values + epsilon

    q_h = df['SimPProbH'].values + epsilon
    q_d = df['SimPProbD'].values + epsilon
    q_a = df['SimPProbA'].values + epsilon

    # KL divergence: sum over outcomes of p * log(p/q)
    kl_divergence = (
        p_h * np.log(p_h / q_h) + p_d * np.log(p_d / q_d) + p_a * np.log(p_a / q_a)
    ).mean()

    return kl_divergence


def optimize_elo_ratings(odds_df: pd.DataFrame, teams: list, method='L-BFGS-B') -> dict:
    """
    Optimize Elo ratings to minimize KL divergence between bookmaker and simulated probabilities.

    Parameters:
    -----------
    odds_df : pd.DataFrame
        DataFrame with match data and bookmaker odds
    teams : list
        List of team names
    method : str
        Optimization method: 'L-BFGS-B', 'differential_evolution', or 'SLSQP'

    Returns:
    --------
    dict : Optimized Elo ratings for each team
    """
    n_teams = len(teams)
    team_to_idx = {team: i for i, team in enumerate(teams)}

    # Objective function for optimizer
    def objective(elo_array):
        team_elo_map = {team: elo_array[team_to_idx[team]] for team in teams}
        return elo_error_function(odds_df, team_elo_map)

    # Initial guess: all teams at 1500
    x0 = np.full(n_teams, 1500.0)

    # Bounds: Elo ratings typically range from 1000 to 2000
    bounds = [(1000, 2000) for _ in range(n_teams)]

    print(f"Optimizing Elo ratings for {n_teams} teams using {method}...")
    print(f"Initial error: {objective(x0):.6f}")

    if method == 'differential_evolution':
        # Global optimization - slower but more robust
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-7,
            workers=-1,  # Use all CPU cores
            updating='deferred',
            disp=True,
        )
    else:
        # Local optimization - faster
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options={'maxiter': 1000, 'disp': True},
        )

    optimized_elos = {team: round(result.x[team_to_idx[team]], 2) for team in teams}

    print(f"\nOptimization complete!")
    print(f"Final error: {result.fun:.6f}")
    print(f"Error reduction: {(objective(x0) - result.fun) / objective(x0) * 100:.2f}%")

    return optimized_elos


def compare_elos(initial_elos: dict, optimized_elos: dict) -> pd.DataFrame:
    """Compare initial and optimized Elo ratings."""
    comparison = pd.DataFrame(
        {
            'Team': list(initial_elos.keys()),
            'Initial_Elo': list(initial_elos.values()),
            'Optimized_Elo': [optimized_elos[team] for team in initial_elos.keys()],
        }
    )
    comparison['Elo_Change'] = comparison['Optimized_Elo'] - comparison['Initial_Elo']
    comparison = comparison.sort_values('Optimized_Elo', ascending=False)
    return comparison


def main():
    odds_df = load_book_odds(ODDS_FILEPATH, SEASON, min_date='2025-09-01')
    teams = get_teams(odds_df)

    # Initial Elo ratings (all 1500)
    initial_team_elo_map = {team: 1500.0 for team in teams}
    initial_error = elo_error_function(odds_df, initial_team_elo_map)
    print(f'\nInitial Elo error (all teams at 1500): {initial_error:.6f}\n')

    # Optimize Elo ratings
    # Try 'L-BFGS-B' first (fast), or 'differential_evolution' for global optimum
    optimized_elos = optimize_elo_ratings(odds_df, teams, method='L-BFGS-B')

    # Calculate optimized error
    optimized_error = elo_error_function(odds_df, optimized_elos)
    print(f'\nOptimized Elo error: {optimized_error:.6f}')

    # Compare results
    comparison_df = compare_elos(initial_team_elo_map, optimized_elos)
    print("\nElo Rating Comparison:")
    print(comparison_df.to_string(index=False))

    # Save optimized Elos
    output_file = f'data/optimized_elos_{SEASON.replace("/", "_")}.json'
    with open(output_file, 'w') as f:
        json.dump(optimized_elos, f, indent=2)
    print(f"\nOptimized Elos saved to: {output_file}")


if __name__ == "__main__":
    main()
