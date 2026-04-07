from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import random

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from etl import *


load_dotenv()

HFA = 0.045  # home field advantage; optimized
K_FACTOR = 20  # Elo rating system K-factor
NU = 1.65  # draw parameter; optimized

_EURO_SORTING_ORDER = [
    'Points',
    'Goal difference',
    'Goals for',
    'Goals away',
    'Wins',
    'Random order',
]

SORTING_ORDERS = {
    'POL': [
        'Points',
        'H2H',
        'Goal difference',
        'Goals for',
        'Wins',
        'Random order',
    ],
    'UCL': _EURO_SORTING_ORDER,
    'UEL': _EURO_SORTING_ORDER,
    'ECL': _EURO_SORTING_ORDER,
}


@dataclass
class TeamInTable:
    name: str
    elo: float | None
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goals_diff: int
    goals_away: int
    points: int


def get_sorting_order_for_country_code(country_code: str) -> list[str]:
    """Get sorting order based on country code."""

    if country_code in SORTING_ORDERS:
        return SORTING_ORDERS[country_code]
    else:
        return SORTING_ORDERS['POL']


def _compute_elo_difference(home_elo: float, away_elo: float) -> float:
    """Elo difference adjusted for home field advantage."""
    return home_elo * (1 + HFA) - away_elo


def _compute_match_probabilities(
    home_elo: float, away_elo: float
) -> tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win) from Elo ratings."""
    elo_difference = _compute_elo_difference(home_elo, away_elo)
    p_win_base = 1 / (1 + math.pow(10, -elo_difference / 400))
    denom = 1 + NU * p_win_base * (1 - p_win_base)
    pH = p_win_base / denom
    pA = (1 - p_win_base) / denom
    pD = 1 - pH - pA
    return pH, pD, pA


def _compute_elo_delta(elo_difference: float, outcome: float) -> float:
    """Compute Elo delta for the home team. outcome: 1=win, 0.5=draw, 0=loss."""
    expected = 1 / (1 + math.pow(10, -elo_difference / 400))
    return (outcome - expected) * K_FACTOR


def _update_elo(
    league_table: dict,
    home_team: str,
    away_team: str,
    outcome: float,
) -> None:
    """Update Elo ratings for both teams after a match."""
    elo_diff = _compute_elo_difference(
        league_table[home_team].elo, league_table[away_team].elo
    )
    delta = _compute_elo_delta(elo_diff, outcome)
    league_table[home_team].elo += delta
    league_table[away_team].elo -= delta


def _outcome_from_goals(home_goals: int, away_goals: int) -> float:
    """Return Elo outcome value: 1=home win, 0.5=draw, 0=away win."""
    if home_goals > away_goals:
        return 1.0
    elif home_goals < away_goals:
        return 0.0
    return 0.5


def _init_fixtures_matrix(teams: list) -> dict[str, dict]:
    """Create an empty fixtures matrix for the given teams."""
    return {team: {team2: None for team2 in teams if team2 != team} for team in teams}


def build_table_from_fixtures_matrix(
    fixtures_matrix: dict[dict],
    sorting_order: list[str] | None = None,
    reverse: bool = False,
    teams: list | None = None,
    elo_dict: dict | None = None,
    _compute_h2h: bool = True,
    point_deductions: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build a league table from a fixtures matrix."""

    if teams is None:
        teams = list(fixtures_matrix.keys())
    else:
        fixtures_matrix = {
            team: {
                team2: fixtures_matrix[team][team2] for team2 in teams if team2 != team
            }
            for team in teams
        }

    league_table = {}
    for team in teams:
        elo = elo_dict[team] if elo_dict and team in elo_dict else 0
        league_table[team] = TeamInTable(team, elo, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    for team_home in fixtures_matrix.keys():
        for team_away in fixtures_matrix[team_home].keys():
            result = fixtures_matrix[team_home][team_away]
            if result is None:
                continue

            home_goals, away_goals = result

            league_table[team_away].goals_away += away_goals

            if home_goals > away_goals:
                league_table[team_home].wins += 1
                league_table[team_away].losses += 1
            elif home_goals < away_goals:
                league_table[team_home].losses += 1
                league_table[team_away].wins += 1
            elif home_goals == away_goals:
                league_table[team_home].draws += 1
                league_table[team_away].draws += 1

            league_table[team_home].goals_for += home_goals
            league_table[team_away].goals_for += away_goals

            league_table[team_home].goals_against += away_goals
            league_table[team_away].goals_against += home_goals

    for team in league_table.keys():
        league_table[team].points = (
            league_table[team].wins * 3 + league_table[team].draws * 1
        )
        league_table[team].goals_diff = (
            league_table[team].goals_for - league_table[team].goals_against
        )
        league_table[team].matches_played = (
            league_table[team].wins
            + league_table[team].draws
            + league_table[team].losses
        )

    standings_df = pd.DataFrame(
        [
            {
                'Club': team_data.name,
                'Elo': team_data.elo,
                'Matches played': team_data.matches_played,
                'Wins': team_data.wins,
                'Draws': team_data.draws,
                'Losses': team_data.losses,
                'Goals for': team_data.goals_for,
                'Goals against': team_data.goals_against,
                'Goal difference': team_data.goals_diff,
                'Goals away': team_data.goals_away,
                'Points': team_data.points,
            }
            for team_data in league_table.values()
        ]
    )

    standings_df['Random order'] = np.random.permutation(standings_df.shape[0])
    standings_df['H2H'] = 1

    if point_deductions:
        for team, deduction in point_deductions.items():
            standings_df.loc[standings_df['Club'] == team, 'Points'] -= deduction

    if not sorting_order:
        sorting_order = ['Points', 'H2H', 'Goal difference', 'Random order']

    if _compute_h2h:
        points_obtained = standings_df['Points'].unique()

        for p in points_obtained:
            tied_teams = standings_df[standings_df['Points'] == p]['Club'].tolist()
            if len(tied_teams) == 1 or len(tied_teams) == standings_df.shape[0]:
                continue
            h2h_table = build_table_from_fixtures_matrix(
                fixtures_matrix,
                sorting_order=sorting_order,
                teams=tied_teams,
                _compute_h2h=False,
            )
            for team in tied_teams:
                team_rank = h2h_table[h2h_table['Club'] == team].index[0]
                standings_df.loc[standings_df['Club'] == team, 'H2H'] = team_rank

    standings_df = standings_df.sort_values(
        by=sorting_order, ascending=reverse
    ).reset_index(drop=True)
    standings_df.index += 1

    return standings_df


def df_to_dict_of_teams(standings_df: pd.DataFrame) -> dict:
    """Convert a DataFrame to a dictionary of TeamInTable."""

    league_table = {}
    for _, row in standings_df.iterrows():
        league_table[row['Club']] = TeamInTable(
            row['Club'],
            row['Elo'],
            row['Matches played'],
            row['Wins'],
            row['Draws'],
            row['Losses'],
            row['Goals for'],
            row['Goals against'],
            row['Goal difference'],
            row['Goals away'],
            row['Points'],
        )

    return league_table


def save_round_divide(num, den, precision=2) -> int | float:
    """Safe division with rounding. Returns -1 if division by zero."""

    if den == 0:
        return -1
    else:
        return round(num / den, precision)


def build_historical_standings_table_after_at_most_n_rounds(
    league_id: str,
    season: str,
    country_code_elo: str | None,
    country_code_api: str | None,
    elo_date: str | None,
    last_round_no: int = 999,
    modify_elo: bool = False,
    stdev: float | None = None,
    update_fixtures: bool | None = True,
    is_european_league: bool | None = False,
    sorting_order: list[str] | None = None,
    reverse: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Build historical standings table after at most n rounds."""

    date_str = elo_date.replace('-', '')
    if not os.path.exists(f"data/elo/{date_str}.csv"):
        download_elo_data(elo_date)

    if country_code_elo is not None:
        elo_df = get_api_teams_and_elo_from_clubelo(elo_date, country_code_elo)
        if stdev is not None:
            elo_df['Elo'] = elo_df['Elo'] + elo_df['Elo'].apply(
                lambda x: random.gauss(0, stdev)
            ).round().astype(int)
    else:
        elo_df = get_data_from_regression(country_code_api)

    if update_fixtures:
        api_get_fixtures_for_league(league_id, season)

    fixtures = read_fixtures(league_id, season, is_european_league)

    fixture_teams = set()
    for fixture in fixtures:
        home_team = fixture['teams']['home']['name']
        fixture_teams.add(home_team)

    elo_dict = {row['Club']: row['Elo'] for _, row in elo_df.iterrows()}
    elo_dict = {k: v for k, v in elo_dict.items() if k in fixture_teams}

    missing_teams = fixture_teams - set(elo_dict.keys())
    if len(missing_teams) > 0:
        print('The following teams are missing from the mapping file:')
        for team in missing_teams:
            print(team)
        raise Exception(
            'Please update the mapping file (fixtures column) - missing ELO ratings.'
        )

    fixtures_filtered = []
    for fixture in fixtures:
        round_str = int(fixture['league']['round'].split(' ')[-1])
        if (round_str > last_round_no) or (
            fixture['fixture']['status']['long'] != 'Match Finished'
        ):
            continue
        if str(fixture['fixture']['id']) == '1396000':  # duplicate
            continue

        fixtures_filtered.append(fixture)

    fixtures_matrix = _populate_fixtures_matrix_from_historical(
        fixture_teams, fixtures_filtered, elo_dict, modify_elo
    )

    standings_df = build_table_from_fixtures_matrix(
        fixtures_matrix, sorting_order=sorting_order, reverse=reverse, elo_dict=elo_dict
    )

    return standings_df, fixtures_matrix


def _populate_fixtures_matrix_from_historical(
    teams: set,
    fixtures: list,
    elo_dict: dict,
    modify_elo: bool = False,
) -> dict[str, dict]:
    """Populate a fixtures matrix from historical (finished) fixtures.

    Optionally updates elo_dict in-place if modify_elo is True.
    """
    fixtures_matrix = _init_fixtures_matrix(list(teams))

    for fixture in fixtures:
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        home_goals = fixture['goals']['home']
        away_goals = fixture['goals']['away']

        fixtures_matrix[home_team][away_team] = (home_goals, away_goals)

        if modify_elo:
            outcome = _outcome_from_goals(home_goals, away_goals)
            elo_diff = _compute_elo_difference(elo_dict[home_team], elo_dict[away_team])
            delta = _compute_elo_delta(elo_diff, outcome)
            elo_dict[home_team] += delta
            elo_dict[away_team] -= delta

    return fixtures_matrix


def _populate_fixtures_matrix_from_simulation(
    league_table: dict,
    fixtures: list,
    fixtures_matrix: dict[str, dict],
    modify_elo: bool = False,
) -> dict[str, dict]:
    """Simulate future fixtures and populate the fixtures matrix.

    Uses Elo-based probabilities to generate results.
    Optionally updates Elo ratings in league_table in-place.
    """
    for fixture in fixtures:
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']

        home_elo = league_table[home_team].elo
        away_elo = league_table[away_team].elo

        pH, pD, pA = _compute_match_probabilities(home_elo, away_elo)
        result = random.choices(['home_win', 'away_win', 'draw'], [pH, pA, pD])[0]

        if result == 'home_win':
            fixtures_matrix[home_team][away_team] = (2, 0)
        elif result == 'away_win':
            fixtures_matrix[home_team][away_team] = (0, 2)
        elif result == 'draw':
            fixtures_matrix[home_team][away_team] = (1, 1)

        if modify_elo:
            outcome = (
                1.0 if result == 'home_win' else (0.0 if result == 'away_win' else 0.5)
            )
            _update_elo(league_table, home_team, away_team, outcome)

    return fixtures_matrix


def simulate_season_after_n_rounds(
    league_id: str,
    season: str,
    standings_df: pd.DataFrame,
    fixtures: dict | None = None,
    reverse: bool = False,
    round_to_overwrite_with_sims_from: int = 999,
    modify_elo_in_sim: bool = False,
    is_european_league: bool | None = False,
    sorting_order: list[str] | None = None,
    fixtures_matrix: dict[dict] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Simulate the rest of the season after n rounds."""

    elo_dict = {row['Club']: row['Elo'] for _, row in standings_df.iterrows()}

    if fixtures is None:
        fixtures = read_fixtures(league_id, season, is_european_league)

    league_table = df_to_dict_of_teams(standings_df)

    if fixtures_matrix is None:
        fixtures_matrix = _init_fixtures_matrix(list(league_table.keys()))

    fixtures_filtered = []
    for fixture in fixtures:
        round_str = int(fixture['league']['round'].split(' ')[-1])
        if (round_str <= round_to_overwrite_with_sims_from) and (
            fixture['fixture']['status']['long'] == 'Match Finished'
        ):
            continue
        fixtures_filtered.append(fixture)

    fixtures_matrix = _populate_fixtures_matrix_from_simulation(
        league_table,
        fixtures_filtered,
        fixtures_matrix,
        modify_elo=modify_elo_in_sim,
    )

    standings_df = build_table_from_fixtures_matrix(
        fixtures_matrix, sorting_order, reverse, elo_dict=elo_dict, _compute_h2h=True
    )

    return standings_df


def run_full_table_sims(
    league_id: str,
    season: str,
    country_code_elo: str | None,
    country_code_api: str | None,
    elo_date: str | None,
    number_of_sims: int,
    reverse: bool = False,
    last_round_for_standings: int = 999,
    round_to_overwrite_with_sims_from: int = 999,
    modify_elo_in_sim: bool = False,
    modify_elo_retro: bool = False,
    stdev: float | None = None,
    standings_df: pd.DataFrame | None = None,
    update_fixtures: bool | None = True,
    is_european_league: bool | None = False,
    sorting_order: list[str] | None = None,
    fixtures_matrix: dict[dict] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run full table simulations of the season after n rounds."""

    if standings_df is None or fixtures_matrix is None:
        standings_df, fixtures_matrix = (
            build_historical_standings_table_after_at_most_n_rounds(
                league_id,
                season,
                country_code_elo,
                country_code_api,
                elo_date,
                last_round_for_standings,
                modify_elo_retro,
                stdev,
                update_fixtures,
                is_european_league,
            )
        )

    xpts = {k: 0 for k in standings_df['Club'].tolist()}

    main_results_df = standings_df[['Club', 'Elo']].copy()
    main_results_df.set_index('Club', inplace=True)
    for place in range(1, standings_df.shape[0] + 1):
        main_results_df[place] = 0

    fixtures = read_fixtures(league_id, season, is_european_league)

    for _ in tqdm(range(number_of_sims)):
        new_standings_df = standings_df.copy()

        if stdev is not None and stdev != 0:
            new_standings_df['Elo'] = new_standings_df['Elo'] + [
                round(random.gauss(0, stdev)) for _ in range(new_standings_df.shape[0])
            ]

        winners_df = simulate_season_after_n_rounds(
            league_id,
            season,
            new_standings_df,
            fixtures,
            reverse,
            round_to_overwrite_with_sims_from,
            modify_elo_in_sim,
            is_european_league,
            sorting_order,
            fixtures_matrix,
        )

        for i in range(winners_df.shape[0]):
            club = winners_df.iloc[i]['Club']
            main_results_df.at[club, i + 1] += 1
            points = winners_df.iloc[i]['Points']
            xpts[club] += points

    xpts_df = pd.DataFrame(list(xpts.items()), columns=['Club', 'Expected Points'])
    df = pd.merge(main_results_df, xpts_df, on='Club', how='inner')
    df.rename(columns={'Expected Points': 'xPts'}, inplace=True)

    df.fillna(0, inplace=True)

    df['xPts'] = df['xPts'].apply(lambda x: round(x / number_of_sims, 2))

    if is_european_league:
        df['Top 8'] = df.apply(
            lambda row: round(
                sum([row[i] for i in range(1, 9)]) / number_of_sims * 100, 1
            ),
            axis=1,
        )
        df['9 - 24'] = df.apply(
            lambda row: round(
                sum([row[i] for i in range(9, 25)]) / number_of_sims * 100, 1
            ),
            axis=1,
        )
        df['25 - 36'] = df.apply(
            lambda row: round(
                sum([row[i] for i in range(25, 37)]) / number_of_sims * 100, 1
            ),
            axis=1,
        )

    for place in range(1, standings_df.shape[0] + 1):
        df[place] = df[place].apply(lambda x: round(x / number_of_sims * 100, 1))

    df.sort_values(by=['xPts'], ascending=False, inplace=True)

    if is_european_league:
        df = df[
            ['Club', 'Elo', 'xPts', 'Top 8', '9 - 24', '25 - 36']
            + list(range(1, standings_df.shape[0] + 1))
        ]
    else:
        df = df[['Club', 'Elo', 'xPts'] + list(range(1, standings_df.shape[0] + 1))]

    df.reset_index(drop=True, inplace=True)
    df.index += 1
    print(f'{number_of_sims} simulations')
    if reverse:
        print('Reverse: TRUE')
    Path('data/sims').mkdir(parents=True, exist_ok=True)
    df.to_excel(
        f'data/sims/{league_id}_{season}_{datetime.today().strftime("%Y-%m-%d")}_full_table.xlsx',
        index=False,
    )
    return df


def simulate_odds(
    standings_df,
    league_id: int,
    season: int,
    is_european_league: bool,
    round_no: int,
    **kwargs,
) -> dict:
    """Simulate odds for a given league, season and round."""

    fixtures = read_fixtures(league_id, season, is_european_league)

    results = []

    for fixture in fixtures:
        round_str = int(fixture['league']['round'].split(' ')[-1])
        if round_str != round_no:
            continue

        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']

        home_elo = standings_df[standings_df['Club'] == home_team]['Elo'].values[0]
        away_elo = standings_df[standings_df['Club'] == away_team]['Elo'].values[0]

        pH, pD, pA = _compute_match_probabilities(home_elo, away_elo)

        odds_home = round(1 / pH, 2)
        odds_draw = round(1 / pD, 2)
        odds_away = round(1 / pA, 2)

        results.append(
            {
                'Home Team': home_team,
                'Away Team': away_team,
                'Odds H': odds_home,
                'Odds D': odds_draw,
                'Odds A': odds_away,
            }
        )

    results = pd.DataFrame(results)
    return results
