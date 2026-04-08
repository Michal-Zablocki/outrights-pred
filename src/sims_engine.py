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

from etl import (
    api_get_fixtures_for_league,
    download_elo_data,
    get_api_teams_and_elo_from_clubelo,
    get_data_from_regression,
    read_fixtures,
)


load_dotenv()

HFA_MULT = 1.045  # home field advantage; optimized
HFA_FLAT = 65  # flat home field advantage in Elo points; alternative to HFA multiplier
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

_RESULTS = [(2, 0), (1, 1), (0, 2)]  # home_win, draw, away_win
_ELO_DIFF_MIN = -1000
_ELO_DIFF_MAX = 1000


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


def _probs_from_elo_diff(elo_diff: float) -> tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win) from Elo difference."""
    p_win_base = 1 / (1 + math.pow(10, -elo_diff / 400))
    denom = 1 + NU * p_win_base * (1 - p_win_base)
    pH = p_win_base / denom
    pA = (1 - p_win_base) / denom
    pD = 1 - pH - pA
    return pH, pD, pA


_PROB_TABLE = np.array(
    [
        _probs_from_elo_diff(elo_diff)
        for elo_diff in range(_ELO_DIFF_MIN, _ELO_DIFF_MAX + 1)
    ]
)


def _lookup_probs(home_elo: float, away_elo: float):
    elo_diff = int(round(home_elo + HFA_FLAT - away_elo))
    idx = np.clip(elo_diff - _ELO_DIFF_MIN, 0, _ELO_DIFF_MAX - _ELO_DIFF_MIN)
    return _PROB_TABLE[idx]


def get_sorting_order_for_country_code(country_code: str) -> list[str]:
    """Get sorting order based on country code."""

    if country_code in SORTING_ORDERS:
        return SORTING_ORDERS[country_code]
    else:
        return SORTING_ORDERS['POL']


def _compute_elo_difference(home_elo: float, away_elo: float) -> float:
    """Elo difference adjusted for home field advantage."""
    return home_elo + HFA_FLAT - away_elo


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

    ascending_setting = [
        True if col in ['H2H', 'Random order'] else False for col in sorting_order
    ]

    standings_df = standings_df.sort_values(
        by=sorting_order, ascending=ascending_setting
    ).reset_index(drop=True)

    if reverse:
        standings_df = standings_df.iloc[::-1].reset_index(drop=True)

    standings_df.index += 1

    return standings_df


def df_to_dict_of_teams(standings_df: pd.DataFrame) -> dict:
    """Convert a DataFrame to a dictionary of TeamInTable."""

    cols = standings_df[
        [
            'Club',
            'Elo',
            'Matches played',
            'Wins',
            'Draws',
            'Losses',
            'Goals for',
            'Goals against',
            'Goal difference',
            'Goals away',
            'Points',
        ]
    ].to_numpy()

    league_table = {row[0]: TeamInTable(*row) for row in cols}

    return league_table


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
    point_deductions: dict[str, int] | None = None,
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

    elo_dict = dict(zip(elo_df['Club'], elo_df['Elo']))
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
        fixtures_matrix,
        sorting_order=sorting_order,
        reverse=reverse,
        elo_dict=elo_dict,
        point_deductions=point_deductions,
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

    Example for one fixture with pH=0.55, pD=0.24, pA=0.21:
        cumul[i] = [0.55, 0.79, 1.00]
        u[i] = 0.62
        → [0.62 >= 0.55, 0.62 >= 0.79, 0.62 >= 1.00] = [True, False, False]
        → sum = 1 → draw
    """
    if not modify_elo:
        home_teams = [f['teams']['home']['name'] for f in fixtures]
        away_teams = [f['teams']['away']['name'] for f in fixtures]

        probs = np.array(
            [
                _lookup_probs(league_table[home].elo, league_table[away].elo)
                for home, away in zip(home_teams, away_teams)
            ]
        )

        cumul = np.cumsum(probs, axis=1)
        u = np.random.random(len(fixtures))
        result_indices = (u[:, None] < cumul).argmax(axis=1)

        for i, (h, a) in enumerate(zip(home_teams, away_teams)):
            fixtures_matrix[h][a] = _RESULTS[result_indices[i]]
    else:
        for fixture in fixtures:
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']

            probs = _lookup_probs(
                league_table[home_team].elo, league_table[away_team].elo
            )
            u = np.random.random()
            result_i = int((u >= np.cumsum(probs)).sum())

            fixtures_matrix[home_team][away_team] = _RESULTS[result_i]

            outcome = [1.0, 0.5, 0.0][result_i]
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
    point_deductions: dict[str, int] | None = None,
    fixtures_to_simulate: list | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Simulate the rest of the season after n rounds.

    Returns the full-season standings including already-played games.
    If fixtures_matrix is not provided it is reconstructed from finished fixtures.
    If fixtures_to_simulate is provided, the filtering step is skipped (pre-computed by caller).
    """

    elo_dict = dict(zip(standings_df['Club'], standings_df['Elo']))

    if fixtures is None:
        fixtures = read_fixtures(league_id, season, is_european_league)

    league_table = df_to_dict_of_teams(standings_df)

    if fixtures_matrix is None:
        historical = [
            f
            for f in fixtures
            if int(f['league']['round'].split(' ')[-1])
            <= round_to_overwrite_with_sims_from
            and f['fixture']['status']['long'] == 'Match Finished'
            and str(f['fixture']['id']) != '1396000'
        ]
        fixtures_matrix = _populate_fixtures_matrix_from_historical(
            set(league_table.keys()), historical, {}, modify_elo=False
        )

    if fixtures_to_simulate is None:
        fixtures_to_simulate = []
        for fixture in fixtures:
            round_str = int(fixture['league']['round'].split(' ')[-1])
            if (round_str <= round_to_overwrite_with_sims_from) and (
                fixture['fixture']['status']['long'] == 'Match Finished'
            ):
                continue
            fixtures_to_simulate.append(fixture)

    fixtures_matrix = _populate_fixtures_matrix_from_simulation(
        league_table,
        fixtures_to_simulate,
        fixtures_matrix,
        modify_elo=modify_elo_in_sim,
    )

    standings_df = build_table_from_fixtures_matrix(
        fixtures_matrix,
        sorting_order,
        reverse,
        elo_dict=elo_dict,
        _compute_h2h=True,
        point_deductions=point_deductions,
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
    point_deductions: dict[str, int] | None = None,
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
                point_deductions=point_deductions,
            )
        )

    xpts = {k: 0 for k in standings_df['Club'].tolist()}

    main_results_df = standings_df[['Club', 'Elo']].copy()
    main_results_df.set_index('Club', inplace=True)
    for place in range(1, standings_df.shape[0] + 1):
        main_results_df[place] = 0

    fixtures = read_fixtures(league_id, season, is_european_league)

    fixtures_to_simulate = [
        f
        for f in fixtures
        if not (
            int(f['league']['round'].split(' ')[-1])
            <= round_to_overwrite_with_sims_from
            and f['fixture']['status']['long'] == 'Match Finished'
        )
    ]

    for _ in tqdm(range(number_of_sims)):
        new_standings_df = standings_df.copy()

        if stdev is not None and stdev != 0:
            new_standings_df['Elo'] = new_standings_df['Elo'] + np.round(
                np.random.normal(0, stdev, size=new_standings_df.shape[0])
            ).astype(int)

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
            point_deductions=point_deductions,
            fixtures_to_simulate=fixtures_to_simulate,
        )

        for row in winners_df.itertuples():
            club = row.Club
            place = row.Index
            points = row.Points
            main_results_df.at[club, place] += 1
            xpts[club] += points

    xpts_df = pd.DataFrame(list(xpts.items()), columns=['Club', 'Expected Points'])
    df = pd.merge(main_results_df, xpts_df, on='Club', how='inner')
    df.rename(columns={'Expected Points': 'xPts'}, inplace=True)

    df.fillna(0, inplace=True)

    df['xPts'] = df['xPts'].apply(lambda x: round(x / number_of_sims, 2))

    if is_european_league:
        df['Top 8'] = df[[i for i in range(1, 9)]].sum(axis=1) * 100 / number_of_sims
        df['9 - 24'] = df[[i for i in range(9, 25)]].sum(axis=1) * 100 / number_of_sims
        df['25 - 36'] = (
            df[[i for i in range(25, 37)]].sum(axis=1) * 100 / number_of_sims
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
    Path('data/sims').mkdir(parents=True, exist_ok=True)
    df.to_excel(
        f'data/sims/{league_id}_{season}_{datetime.today().strftime("%Y-%m-%d")}_full_table.xlsx',
        index=False,
    )
    return df


def get_top_n_odds(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Extract top-N finish probability and decimal odds from a run_full_table_sims result.

    Args:
        df: Output of run_full_table_sims (integer position columns contain % probabilities).
        top_n: Number of finishing positions to consider (e.g. 1 = champion, 6 = top 6).

    Returns:
        DataFrame with columns Club, Elo, xPts, Top N %, Top N Odds sorted by probability desc.
    """
    n_teams = df.shape[0]
    effective_n = min(top_n, n_teams)
    result = df[['Club', 'Elo', 'xPts']].copy()
    result[f'Top {top_n} %'] = (
        df[[i for i in range(1, effective_n + 1)]].sum(axis=1).round(1)
    )
    result[f'Top {top_n} Odds'] = result[f'Top {top_n} %'].apply(
        lambda x: round(100 / x, 2) if x > 0 else 'N/A'
    )
    result = result.sort_values(by=[f'Top {top_n} %'], ascending=False).reset_index(
        drop=True
    )
    result.index += 1
    return result


def run_top_n_sims(
    top_n: list[int],
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
    fixtures_matrix: dict | None = None,
    point_deductions: dict[str, int] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run full-table sims and return top-N finish probabilities and odds for each N in top_n.

    Args:
        top_n: List of N values to compute (e.g. [1, 2, 6]).  For reverse=True these
               represent the bottom-N positions (e.g. relegation spots).
        All other args are forwarded to run_full_table_sims unchanged.

    Returns:
        DataFrame with Club, Elo, xPts and then Top N % / Top N Odds columns for each N,
        sorted by the first element of top_n descending.
        Also saved as data/sims/multiple_sims_league_{id}_top_{n1}_{n2}_..._{date}.csv.
    """
    full_table = run_full_table_sims(
        league_id=league_id,
        season=season,
        country_code_elo=country_code_elo,
        country_code_api=country_code_api,
        elo_date=elo_date,
        number_of_sims=number_of_sims,
        reverse=reverse,
        last_round_for_standings=last_round_for_standings,
        round_to_overwrite_with_sims_from=round_to_overwrite_with_sims_from,
        modify_elo_in_sim=modify_elo_in_sim,
        modify_elo_retro=modify_elo_retro,
        stdev=stdev,
        standings_df=standings_df,
        update_fixtures=update_fixtures,
        is_european_league=is_european_league,
        sorting_order=sorting_order,
        fixtures_matrix=fixtures_matrix,
        point_deductions=point_deductions,
    )

    result = full_table[['Club', 'Elo', 'xPts']].copy()
    for n in top_n:
        odds_df = get_top_n_odds(full_table, n)
        result = result.merge(
            odds_df[['Club', f'Top {n} %', f'Top {n} Odds']], on='Club', how='left'
        )

    sort_col = f'Top {top_n[0]} %'
    result = result.sort_values(by=[sort_col], ascending=False).reset_index(drop=True)
    result.index += 1

    Path('data/sims').mkdir(parents=True, exist_ok=True)
    top_n_str = '_'.join(str(n) for n in top_n)
    reverse_str = 'reverse_' if reverse else ''
    result.to_csv(
        f'data/sims/multiple_sims_league_{league_id}_top_{top_n_str}_{reverse_str}{datetime.today().strftime("%Y-%m-%d")}.csv',
        index=False,
    )

    print(f'Top N: {top_n}')
    if reverse:
        print('Reverse: TRUE')

    return result


def simulate_odds(
    standings_df: pd.DataFrame,
    league_id: str,
    season: str,
    is_european_league: bool,
    round_no: int,
    **kwargs,
) -> pd.DataFrame:
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

        pH, pD, pA = _lookup_probs(home_elo, away_elo)

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
