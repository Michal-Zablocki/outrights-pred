import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
from typing import Optional

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


load_dotenv()

API_TOKEN = os.getenv('X-RapidAPI-Key')
HFA = 60  # home field advantage
K_FACTOR = 20  # Elo rating system K-factor
NU = 1.5  # draw parameter


def read_fixtures(
    league_id: str, season: str, is_european_league: Optional[bool] = False
) -> dict:
    """Read fixtures from a local JSON file."""

    with open(f"data/fixtures_api/fixtures_{league_id}_{season}.json", "r") as f:
        fixtures = json.load(f)['response']
        if is_european_league:
            fixtures = [
                fixture
                for fixture in fixtures
                if fixture['league']['round'].startswith('League Stage')
            ]
    return fixtures


def save_round_divide(num, den, precision=2) -> int | float:
    """Safe division with rounding. Returns -1 if division by zero."""

    if den == 0:
        return -1
    else:
        return round(num / den, precision)


def download_elo_data(date=None) -> None:
    """Note: apparently only European clubs are included."""

    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')
    df = pd.read_csv(f"http://api.clubelo.com/{date}")
    Path("data/elo").mkdir(parents=True, exist_ok=True)
    df = df[['Rank', 'Club', 'Country', 'Level', 'Elo']]
    date = date.replace('-', '')
    df.to_csv(f"data/elo/{date}.csv", index=False)


def api_get_leagues() -> None:
    """Get current leagues from the API and save to a file."""

    url = "https://api-football-v1.p.rapidapi.com/v3/leagues"

    params = {"current": "true"}

    headers = {
        "X-RapidAPI-Key": API_TOKEN,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f'Error: {response.status_code}')

    if response.json()['paging']['total'] != 1:
        raise Exception("Error: multiple pages of leagues")

    Path("data/fixtures_api").mkdir(parents=True, exist_ok=True)
    with open("data/fixtures_api/leagues.json", "w") as f:
        json.dump(response.json(), f)


def api_get_fixtures_for_league(league_id: str, season: str) -> None:
    """Get fixtures for a given league and season from the API and save to a file."""
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    params = {"league": league_id, "season": season}

    headers = {
        "X-RapidAPI-Key": API_TOKEN,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.json()['results'] == 0:
        raise Exception("No results found.")

    if response.json()['paging']['total'] != 1:
        raise Exception("Error: multiple pages of leagues")

    Path("data/fixtures_api").mkdir(parents=True, exist_ok=True)
    with open(f"data/fixtures_api/fixtures_{league_id}_{season}.json", "w") as f:
        json.dump(response.json(), f)


def api_get_standings_for_league() -> None:
    pass


def find_latest_elo_file() -> str:
    """Find the latest ELO file in the data/elo directory."""

    elo_files = os.listdir("data/elo")
    return f'data/elo/{sorted(elo_files)[-1]}'


# def get_team_names_from_elo(elo_country_code: str) -> None:
#     """Get team names from the latest ELO file for a given country code; 1st league tier."""
#     df = pd.read_csv(find_latest_elo_file())
#     df = df[(df['Country'] == elo_country_code) & (df['Level'] == 1)]
#     names = sorted(df['Club'].tolist())
#     df = pd.DataFrame(names)
#     df.to_excel('tmp_team_names_elo.xlsx', index=False)


# def get_team_names_from_api_dump(path: str) -> None:
#     """Get team names from an API fixtures dump."""
#     with open(path, "r") as f:
#         fixtures = json.load(f)
#     names = sorted(
#         set([fixture['teams']['home']['name'] for fixture in fixtures['response']])
#     )
#     df = pd.DataFrame(names)
#     df.to_excel('tmp_team_names_api.xlsx', index=False)


def find_league_id(country_code: str, league_name: str) -> str:
    """Find league ID based on country code and league name."""

    with open("data/fixtures_api/leagues.json", "r") as f:
        leagues = json.load(f)
        for league in leagues['response']:
            if (
                league['country']['code'] == country_code
                and league['league']['type'] == 'League'
                and league['league']['name'] == league_name
            ):
                return league['league']['id']


def get_api_teams_and_elo_from_clubelo(date: str, country_code: str) -> pd.DataFrame:
    """Get teams and their ELO ratings from ClubElo data for a given date and country code, 1st league tier."""

    date = date.replace('-', '')
    elo_data = pd.read_csv(f"data/elo/{date}.csv")

    if country_code not in ['UCL', 'UEL', 'ECL']:
        elo_data = elo_data[
            (elo_data['Country'] == country_code) & (elo_data['Level'] == 1)
        ]

    elo_data['Elo'] = elo_data['Elo'].apply(round)

    team_map_df = pd.read_excel('teams_mapping/team_names.xlsx')

    # missing_teams = set(elo_data['Club'].tolist()) - set(
    #     team_map_df['ELO_name'].tolist()
    # )

    # if len(missing_teams) > 0:
    #     print(
    #         f"The following teams are missing in team_names.xlsx: {missing_teams}. Please add them."
    #     )
    #     raise Exception("Missing teams in team_names.xlsx")

    # team_map = {
    #     row['ELO_name']: row['fixtures_name'] for _, row in team_map_df.iterrows()
    # }

    # elo_data['Club'] = elo_data['Club'].apply(lambda x: team_map[x])

    # elo_data.reset_index(drop=True, inplace=True)

    # return elo_data[['Club', 'Elo']]

    df = pd.merge(
        elo_data, team_map_df, left_on='Club', right_on='ELO_name', how='inner'
    )

    df = df[['fixtures_name', 'Elo']]
    df.dropna(inplace=True)

    df.drop_duplicates(inplace=True)

    df.rename(columns={'fixtures_name': 'Club'}, inplace=True)

    return df


def get_data_from_regression(country_code: Optional[str]) -> pd.DataFrame:
    """Get team names and predicted ELO from regression results, optionally filtered by country code."""

    team_map_df = pd.read_excel('teams_mapping/team_names.xlsx')

    elo_df = pd.read_csv('data/reg_results.csv')

    if country_code is not None:
        elo_df = elo_df[elo_df['Country'] == country_code]
        team_map_df = team_map_df[team_map_df['Country_code'] == country_code]

    team_map_df['Opta_name'] = team_map_df['Opta_name'].str.title().str.strip()
    team_map_df = team_map_df[['fixtures_name', 'Opta_name']]

    elo_df = elo_df[['Opta_name', 'predicted_Elo']]

    df = pd.merge(elo_df, team_map_df, on='Opta_name', how='inner')

    df = df.rename(columns={'fixtures_name': 'Club', 'predicted_Elo': 'Elo'})

    return df


def build_historical_standings_table_after_at_most_n_rounds(
    league_id: str,
    season: str,
    country_code_elo: Optional[str],
    country_code_api: Optional[str],
    elo_date: Optional[str],
    last_round_no: int = 999,
    modify_elo: bool = False,
    stdev: Optional[float] = None,
    update_fixtures: Optional[bool] = True,
    is_european_league: Optional[bool] = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    elo_dict = {row['Club']: row['Elo'] for _, row in elo_df.iterrows()}
    points_dict = {row['Club']: 0 for _, row in elo_df.iterrows()}
    games_played_dict = {row['Club']: 0 for _, row in elo_df.iterrows()}

    fixture_teams = set()
    for fixture in fixtures:
        home_team = fixture['teams']['home']['name']
        fixture_teams.add(home_team)

    elo_dict = {k: v for k, v in elo_dict.items() if k in fixture_teams}
    points_dict = {k: v for k, v in points_dict.items() if k in fixture_teams}
    games_played_dict = {
        k: v for k, v in games_played_dict.items() if k in fixture_teams
    }

    missing_teams = fixture_teams - set(elo_dict.keys())
    if len(missing_teams) > 0:

        print('The following teams are missing from the mapping file:')
        for team in missing_teams:
            print(team)
        print('Please update the mapping file (fixtures column).')
        raise Exception("Missing ELO ratings for some teams.")

    for fixture in fixtures:
        round_str = int(fixture['league']['round'].split(' ')[-1])
        if (round_str > last_round_no) or (
            fixture['fixture']['status']['long'] != 'Match Finished'
        ):
            continue

        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']

        home_goals = fixture['goals']['home']
        away_goals = fixture['goals']['away']

        if home_goals > away_goals:
            points_dict[home_team] += 3
        elif home_goals < away_goals:
            points_dict[away_team] += 3
        else:
            points_dict[home_team] += 1
            points_dict[away_team] += 1

        if modify_elo:
            home_elo = elo_dict[home_team]
            away_elo = elo_dict[away_team]

            elo_difference = home_elo - away_elo + HFA

            if home_goals > away_goals:
                elo_delta = (
                    1 - 1 / (1 + math.pow(10, -elo_difference / 400))
                ) * K_FACTOR
            elif home_goals < away_goals:
                elo_delta = (
                    0 - 1 / (1 + math.pow(10, -elo_difference / 400))
                ) * K_FACTOR
            else:
                elo_delta = (
                    1 / 2 - 1 / (1 + math.pow(10, -elo_difference / 400))
                ) * K_FACTOR

            elo_dict[home_team] += elo_delta
            elo_dict[away_team] -= elo_delta

        games_played_dict[home_team] += 1
        games_played_dict[away_team] += 1

    points_df = pd.DataFrame(points_dict.items(), columns=['Club', 'Points'])
    elo_df = pd.DataFrame(elo_dict.items(), columns=['Club', 'Elo'])
    games_played_df = pd.DataFrame(
        games_played_dict.items(), columns=['Club', 'Games played']
    )

    standings_df = pd.merge(elo_df, points_df, on='Club', how='inner')
    standings_df = pd.merge(standings_df, games_played_df, on='Club', how='inner')

    standings_df = standings_df.sort_values(by=['Points'], ascending=False).reset_index(
        drop=True
    )
    standings_df.index += 1

    return standings_df


def simulate_season_after_n_rounds(
    league_id: str,
    season: str,
    standings_df: pd.DataFrame,
    fixtures: Optional[dict] = None,
    reverse: bool = False,
    round_to_overwrite_with_sims_from: int = 999,
    modify_elo_in_sim: bool = False,
    is_european_league: Optional[bool] = False,
) -> pd.DataFrame:
    """Simulate the rest of the season after n rounds."""

    if fixtures is None:
        fixtures = read_fixtures(league_id, season, is_european_league)

    elo_dict = {row['Club']: row['Elo'] for _, row in standings_df.iterrows()}
    points_dict = {row['Club']: row['Points'] for _, row in standings_df.iterrows()}
    games_played_dict = {
        row['Club']: row['Games played'] for _, row in standings_df.iterrows()
    }

    for fixture in fixtures:
        round_str = int(fixture['league']['round'].split(' ')[-1])
        if (round_str <= round_to_overwrite_with_sims_from) and (
            fixture['fixture']['status']['long'] == 'Match Finished'
        ):
            continue

        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']

        home_elo = elo_dict[home_team]
        away_elo = elo_dict[away_team]

        elo_difference = home_elo - away_elo + HFA

        p_win_base = 1 / (1 + math.pow(10, -elo_difference / 400))
        denom = 1 + NU * p_win_base * (1 - p_win_base)
        pH = p_win_base / denom
        pA = (1 - p_win_base) / denom
        pD = 1 - pH - pA

        result = random.choices(['home_win', 'away_win', 'draw'], [pH, pA, pD])[0]

        if result == 'home_win':
            points_dict[home_team] += 3
        elif result == 'away_win':
            points_dict[away_team] += 3
        elif result == 'draw':
            points_dict[home_team] += 1
            points_dict[away_team] += 1

        if modify_elo_in_sim:
            if result == 'home_win':
                elo_delta = (
                    1 - 1 / (1 + math.pow(10, -elo_difference / 400))
                ) * K_FACTOR
            elif result == 'away_win':
                elo_delta = (
                    0 - 1 / (1 + math.pow(10, -elo_difference / 400))
                ) * K_FACTOR
            else:
                elo_delta = (
                    1 / 2 - 1 / (1 + math.pow(10, -elo_difference / 400))
                ) * K_FACTOR

            elo_dict[home_team] += elo_delta
            elo_dict[away_team] -= elo_delta

        games_played_dict[home_team] += 1
        games_played_dict[away_team] += 1

    points_df = pd.DataFrame(points_dict.items(), columns=['Club', 'Points'])
    elo_df = pd.DataFrame(elo_dict.items(), columns=['Club', 'Elo'])
    games_played_dict = pd.DataFrame(
        games_played_dict.items(), columns=['Club', 'Games played']
    )

    season_standings_df = pd.merge(points_df, elo_df, on='Club', how='inner')
    season_standings_df = pd.merge(
        season_standings_df, games_played_dict, on='Club', how='inner'
    )

    season_standings_df = season_standings_df.sort_values(
        by=['Points'], ascending=reverse
    )

    return season_standings_df


def run_multiple_sims(
    league_id: str,
    season: str,
    country_code_elo: Optional[str],
    country_code_api: Optional[str],
    elo_date: Optional[str],
    number_of_sims: int,
    number_of_winning_places: int,
    reverse: bool = False,
    last_round_for_standings: int = 999,
    round_to_overwrite_with_sims_from: int = 999,
    modify_elo_in_sim: bool = False,
    modify_elo_retro: bool = False,
    stdev: Optional[float] = None,
    standings_df: Optional[pd.DataFrame] = None,
    update_fixtures: Optional[bool] = True,
    is_european_league: Optional[bool] = False,
    range_from: Optional[int] = None,
    range_to: Optional[int] = None,
) -> pd.DataFrame:
    """Run multiple simulations of the season after n rounds."""

    if standings_df is None:
        standings_df = build_historical_standings_table_after_at_most_n_rounds(
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

    winners_with_losing_all_tbs = dict()
    winners_with_random_tbs = dict()

    xpts = {k: 0 for k in standings_df['Club'].tolist()}

    fixtures = read_fixtures(league_id, season, is_european_league)

    for _ in tqdm(range(number_of_sims)):
        new_standings_df = copy.deepcopy(standings_df)

        if stdev is not None:
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
        )

        winners_df['Tiebreaking order'] = np.random.permutation(winners_df.shape[0])

        winners_df = winners_df.sort_values(
            by=['Points', 'Tiebreaking order'], ascending=reverse
        ).reset_index(drop=True)

        if (
            winners_df.iloc[number_of_winning_places - 1]['Points']
            == winners_df.iloc[number_of_winning_places]['Points']
        ):
            tie_points = winners_df.iloc[number_of_winning_places - 1]['Points']

            for i in range(number_of_winning_places):
                if winners_df.iloc[i]['Points'] != tie_points:
                    try:
                        winners_with_losing_all_tbs[winners_df.iloc[i]['Club']] += 1
                    except KeyError:
                        winners_with_losing_all_tbs[winners_df.iloc[i]['Club']] = 1
        else:
            for i in range(number_of_winning_places):
                try:
                    winners_with_losing_all_tbs[winners_df.iloc[i]['Club']] += 1
                except KeyError:
                    winners_with_losing_all_tbs[winners_df.iloc[i]['Club']] = 1

        for i in range(number_of_winning_places):
            try:
                winners_with_random_tbs[winners_df.iloc[i]['Club']] += 1
            except KeyError:
                winners_with_random_tbs[winners_df.iloc[i]['Club']] = 1

        for i in range(winners_df.shape[0]):
            club = winners_df.iloc[i]['Club']
            points = winners_df.iloc[i]['Points']
            xpts[club] += points

    random_tbs_df = pd.DataFrame(
        list(winners_with_random_tbs.items()), columns=['Club', 'RTB Wins']
    )
    losing_all_tbs_df = pd.DataFrame(
        list(winners_with_losing_all_tbs.items()), columns=['Club', 'LTB Wins']
    )
    xpts_df = pd.DataFrame(list(xpts.items()), columns=['Club', 'Expected Points'])

    df = pd.merge(random_tbs_df, losing_all_tbs_df, on='Club', how='outer')
    df = pd.merge(df, xpts_df, on='Club', how='inner')
    df = df[['Club', 'Expected Points', 'RTB Wins', 'LTB Wins']]
    df.rename(columns={'Expected Points': 'xPts'}, inplace=True)

    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['xPts'] = df['xPts'].apply(lambda x: round(x / number_of_sims, 2))
    df['RTB Wins'] = df['RTB Wins'].astype(int)
    df['LTB Wins'] = df['LTB Wins'].astype(int)
    df['% RTB winrate'] = round(df['RTB Wins'] / number_of_sims * 100, 1)
    df['% LTB winrate'] = round(df['LTB Wins'] / number_of_sims * 100, 1)
    df['Exp. RTB odds'] = df.apply(
        lambda row: save_round_divide(number_of_sims, row['RTB Wins']), axis=1
    )
    df['Exp. LTB odds'] = df.apply(
        lambda row: save_round_divide(number_of_sims, row['LTB Wins']), axis=1
    )
    df = df.sort_values(by=['RTB Wins'], ascending=False).reset_index(drop=True)
    df.index += 1
    print(f'{number_of_sims} simulations')
    print(f'{number_of_winning_places} winning places')
    if reverse:
        print('Reverse: TRUE')
    return df


def run_full_table_sims(
    league_id: str,
    season: str,
    country_code_elo: Optional[str],
    country_code_api: Optional[str],
    elo_date: Optional[str],
    number_of_sims: int,
    number_of_winning_places: int,
    reverse: bool = False,
    last_round_for_standings: int = 999,
    round_to_overwrite_with_sims_from: int = 999,
    modify_elo_in_sim: bool = False,
    modify_elo_retro: bool = False,
    stdev: Optional[float] = None,
    standings_df: Optional[pd.DataFrame] = None,
    update_fixtures: Optional[bool] = True,
    is_european_league: Optional[bool] = False,
) -> pd.DataFrame:
    """Run full table simulations of the season after n rounds."""

    if standings_df is None:
        standings_df = build_historical_standings_table_after_at_most_n_rounds(
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

    xpts = {k: 0 for k in standings_df['Club'].tolist()}

    main_results_df = standings_df[['Club', 'Elo']].copy()
    main_results_df.set_index('Club', inplace=True)
    for place in range(1, standings_df.shape[0] + 1):
        main_results_df[place] = 0

    fixtures = read_fixtures(league_id, season, is_european_league)

    for _ in tqdm(range(number_of_sims)):
        new_standings_df = copy.deepcopy(standings_df)

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
        )

        winners_df['Tiebreaking order'] = np.random.permutation(winners_df.shape[0])

        winners_df = winners_df.sort_values(
            by=['Points', 'Tiebreaking order'], ascending=reverse
        ).reset_index(drop=True)

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
    else:
        df['Top 4'] = df.apply(
            lambda row: round(
                sum([row[i] for i in range(1, 5)]) / number_of_sims * 100, 1
            ),
            axis=1,
        )

    for place in range(1, standings_df.shape[0] + 1):
        df[place] = df[place].apply(lambda x: round(x / number_of_sims * 100, 1))

    df.sort_values(by=['xPts'], ascending=False, inplace=True)

    if is_european_league:
        df = df[
            ['Club', 'Elo', 'xPts', 'Top 8', '9 - 24']
            + list(range(1, standings_df.shape[0] + 1))
        ]
    else:
        df = df[
            ['Club', 'Elo', 'xPts', 'Top 4'] + list(range(1, standings_df.shape[0] + 1))
        ]

    df.reset_index(drop=True, inplace=True)
    df.index += 1
    print(f'{number_of_sims} simulations')
    print(f'{number_of_winning_places} winning places')
    if reverse:
        print('Reverse: TRUE')
    Path('data/sims').mkdir(parents=True, exist_ok=True)
    df.to_excel(f'data/sims/{league_id}_{season}_full_table_sim.xlsx', index=False)
    return df


# standings_df = build_historical_standings_table_after_at_most_n_rounds(
#     league_id=109,
#     season=2025,
#     country_code_elo=None,
#     country_code_api='POL',
#     elo_date='2025-11-10',
#     update_fixtures=False,
# )
