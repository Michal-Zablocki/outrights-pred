import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
from typing import Optional

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


load_dotenv()

API_TOKEN = os.getenv('X-RapidAPI-Key')
HFA = 60  # home field advantage
K_FACTOR = 20  # Elo rating system K-factor
NU = 1.5  # draw parameter


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
    url = "https://api-football-v1.p.rapidapi.com/v3/leagues"

    params = {"current": "true"}

    headers = {
        "X-RapidAPI-Key": API_TOKEN,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=params)
    print(response.status_code)

    if response.json()['paging']['total'] != 1:
        raise Exception("Error: multiple pages of leagues")

    Path("data/api").mkdir(parents=True, exist_ok=True)
    with open("data/api/leagues.json", "w") as f:
        json.dump(response.json(), f)


def api_get_fixtures_for_league(league_id: str, season: str) -> None:
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

    Path("data/api").mkdir(parents=True, exist_ok=True)
    with open(f"data/api/fixtures_{league_id}_{season}.json", "w") as f:
        json.dump(response.json(), f)


def api_get_standings_for_league() -> None:
    pass


def find_latest_elo_file() -> str:
    elo_files = os.listdir("data/elo")
    return f'data/elo/{sorted(elo_files)[-1]}'


def get_team_names_from_elo(elo_country_code: str) -> None:
    df = pd.read_csv(find_latest_elo_file())
    df = df[(df['Country'] == elo_country_code) & (df['Level'] == 1)]
    names = sorted(df['Club'].tolist())
    df = pd.DataFrame(names)
    df.to_excel('tmp_team_names_elo.xlsx', index=False)


def get_team_names_from_api_dump(path: str) -> None:
    with open(path, "r") as f:
        fixtures = json.load(f)
    names = sorted(
        set([fixture['teams']['home']['name'] for fixture in fixtures['response']])
    )
    df = pd.DataFrame(names)
    df.to_excel('tmp_team_names_api.xlsx', index=False)


def find_league_id(country_code: str, league_name: str) -> str:
    with open("data/api/leagues.json", "r") as f:
        leagues = json.load(f)
        for league in leagues['response']:
            if (
                league['country']['code'] == country_code
                and league['league']['type'] == 'League'
                and league['league']['name'] == league_name
            ):
                return league['league']['id']


def get_api_teams_and_elo_from_clubelo(date: str, country_code: str) -> pd.DataFrame:

    date = date.replace('-', '')
    elo_data = pd.read_csv(f"data/elo/{date}.csv")

    elo_data = elo_data[
        (elo_data['Country'] == country_code) & (elo_data['Level'] == 1)
    ]

    elo_data['Elo'] = elo_data['Elo'].apply(round)

    team_map_df = pd.read_excel('teams_mapping/team_names.xlsx')

    missing_teams = set(elo_data['Club'].tolist()) - set(
        team_map_df['ELO_name'].tolist()
    )
    if len(missing_teams) > 0:
        print(
            f"The following teams are missing in team_names.xlsx: {missing_teams}. Please add them."
        )
        raise Exception("Missing teams in team_names.xlsx")

    team_map = {
        row['ELO_name']: row['fixtures_name'] for _, row in team_map_df.iterrows()
    }

    elo_data['Club'] = elo_data['Club'].apply(lambda x: team_map[x])

    elo_data.reset_index(drop=True, inplace=True)

    return elo_data[['Club', 'Elo']]


def get_data_from_regression(country_code: str) -> pd.DataFrame:
    team_map_df = pd.read_excel('teams_mapping/team_names.xlsx')
    team_map_df = team_map_df[team_map_df['Country_code'] == country_code]
    team_map_df = team_map_df[['fixtures_name', 'Opta_name']]

    elo_df = pd.read_csv('data/reg_results.csv')
    elo_df = elo_df[elo_df['Country_Code'] == country_code]
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
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if country_code_elo is not None:
        elo_df = get_api_teams_and_elo_from_clubelo(elo_date, country_code_elo)
        if stdev is not None:
            elo_df['Elo'] = elo_df['Elo'] + elo_df['Elo'].apply(
                lambda x: random.gauss(0, stdev)
            ).round().astype(int)
    else:
        elo_df = get_data_from_regression(country_code_api)

    api_get_fixtures_for_league(league_id, season)

    with open(f"data/api/fixtures_{league_id}_{season}.json", "r") as f:
        fixtures = json.load(f)['response']

    elo_dict = {row['Club']: row['Elo'] for _, row in elo_df.iterrows()}
    points_dict = {row['Club']: 0 for _, row in elo_df.iterrows()}
    games_played_dict = {row['Club']: 0 for _, row in elo_df.iterrows()}

    fixture_teams = set()
    for fixture in fixtures:
        home_team = fixture['teams']['home']['name']
        fixture_teams.add(home_team)

    missing_teams = fixture_teams - set(elo_dict.keys())
    if len(missing_teams) > 0:
        print(
            f"The following teams are missing ELO ratings: {missing_teams}. Please update the mapping file or provide ELO data for these teams."
        )
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
    reverse: bool = False,
    round_to_overwrite_with_sims_from: int = 999,
    modify_elo_in_sim: bool = False,
) -> pd.DataFrame:
    with open(f"data/api/fixtures_{league_id}_{season}.json", "r") as f:
        fixtures = json.load(f)['response']

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
) -> pd.DataFrame:
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
        )
    print(standings_df)

    winners = dict()
    number_of_successful_sims = 0

    for _ in tqdm(range(number_of_sims)):
        new_standings_df = copy.deepcopy(standings_df)

        if stdev is not None:
            new_standings_df['Elo'] = new_standings_df['Elo'] + new_standings_df[
                'Elo'
            ].apply(lambda x: random.gauss(0, stdev)).round().astype(int)

        winners_df = simulate_season_after_n_rounds(
            league_id,
            season,
            new_standings_df,
            reverse,
            round_to_overwrite_with_sims_from,
            modify_elo_in_sim,
        )

        if (
            winners_df.iloc[number_of_winning_places - 1]['Points']
            == winners_df.iloc[number_of_winning_places]['Points']
        ):
            continue
        number_of_successful_sims += 1
        for i in range(number_of_winning_places):
            try:
                winners[winners_df.iloc[i]['Club']] += 1
            except KeyError:
                winners[winners_df.iloc[i]['Club']] = 1

    df = pd.DataFrame(list(winners.items()), columns=['Club', 'Wins'])
    df['% winrate'] = round(df['Wins'] / number_of_successful_sims * 100)
    df['Expected odds'] = round(number_of_successful_sims / df['Wins'], 2)
    df = df.sort_values(by=['Wins'], ascending=False).reset_index(drop=True)
    df.index += 1
    print(f'{number_of_successful_sims} simulations')
    print(f'{number_of_winning_places} winning places')
    return df
