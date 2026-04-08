from datetime import datetime
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests


load_dotenv()

API_TOKEN = os.getenv('X-RapidAPI-Key')


def read_fixtures(
    league_id: str, season: str, is_european_league: bool | None = False, **kwargs
) -> list[dict]:
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


def download_elo_data(date=None, **kwargs) -> None:
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


def api_get_fixtures_for_league(league_id: str, season: str, **kwargs) -> None:
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


def find_league_id(country_code: str, league_name: str, **kwargs) -> str | None:
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


def get_api_teams_and_elo_from_clubelo(
    date: str, country_code: str | None, **kwargs
) -> pd.DataFrame:
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


def get_data_from_regression(country_code: str | None, **kwargs) -> pd.DataFrame:
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
