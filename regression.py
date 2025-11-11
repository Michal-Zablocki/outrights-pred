from pathlib import Path

from typing import Optional

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

matplotlib.interactive(False)

OPTA_DATE = '2025-11-07'
ELO_DATE = '2025-11-03'
MAPPING_FILEPATH = 'teams_mapping/team_names.xlsx'

COUNTRY_DICT = {
    'Austria': 'AUT',
    'Belgium': 'BEL',
    'Brazil': 'BRA',
    'Bulgaria': 'BUL',
    'Croatia': 'CRO',
    'Czechia': 'CZE',
    'Denmark': 'DEN',
    'England': 'ENG',
    'France': 'FRA',
    'Germany': 'GER',
    'Greece': 'GRE',
    'Hungary': 'HUN',
    'Italy': 'ITA',
    'Netherlands': 'NED',
    'Norway': 'NOR',
    'Poland': 'POL',
    'Portugal': 'POR',
    'Romania': 'ROM',
    'Scotland': 'SCO',
    'Serbia': 'SRB',
    'Slovenia': 'SLO',
    'Spain': 'ESP',
    'Sweden': 'SWE',
    'Switzerland': 'SUI',
    'Turkey': 'TUR',
}


def concat_opta_csvs(input_date: str) -> None:
    """Concats all opta csv files for a given date into a single csv file."""
    input_date = input_date.replace('-', '')
    source_files = Path('data/opta/raw/').glob(f'*_{input_date}.csv')
    main_df = pd.DataFrame()
    for file in source_files:
        df = pd.read_csv(file)
        country = file.stem.split('_')[0].title()
        if country not in COUNTRY_DICT.keys():
            print(
                f'Warning: country {country} not in COUNTRY_DICT, skipping file {file}'
            )
        try:
            df['Country'] = COUNTRY_DICT[country]
        except KeyError:
            df['Country'] = country
        main_df = pd.concat([main_df, df], ignore_index=True)

    main_df.rename(columns=lambda x: x.title().strip(), inplace=True)
    main_df.rename(columns={'Rank': 'Country Rank'}, inplace=True)

    Path('data/transformed/opta/').mkdir(parents=True, exist_ok=True)
    main_df.to_csv(f'data/transformed/opta/{input_date}.csv', index=False)


def load_opta_ratings(input_date: str) -> pd.DataFrame:
    """
    Loads Opta ratings from an Excel file.
    Scraped manually from https://theanalyst.com/articles/who-are-the-best-football-team-in-the-world-opta-power-rankings
    Screenshot => OCR conversion in Gemini to a CSV => Concatenation of multiple sheets into a single file
    Returns a DataFrame with team ratings and country codes.
    """
    opta_df = pd.read_csv(f'data/transformed/opta/{input_date.replace("-", "")}.csv')

    opta_df = opta_df[['Team', 'Rating', 'Country']].copy()

    opta_df.dropna(inplace=True)

    print(opta_df.head())
    print(opta_df.dtypes)

    opta_df['Team'] = opta_df['Team'].apply(lambda x: str(x).title().strip())

    opta_df.drop_duplicates(subset=['Team'], keep='first', inplace=True)

    # opta_df.sort_values(by='Rating', ascending=False, inplace=True)
    opta_df.reset_index(drop=True, inplace=True)

    opta_df.to_csv(
        f'data/transformed/opta/{input_date.replace("-", "")}_v2.csv', index=False
    )

    return opta_df


def get_opta_country_codes(opta_df: pd.DataFrame) -> list[str]:
    """Gets unique country codes from the Opta DataFrame."""
    return opta_df['Country'].unique().tolist()


def get_elo_ratings(
    opta_country_codes: list[str], file_path: Optional[str] = None
) -> pd.DataFrame:
    """Loads ELO ratings and filters data matched with Opta country codes."""
    if file_path is None:
        file_path = f'data/elo/{ELO_DATE.replace("-", "")}.csv'
    elo_df = pd.read_csv(file_path)

    elo_df = elo_df[elo_df['Country'].isin(opta_country_codes)]
    elo_df['Elo'] = elo_df['Elo'].apply(lambda x: round(x, 2))
    elo_df.drop(columns=['Rank'], inplace=True)
    elo_df['Club'] = elo_df['Club'].apply(lambda x: x.title().strip())
    elo_df.reset_index(drop=True, inplace=True)

    return elo_df


def get_map_df() -> pd.DataFrame:
    """Loads the team name mapping DataFrame."""
    map_df = pd.read_excel(MAPPING_FILEPATH)
    map_df = map_df[['ELO_name', 'Opta_name']]
    map_df.dropna(inplace=True)
    map_df['Opta_name'] = map_df['Opta_name'].apply(lambda x: x.title().strip())
    map_df.dropna(subset=['ELO_name'], inplace=True)
    return map_df


def sanitize_elo_df(map_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps ELO team names to other names.
    Raises an error if there are teams in the ELO DataFrame which are absent from the other ELO rating dataframe - should be very rare (?)
    """
    elo_matched_df = pd.merge(
        elo_df, map_df, left_on='Club', right_on='ELO_name', how='inner'
    )
    print(f'{elo_matched_df.shape[0]} ELO teams matched.')
    elo_unmatched_df = elo_df[~elo_df['Club'].isin(elo_matched_df['Club'])].copy()
    elo_unmatched_df.sort_values(
        by=['Country', 'Elo'], ascending=[True, False], inplace=True
    )
    if not elo_unmatched_df.empty:
        print(
            f"{elo_unmatched_df.shape[0]} unmatched ELO teams found, see transformed/elo/elo_unmatched.csv:"
        )
        Path('data/transformed/elo').mkdir(parents=True, exist_ok=True)
        elo_unmatched_df.to_csv('data/transformed/elo/elo_unmatched.csv', index=False)
    return elo_matched_df


def sanitize_opta_df(map_df: pd.DataFrame, opta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps Opta team names other team names.
    Returns a DataFrame with matched and unmatched teams.
    (It's normal to have unmatched teams here, especially for non-European and lower leagues.)
    """
    opta_matched_df = pd.merge(
        opta_df, map_df, left_on='Team', right_on='Opta_name', how='inner'
    )
    print(f'{opta_matched_df.shape[0]} Opta teams matched.')
    opta_unmatched_df = opta_df[~opta_df['Team'].isin(opta_matched_df['Team'])].copy()
    opta_unmatched_df.sort_values(
        by=['Country', 'Rating'], ascending=[True, False], inplace=True
    )
    if not opta_unmatched_df.empty:
        print(
            f"{opta_unmatched_df.shape[0]} unmatched Opta teams found, see transformed/opta/opta_unmatched.csv:"
        )
        Path('data/transformed/opta').mkdir(parents=True, exist_ok=True)
        opta_unmatched_df.to_csv(
            'data/transformed/opta/opta_unmatched.csv', index=False
        )
    opta_unmatched_df.rename(columns={'Team': 'Opta_name'}, inplace=True)
    return opta_matched_df, opta_unmatched_df


def get_final_df(elo_matched_df: pd.DataFrame, opta_df: pd.DataFrame) -> pd.DataFrame:
    """Merges ELO and Opta DataFrames on country and team names."""
    merged_df = pd.merge(
        elo_matched_df,
        opta_df,
        left_on=['Country', 'Opta_name'],
        right_on=['Country', 'Team'],
        how='inner',
    )
    merged_df = merged_df[
        ['ELO_name', 'Opta_name', 'Country', 'Level', 'Elo', 'Rating']
    ]

    merged_df.to_csv('data/transformed/merged_df_nonfiltered.csv', index=False)

    # IMPORTANT: outlier removal

    merged_df = merged_df[(merged_df['Rating'] >= 60.0) & (merged_df['Elo'] >= 1200.0)]

    merged_df.to_csv('data/transformed/merged_df.csv', index=False)

    return merged_df


def run_regression(
    merged_df: pd.DataFrame, country_to_ignore: str, level_to_ignore: int
) -> LinearRegression:
    """Runs linear regression on the merged DataFrame."""
    if country_to_ignore and level_to_ignore:
        merged_df = merged_df[
            (merged_df['Level'] != level_to_ignore)
            | (merged_df['Country'] != country_to_ignore)
        ]

    print(merged_df.shape[0])

    # IMPORTANT: remove bottom outliers

    X = merged_df['Rating']
    y = merged_df['Elo']

    print(merged_df.shape[0])
    print(merged_df.head())

    lr = LinearRegression().fit(X.values.reshape(-1, 1), y.values)
    print(lr.score(X.values.reshape(-1, 1), y.values))
    print(f'y = {lr.coef_[0]} * x + {lr.intercept_}')

    return lr


def predict_elo(merged_df: pd.DataFrame, lr: LinearRegression) -> float:
    """Predicts ELO ratings using the regression model."""
    X = merged_df['Rating']
    merged_df['predicted_Elo'] = lr.predict(X.values.reshape(-1, 1))
    merged_df['predicted_Elo'] = merged_df['predicted_Elo'].apply(lambda x: round(x, 2))
    merged_df['error'] = merged_df['Elo'] - merged_df['predicted_Elo']
    merged_df['error'] = merged_df['error'].apply(
        lambda x: round(x) if pd.notnull(x) else x
    )
    merged_df.to_csv('data/reg_results.csv', index=False)
    return merged_df


def plot_results(merged_df: pd.DataFrame, lr: LinearRegression) -> None:
    """Plots the regression results."""
    plt.figure(figsize=(12, 8))
    plt.scatter(merged_df['Rating'], merged_df['Elo'], color='blue', label='Actual ELO')
    plt.plot(
        merged_df['Rating'],
        lr.coef_[0] * merged_df['Rating'] + lr.intercept_,
        color='red',
        label='Regression Line',
    )
    plt.xlabel('Opta Rating')
    plt.ylabel('ELO Rating')
    plt.title('ELO vs Opta Rating Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/regression_plot.png')


def main_regression(
    country_to_ignore: Optional[str] = None, level_to_ignore: Optional[int] = None
):
    """Main function to run the regression."""
    concat_opta_csvs(OPTA_DATE)
    opta_df = load_opta_ratings(OPTA_DATE)
    opta_country_codes = get_opta_country_codes(opta_df)
    elo_df = get_elo_ratings(opta_country_codes)
    map_df = get_map_df()
    elo_matched_df = sanitize_elo_df(map_df, elo_df)
    opta_matched_df, opta_unmatched_df = sanitize_opta_df(map_df, opta_df)
    final_df = get_final_df(elo_matched_df, opta_df)

    lr = run_regression(final_df, country_to_ignore, level_to_ignore)

    plot_results(final_df, lr)

    new_df = pd.concat(
        [final_df, opta_unmatched_df],
        ignore_index=True,
    )
    print(new_df.head())
    predicted_df = predict_elo(new_df, lr)


if __name__ == "__main__":
    main_regression()
