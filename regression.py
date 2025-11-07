import pandas as pd
from sklearn.linear_model import LinearRegression


OPTA_FILEPATH = 'data/opta/ratings_2025-11-03.xlsx'
ELO_FILEPATH = 'data/elo/20251103.csv'
MAPPING_FILEPATH = 'teams_mapping/team_names.xlsx'

COUNTRY_DICT = {
    'Brazil': 'BR',
    'England': 'ENG',
    'Poland': 'POL',
    'Portugal': 'POR',
    'Spain': 'ESP',
}


def get_opta_ratings() -> pd.DataFrame:
    opta_dfs = pd.read_excel(OPTA_FILEPATH, sheet_name=None)

    opta_df = pd.DataFrame()

    for sheet_name, df in opta_dfs.items():
        df['Country'] = sheet_name
        opta_df = pd.concat([opta_df, df], ignore_index=True)

    opta_df = opta_df[['Team', 'Rating', 'Country']]
    opta_df.sort_values(by='Rating', ascending=False, inplace=True)
    opta_df.reset_index(drop=True, inplace=True)
    opta_df['Country_Code'] = opta_df['Country'].map(COUNTRY_DICT)
    return opta_df


def get_opta_country_codes(opta_df) -> list:
    return opta_df['Country_Code'].unique().tolist()


def get_elo_ratings(opta_country_codes) -> pd.DataFrame:
    elo_df = pd.read_csv(ELO_FILEPATH)

    elo_df = elo_df[elo_df['Country'].isin(opta_country_codes)]
    elo_df['Elo'] = elo_df['Elo'].apply(lambda x: round(x, 2))
    elo_df.drop(columns=['Rank'], inplace=True)
    elo_df.reset_index(drop=True, inplace=True)

    return elo_df


def get_map_df() -> pd.DataFrame:
    map_df = pd.read_excel(MAPPING_FILEPATH)
    map_df = map_df[['ELO_name', 'Opta_name']]
    map_df.dropna(subset=['ELO_name'], inplace=True)
    return map_df


def sanitize_elo_df(map_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    elo_matched_df = pd.merge(
        elo_df, map_df, left_on='Club', right_on='ELO_name', how='inner'
    )
    elo_unmatched_df = elo_df[~elo_df['Club'].isin(elo_matched_df['Club'])]
    if not elo_unmatched_df.empty:
        print("Unmatched ELO teams:")
        print(elo_unmatched_df)
        raise ValueError(
            "There are unmatched ELO teams. Please update the mapping file."
        )
    return elo_matched_df


def sanitize_opta_df(map_df: pd.DataFrame, opta_df: pd.DataFrame) -> pd.DataFrame:
    opta_matched_df = pd.merge(
        opta_df, map_df, left_on='Team', right_on='Opta_name', how='inner'
    )
    opta_unmatched_df = opta_df[~opta_df['Team'].isin(opta_matched_df['Team'])].copy()
    print("Unmatched Opta teams:")
    print(opta_unmatched_df.head(20))
    opta_unmatched_df.rename(columns={'Team': 'Opta_name'}, inplace=True)
    opta_unmatched_df.drop(columns=['Country'], inplace=True)
    return opta_matched_df, opta_unmatched_df


def get_final_df(elo_matched_df: pd.DataFrame, opta_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(
        elo_matched_df,
        opta_df,
        left_on=['Country', 'Opta_name'],
        right_on=['Country_Code', 'Team'],
        how='inner',
    )
    merged_df = merged_df[
        ['ELO_name', 'Opta_name', 'Country_Code', 'Level', 'Elo', 'Rating']
    ]
    return merged_df


def run_regression(
    merged_df: pd.DataFrame, country_to_ignore: str, level_to_ignore: int
) -> LinearRegression:
    if country_to_ignore and level_to_ignore:
        merged_df = merged_df[
            (merged_df['Level'] != level_to_ignore)
            | (merged_df['Country_Code'] != country_to_ignore)
        ]
    X = merged_df['Rating']
    y = merged_df['Elo']
    lr = LinearRegression().fit(X.values.reshape(-1, 1), y.values)
    print(lr.score(X.values.reshape(-1, 1), y.values))
    print(f'y = {lr.coef_[0]} * x + {lr.intercept_}')

    return lr


def predict_elo(merged_df, lr) -> float:
    X = merged_df['Rating']
    merged_df['predicted_Elo'] = lr.predict(X.values.reshape(-1, 1))
    merged_df['predicted_Elo'] = merged_df['predicted_Elo'].apply(lambda x: round(x, 2))
    merged_df['error'] = merged_df['Elo'] - merged_df['predicted_Elo']
    merged_df['error'] = merged_df['error'].apply(
        lambda x: round(x) if pd.notnull(x) else x
    )
    merged_df.to_csv('data/reg_results.csv', index=False)
    return merged_df


def main_regression(country_to_ignore=None, level_to_ignore=None):
    opta_df = get_opta_ratings()
    opta_country_codes = get_opta_country_codes(opta_df)
    elo_df = get_elo_ratings(opta_country_codes)
    map_df = get_map_df()
    elo_matched_df = sanitize_elo_df(map_df, elo_df)
    opta_matched_df, opta_unmatched_df = sanitize_opta_df(map_df, opta_df)
    final_df = get_final_df(elo_matched_df, opta_df)

    lr = run_regression(final_df, country_to_ignore, level_to_ignore)

    new_df = pd.concat(
        [final_df, opta_unmatched_df],
        ignore_index=True,
    )
    predicted_df = predict_elo(new_df, lr)


if __name__ == "__main__":
    main_regression()
