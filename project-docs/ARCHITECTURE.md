# Architecture

## Purpose

Football (soccer) outright odds simulator. Uses Elo ratings and Monte Carlo simulation to predict season-end outcomes (champion, top-N, relegation) and compare them against bookmaker odds to identify value bets.

---

## Python Modules

### `etl.py`

Data ingestion and retrieval. Imported by `sims_engine.py`.

**Data ingestion**

- `download_elo_data()` — fetches club Elo ratings from `api.clubelo.com` and saves to `data/elo/<date>.csv`
- `api_get_fixtures_for_league()` / `api_get_leagues()` — calls API-Football (RapidAPI) and saves responses to `data/fixtures_api/`
- `read_fixtures()` — reads a fixture JSON from `data/fixtures_api/fixtures_<league_id>_<year>.json`

**Team/Elo retrieval**

- `get_api_teams_and_elo_from_clubelo()` — loads Elo from a dated CSV, maps names via `teams_mapping/team_names.xlsx`, returns `(Club, Elo)` DataFrame
- `get_data_from_regression()` — loads predicted Elo from `data/reg_results.csv` (output of the regression pipeline) and maps names
- `find_latest_elo_file()` — returns the most recent file in `data/elo/`
- `find_league_id()` — looks up a league ID from `data/fixtures_api/leagues.json` by country code and name

---

### `sims_engine.py`

Simulation engine. Imported by all active notebooks via `from sims_engine import *`. Imports ETL helpers from `etl.py`.

**Core standings function**

- `build_table_from_fixtures_matrix()` — the single source of truth for standings. Takes a `fixtures_matrix` dict (home→away→(goals_h, goals_a)) and computes the full league table with tie-breaking (H2H, goal difference, etc.). All other functions populate the matrix; this one turns it into a sorted DataFrame.

**Fixtures matrix population**

- `_populate_fixtures_matrix_from_historical()` — populates the matrix from finished API fixtures; optionally updates Elo ratings in-place
- `_populate_fixtures_matrix_from_simulation()` — simulates unplayed matches using Elo win probabilities (HFA + draw parameter ν) and writes results into the matrix; optionally updates Elo

**Orchestration**

- `build_historical_standings_table_after_at_most_n_rounds()` — loads Elo + fixtures, filters to round N, calls `_populate_fixtures_matrix_from_historical` then `build_table_from_fixtures_matrix`
- `simulate_season_after_n_rounds()` — single full-season simulation from a standings snapshot; calls `_populate_fixtures_matrix_from_simulation` then `build_table_from_fixtures_matrix`
- `simulate_odds()` — returns match-level win/draw/loss odds for a given round

**Multi-sim runner**

- `run_full_table_sims()` — runs N simulations, outputs a probability for every finishing position per team; saves to `data/sims/`

**Internal helpers**

- `_compute_elo_difference()` — Elo difference adjusted for home field advantage (single source for the HFA formula)
- `_compute_match_probabilities()` — (p_home, p_draw, p_away) from two Elo ratings
- `_compute_elo_delta()` — Elo rating change for a single match outcome
- `_update_elo()` — applies Elo delta to both teams in a league_table dict
- `_outcome_from_goals()` — maps (home_goals, away_goals) to Elo outcome (1/0.5/0)
- `_init_fixtures_matrix()` — creates an empty matrix for a set of teams

**Constants** (tuned via historical optimisation): `HFA = 0.045`, `K_FACTOR = 20`, `NU = 1.65`

---

### `regression.py`

Bridges Opta Power Rankings and ClubElo ratings.

- `concat_opta_csvs()` — merges per-country raw Opta CSVs from `data/opta/raw/` into `data/transformed/opta/<date>.csv`
- `load_opta_ratings()` — loads a transformed Opta CSV
- `get_elo_ratings()` — loads a dated ClubElo CSV filtered by country codes
- `get_map_df()` — loads the ELO↔Opta name mapping from `teams_mapping/team_names.xlsx`
- `sanitize_elo_df()` / `sanitize_opta_df()` — join/validate both datasets; writes unmatched teams to `data/transformed/elo/` and `data/transformed/opta/`
- `get_final_df()` — inner-joins Elo and Opta on (Country, team name); saves `data/transformed/merged_df.csv`
- `run_regression()` — linear regression of Opta rating → Elo; prints R²
- `predict_elo()` — applies the model to every team, saves `data/reg_results.csv`

---

### `book_alignment.py`

Standalone script (also runnable as a module). Back-calculates Elo ratings that best match bookmaker match odds.

- `load_book_odds()` — reads `data/football-data/<COUNTRY>.csv` (football-data.co.uk format), normalises odds, computes implied probabilities and bookmaker spread
- `elo_error_function()` — KL divergence between bookmaker implied probs and Elo-model probs
- `optimize_elo_ratings()` — minimises KL divergence over all team Elos (L-BFGS-B or differential evolution)
- Saves result to `data/optimized_elos_<season>.json`

---

## Notebooks

### Active / production

| Notebook                                  | Purpose                                                                                                                       |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `working_version.ipynb`                 | Main simulation runner; calls `run_full_table_sims` for multiple leagues/outrights                                            |
| `working_version_ekstraklasa_v2.ipynb`  | Same workflow scoped to the Polish Ekstraklasa                                                                                |
| `working_version_books_odds.ipynb`      | Variant that substitutes Elo from `book_alignment.py` output (`data/optimized_elos_*.json`) instead of ClubElo/regression |
| `hist_opta_elo_linear_regression.ipynb` | Runs the regression pipeline from `regression.py`; produces `data/reg_results.csv`                                        |
| `bet_backtest_elo_only.ipynb`           | Backtests betting strategy using ClubElo ratings against historical results                                                   |
| `bet_backtest_elo_with_opta.ipynb`      | Same backtest but with Opta-derived (regression) Elo                                                                          |
| `bet_backtest_v1.ipynb`                 | Earlier backtest; loads concatenated football-data.co.uk CSVs                                                                 |
| `kelly.ipynb`                           | Kelly criterion calculations on personal bet history (`priv_bets.csv`); outputs `priv_bets2.csv`                          |
| `odds_comparison.ipynb`                 | Ad-hoc comparison of model odds vs bookmaker odds                                                                             |
| `model_v2.ipynb`                        | Experimental model iteration                                                                                                  |

### Reference / samples

`sample_notebooks/` — frozen snapshots of working notebooks from earlier dates, kept for reference.

---

## Data Directory

```
data/
├── elo/                        ClubElo snapshots:  <YYYYMMDD>.csv
│                                 columns: Rank, Club, Country, Level, Elo
├── fixtures_api/               API-Football responses
│   ├── leagues.json
│   └── fixtures_<id>_<year>.json
├── opta/
│   ├── raw/                    Per-country Opta CSVs
│   └── (transformed via regression.py)
├── football-data/              football-data.co.uk per-country CSVs
│                                 used by book_alignment.py and bet_backtest notebooks
├── transformed/                Intermediate artefacts
│   ├── opta/<date>.csv         Concatenated Opta ratings
│   ├── elo/elo_unmatched.csv   Elo teams not found in name map
│   └── merged_df.csv           Elo + Opta joined dataset
├── reg_results.csv             Regression output (Opta → predicted Elo per team)
├── optimized_elos_<season>.json  Book-aligned Elo per team
└── sims/                       Full-table simulation outputs (xlsx)
```

---

## Reference / Config Files

| File                                   | Role                                                                                                     |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `teams_mapping/team_names.xlsx`      | Master name map:`ELO_name` ↔ `Opta_name` ↔ `fixtures_name` ↔ `football-data_name` per country |
| `priv_bets.csv` / `priv_bets2.csv` | Personal bet ledger; input/output for `kelly.ipynb`                                                    |
| `.env`                               | `X-RapidAPI-Key` for API-Football calls                                                                |
| `TODO.py`                            | Informal backlog                                                                                         |

---

## Key Data Flow

```
ClubElo API ──────────────────────────────────────────────────────┐
                                                                   ▼
Opta (manual scrape → OCR) → regression.py → reg_results.csv ──► etl.py
                                                                   │
API-Football ──► fixtures_api/*.json ──────────────────────────►  │
                                                                   ▼
football-data.co.uk CSVs ──► book_alignment.py              sims_engine.py
                                  └── optimized_elos_*.json ──►   │  run_full_table_sims
                                                                   ▼
                                                        Simulation output
                                                   (odds, % finish per position)
```
