# Outrights simulator

##### The idea is to build a tool, utilising various data sources and models already existing, which is more accurate than the bookmakers in prediciting the outcomes which can be wagered on.

##### I do not claim I have succeeded at that - this is only an attempt at doing so.

##### Note: gambling is net negative, mkay.

## 1. Introduction

The most popular bets offered by bookmakers are wagered on the outcome of single games.

More complex bets (e.g. parlays) are characterised by a significantly higher spread.

Bookmakers are generally pretty good at estimating the odds and so it's difficult to be profitable, even theoretically.

Practically, there are also issues of taxation and the pattern of restricting bettors who seem to be profitable.

Now, contrary to what would seem to be the case, I have justification to believe the following two theses:

1) There exist models which can predict the outcomes of games more accurately than what the implied odds are.

2) **Bookmakers are bad at predicting the outright outcomes.**

Therefore, the strategy I have come up with is to build on the outrights according to external models - and hope for the best.

## 2. An example

More seriously, there are 2 kinds of errors that the bookmaker can make.

Let me illustrate it with a simplified example.

There's a football (soccer) team who has 2 games in the season left to play. They'll become a champion only if they win both of these games.

The bookmaker offers odds 2.5 for the team to win the first game and 2.0 to win the second game, meaning 40% and 50% respectively. Let's assume these are independent events.

The bookmaker also offers 6.0 odds for the team to be the champion.

The first kind of error that might have happened is that the bookmaker estimated the strength of the team of its opponents inaccurately. 

Perhaps the team is signifantly stronger and the actual probabilities are 50% and 60% instead.

The second kind of error, which is quite clear, is that the bookmaker's outright odds are better than the odds that would be implied from their own predictions on the outcomes of single games.

If the odds are 2.5 and 2.0 respectively, the implied odds on the team to become the champion should be 5.0.

Anything less than that means a higher spread for the bookmarker, whereas odds better than that allow the punter to hedge profitably.

Using a model to predict the outcomes of all games in the season allows one to potentially exploit the second error.

## 3. How it works

Note: The whole process relies on an external model(s) and a number of **sketchy heuristics**.

The results are bound to be very rough estimations.

The idea is that in cases where there is a very significant disparity between the model's prediction and the bookie's offer following the model's pick is +EV.

The model estimates ELO rating of the teams in a given league and takes into account all games already played in a given season.

It then utilises a popular Monte Carlo method on the Author's supercomputer to run a number of simulations how the season's going to play out.

In more detail:

1. Get details on all games in the season via the API-FOOTBALL.

2. Get ratings of the teams in the league from Opta Power Rankings.

3. Run a linear regression to translate the ratings from Opta into ELO ratings from clubelo.com.

4. Simulate the season a number of times based on the ELO ratings obtained above.

5. Compare the model's predictions to the odds offered by the bookmaker.

The whole process is presented below. There are a couple of steps which require a small amount of manual labor, notably: scraping data from Opta, aligning the teams' names which differ across the data websites and checking the available odds and bets offered by the bookie.

### 3. When it worked in the past - and when it did not 

What follows are my somewhat subjective observations over some time.

I believe that models updating the team's strength after each games - as both Opta and clubelo do - are slightly more sensitive to the cases where either a "weak" team is outperforming or a "strong" team is underpeforming.

An example would be the famous Leicester run in 2015/2016 Premier League season - I am hoping to see the elo-based model suggest that Leicester is being underestimated throughout the season, compared to the bookmakers' odds.

See: https://www.skysports.com/football/news/11712/10261535/premier-league-2015-16-how-the-odds-changed-as-leicester-claimed-the-title for a nice overview of the outright odds for Leicester for that season.


```python
import statistics
```


```python
import pandas as pd
```


```python
%load_ext autoreload
%autoreload 2
```


```python
from helpers import *
```


```python
ELO_DATE = '2015-11-24'
```


```python
NUMBER_OF_SIMS = 10000
```


```python
# download_elo_data(ELO_DATE)
```


```python
# api_get_leagues()
```


```python
find_league_id('GB-ENG', 'Premier League')
```




    39




```python
df = get_api_teams_and_elo_from_clubelo(ELO_DATE, 'ENG')
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Club</th>
      <th>Elo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manchester City</td>
      <td>1869</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manchester United</td>
      <td>1835</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arsenal</td>
      <td>1834</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chelsea</td>
      <td>1797</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tottenham</td>
      <td>1767</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Liverpool</td>
      <td>1755</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Everton</td>
      <td>1728</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Southampton</td>
      <td>1705</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Stoke City</td>
      <td>1692</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Leicester</td>
      <td>1689</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Crystal Palace</td>
      <td>1659</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Swansea</td>
      <td>1647</td>
    </tr>
    <tr>
      <th>12</th>
      <td>West Ham</td>
      <td>1646</td>
    </tr>
    <tr>
      <th>13</th>
      <td>West Brom</td>
      <td>1639</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Watford</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Norwich</td>
      <td>1591</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sunderland</td>
      <td>1570</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Newcastle</td>
      <td>1570</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Bournemouth</td>
      <td>1559</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Aston Villa</td>
      <td>1539</td>
    </tr>
  </tbody>
</table>
</div>




```python
# df2 = get_api_teams_and_elo_from_clubelo('2016-05-20', 'ENG')
# df2.head(20)
```


```python
# elo_drift_df = pd.merge(df, df2, how='inner', on='Club', suffixes=('_before', '_after'))
# elo_drift_df['Elo_Drift'] = elo_drift_df['Elo_after'] - elo_drift_df['Elo_before']
# season_stdev = statistics.stdev(elo_drift_df['Elo_Drift'])
# print(season_stdev)
# elo_drift_df.head(20)
```


```python
standings_df = build_historical_standings_table_after_at_most_n_rounds(league_id=39, season=2015, country_code_elo='ENG', country_code_api='ENG', elo_date=ELO_DATE, last_round_no=13)
standings_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Club</th>
      <th>Elo</th>
      <th>Points</th>
      <th>Games played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Leicester</td>
      <td>1689</td>
      <td>28</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manchester United</td>
      <td>1835</td>
      <td>27</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arsenal</td>
      <td>1834</td>
      <td>26</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manchester City</td>
      <td>1869</td>
      <td>26</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tottenham</td>
      <td>1767</td>
      <td>24</td>
      <td>13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>West Ham</td>
      <td>1646</td>
      <td>21</td>
      <td>13</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Southampton</td>
      <td>1705</td>
      <td>20</td>
      <td>13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Everton</td>
      <td>1728</td>
      <td>20</td>
      <td>13</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Liverpool</td>
      <td>1755</td>
      <td>20</td>
      <td>13</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Crystal Palace</td>
      <td>1659</td>
      <td>19</td>
      <td>13</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Stoke City</td>
      <td>1692</td>
      <td>19</td>
      <td>13</td>
    </tr>
    <tr>
      <th>12</th>
      <td>West Brom</td>
      <td>1639</td>
      <td>17</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Watford</td>
      <td>1602</td>
      <td>16</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Chelsea</td>
      <td>1797</td>
      <td>14</td>
      <td>13</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Swansea</td>
      <td>1647</td>
      <td>14</td>
      <td>13</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Norwich</td>
      <td>1591</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Newcastle</td>
      <td>1570</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sunderland</td>
      <td>1570</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Bournemouth</td>
      <td>1559</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Aston Villa</td>
      <td>1539</td>
      <td>5</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample_season = simulate_season_after_n_rounds(league_id=39, season=2015, standings_df=standings_df, reverse=False, round_to_overwrite_with_sims_from=14)
sample_season.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Club</th>
      <th>Points</th>
      <th>Elo</th>
      <th>Games played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Manchester City</td>
      <td>75</td>
      <td>1869</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manchester United</td>
      <td>72</td>
      <td>1835</td>
      <td>37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arsenal</td>
      <td>72</td>
      <td>1834</td>
      <td>37</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Everton</td>
      <td>64</td>
      <td>1728</td>
      <td>37</td>
    </tr>
    <tr>
      <th>5</th>
      <td>West Ham</td>
      <td>58</td>
      <td>1646</td>
      <td>37</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Stoke City</td>
      <td>57</td>
      <td>1692</td>
      <td>37</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Chelsea</td>
      <td>57</td>
      <td>1797</td>
      <td>37</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Leicester</td>
      <td>56</td>
      <td>1689</td>
      <td>37</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Liverpool</td>
      <td>52</td>
      <td>1755</td>
      <td>37</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Southampton</td>
      <td>52</td>
      <td>1705</td>
      <td>37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tottenham</td>
      <td>51</td>
      <td>1767</td>
      <td>37</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Crystal Palace</td>
      <td>50</td>
      <td>1659</td>
      <td>37</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Swansea</td>
      <td>50</td>
      <td>1647</td>
      <td>37</td>
    </tr>
    <tr>
      <th>11</th>
      <td>West Brom</td>
      <td>44</td>
      <td>1639</td>
      <td>37</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Norwich</td>
      <td>44</td>
      <td>1591</td>
      <td>37</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Watford</td>
      <td>39</td>
      <td>1602</td>
      <td>37</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Aston Villa</td>
      <td>34</td>
      <td>1539</td>
      <td>37</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Bournemouth</td>
      <td>32</td>
      <td>1559</td>
      <td>37</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sunderland</td>
      <td>31</td>
      <td>1570</td>
      <td>37</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Newcastle</td>
      <td>31</td>
      <td>1570</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



After 13 matchdays - 2015-11-24

Official odds: 101.00


```python
results = run_multiple_sims(league_id=39, season=2015, country_code_elo='ENG', country_code_api=None, elo_date=ELO_DATE, number_of_sims=NUMBER_OF_SIMS, number_of_winning_places=1, last_round_for_standings=13, round_to_overwrite_with_sims_from=14)
results.head(20)
```

                     Club   Elo  Points  Games played
    1           Leicester  1689      28            13
    2   Manchester United  1835      27            13
    3             Arsenal  1834      26            13
    4     Manchester City  1869      26            13
    5           Tottenham  1767      24            13
    6            West Ham  1646      21            13
    7         Southampton  1705      20            13
    8             Everton  1728      20            13
    9           Liverpool  1755      20            13
    10     Crystal Palace  1659      19            13
    11         Stoke City  1692      19            13
    12          West Brom  1639      17            13
    13            Watford  1602      16            13
    14            Chelsea  1797      14            13
    15            Swansea  1647      14            13
    16            Norwich  1591      12            13
    17          Newcastle  1570      10            13
    18         Sunderland  1570       9            13
    19        Bournemouth  1559       9            13
    20        Aston Villa  1539       5            13
    

    100%|██████████| 10000/10000 [01:35<00:00, 104.43it/s]

    10000 simulations
    1 winning places
    

    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Club</th>
      <th>RTB Wins</th>
      <th>LTB Wins</th>
      <th>% RTB winrate</th>
      <th>% LTB winrate</th>
      <th>Exp. RTB odds</th>
      <th>Exp. LTB odds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Manchester City</td>
      <td>3448</td>
      <td>3204.0</td>
      <td>34.5</td>
      <td>32.0</td>
      <td>2.90</td>
      <td>3.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manchester United</td>
      <td>3272</td>
      <td>3022.0</td>
      <td>32.7</td>
      <td>30.2</td>
      <td>3.06</td>
      <td>3.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arsenal</td>
      <td>2614</td>
      <td>2418.0</td>
      <td>26.1</td>
      <td>24.2</td>
      <td>3.83</td>
      <td>4.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tottenham</td>
      <td>403</td>
      <td>346.0</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>24.81</td>
      <td>28.90</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Leicester</td>
      <td>111</td>
      <td>92.0</td>
      <td>1.1</td>
      <td>0.9</td>
      <td>90.09</td>
      <td>108.70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Liverpool</td>
      <td>81</td>
      <td>60.0</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>123.46</td>
      <td>166.67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Everton</td>
      <td>32</td>
      <td>23.0</td>
      <td>0.3</td>
      <td>0.2</td>
      <td>312.50</td>
      <td>434.78</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Chelsea</td>
      <td>16</td>
      <td>13.0</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>625.00</td>
      <td>769.23</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Southampton</td>
      <td>10</td>
      <td>9.0</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>1000.00</td>
      <td>1111.11</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Stoke City</td>
      <td>6</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>1666.67</td>
      <td>2000.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>West Ham</td>
      <td>6</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>1666.67</td>
      <td>2500.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Crystal Palace</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000.00</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
</div>



After 23 matchdays - 2016-01-25

Official odds: 9.00


```python
ELO_DATE = '2016-01-25'
results = run_multiple_sims(league_id=39, season=2015, country_code_elo='ENG', country_code_api=None, elo_date=ELO_DATE, number_of_sims=NUMBER_OF_SIMS, number_of_winning_places=1, last_round_for_standings=23, round_to_overwrite_with_sims_from=24)
results.head(20)
```

                     Club   Elo  Points  Games played
    1           Leicester  1742      47            23
    2     Manchester City  1865      44            23
    3             Arsenal  1834      44            23
    4           Tottenham  1795      42            23
    5   Manchester United  1778      37            23
    6            West Ham  1683      36            23
    7           Liverpool  1729      34            23
    8         Southampton  1722      33            23
    9          Stoke City  1706      33            23
    10            Watford  1639      32            23
    11     Crystal Palace  1656      31            23
    12            Everton  1713      29            23
    13            Chelsea  1811      28            23
    14          West Brom  1644      28            23
    15            Swansea  1644      25            23
    16        Bournemouth  1614      25            23
    17            Norwich  1598      23            23
    18          Newcastle  1591      21            23
    19         Sunderland  1589      19            23
    20        Aston Villa  1550      13            23
    

    100%|██████████| 10000/10000 [01:37<00:00, 102.68it/s]

    10000 simulations
    1 winning places
    

    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Club</th>
      <th>RTB Wins</th>
      <th>LTB Wins</th>
      <th>% RTB winrate</th>
      <th>% LTB winrate</th>
      <th>Exp. RTB odds</th>
      <th>Exp. LTB odds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Manchester City</td>
      <td>3941</td>
      <td>3608.0</td>
      <td>39.4</td>
      <td>36.1</td>
      <td>2.54</td>
      <td>2.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arsenal</td>
      <td>3167</td>
      <td>2893.0</td>
      <td>31.7</td>
      <td>28.9</td>
      <td>3.16</td>
      <td>3.46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Leicester</td>
      <td>1989</td>
      <td>1751.0</td>
      <td>19.9</td>
      <td>17.5</td>
      <td>5.03</td>
      <td>5.71</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tottenham</td>
      <td>815</td>
      <td>710.0</td>
      <td>8.2</td>
      <td>7.1</td>
      <td>12.27</td>
      <td>14.08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Manchester United</td>
      <td>77</td>
      <td>56.0</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>129.87</td>
      <td>178.57</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Liverpool</td>
      <td>5</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2000.00</td>
      <td>3333.33</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Southampton</td>
      <td>3</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3333.33</td>
      <td>5000.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>West Ham</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5000.00</td>
      <td>10000.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stoke City</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000.00</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
</div>



After 27 matchdays - 2016-03-01

Official odds: 2.87


```python
ELO_DATE = '2016-03-01'
results = run_multiple_sims(league_id=39, season=2015, country_code_elo='ENG', country_code_api=None, elo_date=ELO_DATE, number_of_sims=NUMBER_OF_SIMS, number_of_winning_places=1, last_round_for_standings=27, round_to_overwrite_with_sims_from=28)
results.head(20)
```

                     Club   Elo  Points  Games played
    1           Leicester  1776      56            27
    2           Tottenham  1844      54            27
    3             Arsenal  1839      51            27
    4     Manchester City  1861      48            27
    5   Manchester United  1799      44            27
    6            West Ham  1697      43            27
    7           Liverpool  1746      41            27
    8         Southampton  1742      40            27
    9          Stoke City  1707      39            27
    10            Watford  1656      37            27
    11            Chelsea  1827      36            27
    12            Everton  1734      35            27
    13          West Brom  1663      35            27
    14     Crystal Palace  1641      32            27
    15        Bournemouth  1621      29            27
    16            Swansea  1644      27            27
    17          Newcastle  1598      25            27
    18            Norwich  1588      24            27
    19         Sunderland  1608      23            27
    20        Aston Villa  1550      16            27
    

    100%|██████████| 10000/10000 [01:35<00:00, 104.63it/s]

    10000 simulations
    1 winning places
    

    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Club</th>
      <th>RTB Wins</th>
      <th>LTB Wins</th>
      <th>% RTB winrate</th>
      <th>% LTB winrate</th>
      <th>Exp. RTB odds</th>
      <th>Exp. LTB odds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Leicester</td>
      <td>4056</td>
      <td>3675.0</td>
      <td>40.6</td>
      <td>36.8</td>
      <td>2.47</td>
      <td>2.72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tottenham</td>
      <td>3823</td>
      <td>3461.0</td>
      <td>38.2</td>
      <td>34.6</td>
      <td>2.62</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arsenal</td>
      <td>1649</td>
      <td>1403.0</td>
      <td>16.5</td>
      <td>14.0</td>
      <td>6.06</td>
      <td>7.13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manchester City</td>
      <td>452</td>
      <td>352.0</td>
      <td>4.5</td>
      <td>3.5</td>
      <td>22.12</td>
      <td>28.41</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Manchester United</td>
      <td>18</td>
      <td>16.0</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>555.56</td>
      <td>625.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Liverpool</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>7</th>
      <td>West Ham</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000.00</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
</div>



After 34 matchdays - 2016-04-19

Official odds: 1.44


```python
ELO_DATE = '2016-04-19'
results = run_multiple_sims(league_id=39, season=2015, country_code_elo='ENG', country_code_api=None, elo_date=ELO_DATE, number_of_sims=NUMBER_OF_SIMS, number_of_winning_places=1, last_round_for_standings=34, round_to_overwrite_with_sims_from=35)
results.head(20)
```

                     Club   Elo  Points  Games played
    1           Leicester  1797      73            34
    2           Tottenham  1847      68            34
    3             Arsenal  1835      63            34
    4     Manchester City  1872      61            34
    5   Manchester United  1785      59            34
    6            West Ham  1724      56            34
    7           Liverpool  1808      55            34
    8         Southampton  1741      51            34
    9          Stoke City  1692      47            34
    10            Chelsea  1796      45            34
    11            Watford  1645      41            34
    12        Bournemouth  1641      41            34
    13            Swansea  1659      40            34
    14            Everton  1714      40            34
    15          West Brom  1653      40            34
    16     Crystal Palace  1647      38            34
    17         Sunderland  1623      33            34
    18            Norwich  1582      31            34
    19          Newcastle  1585      29            34
    20        Aston Villa  1518      16            34
    

    100%|██████████| 10000/10000 [01:31<00:00, 108.85it/s]

    10000 simulations
    1 winning places
    

    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Club</th>
      <th>RTB Wins</th>
      <th>LTB Wins</th>
      <th>% RTB winrate</th>
      <th>% LTB winrate</th>
      <th>Exp. RTB odds</th>
      <th>Exp. LTB odds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Leicester</td>
      <td>8955</td>
      <td>8589</td>
      <td>89.6</td>
      <td>85.9</td>
      <td>1.12</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tottenham</td>
      <td>1045</td>
      <td>743</td>
      <td>10.4</td>
      <td>7.4</td>
      <td>9.57</td>
      <td>13.46</td>
    </tr>
  </tbody>
</table>
</div>



Whether a fluke or not, it is difficult to tell; while the odds after 13 rounds seemed to be on point, futher down in the season, we have estimated higher probability of Leicester winning the title - which ultimately did happen.

We can imagine a model which would be even more aggressive with updating each team's estimated strength: even in the one here, Leicester has basically the same Elo as Chelsea, while there's a difference of 73 vs 45 points in the table between the two teams.

K-factor, which basically describes this "speed" of updates, is set to 20 by the clubelo themselves as one fitting best.
