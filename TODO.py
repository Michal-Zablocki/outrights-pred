###

# backtesting [high prio]

# 1. write some script to "generate" own odds and compare them to the bookies' odds on single games; look up old kaggle stuff
# 2. check how successful opta's and clubelo's simulations/predictitons were in the past
# 3. run an analysis like in the paper - middle of season standings vs. end of season standings
# 4. manual collection of data on historical outrights odds: source | date | season | round | league | bookie's odds for teams | our model's odds for teams

###

# the tool [medium prio]

# 1. implement range 9-24 in run_multiple_sims;
# 2. add WTB (winning tie-breakers) sim
# 3. [low prio] visualization: conditional formatting of odds, graph odds over time
# 4. additional factor E: every elo is multiplied by E (e.g. 1.2) to potentially decrease variance
# 5. set HFA to 90 for international competitions

###

# scraping [low prio]

# 1. get/scrape outrights odds [check the odds api] OR parse the screenshots
# 2. scrape opta power rankings

###

# european leagues with seasons not ending around May: (Finland, Ireland), Norway, Sweden
# american leagues -||-: Brazil, Canada, Ecuador, USA, Uruguay
# asian leagues -||-: Japan, Korea et al.
