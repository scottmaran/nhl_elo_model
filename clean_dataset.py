import pandas as pd
import re

def get_mapping():
    mapping = {
        "CHI": "Chicago",
        "STL": "St Louis",
        "NSH": "Nashville",
        "TB": "Tampa Bay",
        "WPG": "Winnipeg",
        "PHI": "Philadelphia",
        "DAL": "Dallas",
        "NJ": "New Jersey",
        "TOR": "Toronto",
        "NYR": "NY Rangers",
        "NYI": "NY Islanders",
        "BOS": "Boston",
        "BUF": "Buffalo",
        "PIT": "Pittsburgh",
        "CAR": "Carolina",
        "DET": "Detroit",
        "CGY": "Calgary",
        "EDM": "Edmonton",
        "VAN": "Vancouver",
        "COL": "Colorado",
        "ARI": "Arizona",
        "ANA": "Anaheim",
        "LA": "Los Angeles",
        "SJ": "San Jose",
        "FLA": "Florida",
        "OTT": "Ottawa",
        "MTL": "Montreal",
        "VGK": "Vegas",
        "CBS": "Columbus",
        "MIN": "Minnesota",
        "WSH": "Washington",
        "CGY": "Calgary",
        "SEA": "Seattle"
    }
    return mapping

# Function used to get numeric scores and home team information from 'score' string in dataset
def extract_info(row, team_mapping):
    venue = row['venue']
    score = row['score']
    
    home_team = team_mapping.get(venue, "")
    away_team = ""
    away_score = ""
    
    for key, value in team_mapping.items():
        if key != venue and value in score:
            away_team = value
            away_score = int(re.search(r'[0-9]+',score.split(value)[-1]).group())   # get numbers after name
            break
    
    home_score = int(re.search(r'[0-9]+',score.split(home_team)[-1]).group())   # get numbers after name
    return pd.Series([home_team, home_score, away_team, away_score], index=['home_team', 'home_score', 'away_team', 'away_score'])

def get_dataset():
    url = "nhl_scores.csv"
    df = pd.read_csv(url, header=None).drop([4,5], axis=1)
    df.columns = ['season', 'date', 'venue', 'score', 'misc_location']
    # standard venue, get all team names
    df.venue = df.venue.str.replace("at ", "").str.upper()
    df['numeric_season'] = df.season.astype('category').cat.codes
    df['overtime'] = df.score.apply(lambda x : ("OT" in x) | ("SO" in x)).astype(int)
    return df

# adds needed columns for elo model, derived fromn info in 'score' column of dataframe
def create_cols(df, mapping):
    df[['home_team', 'home_score', 'away_team', 'away_score']] = df.apply(extract_info, axis=1, team_mapping=mapping)
    df['home_win'] = (df.home_score > df.away_score).astype(int)
    df['away_win'] = (df.away_score > df.home_score).astype(int)
    df['score_diff'] = df['home_score'] - df['away_score']
    return df

