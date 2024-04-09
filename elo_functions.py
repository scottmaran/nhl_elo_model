import numpy as np

# Function that calculates the expected score of a team
def calc_expected_score(team_rating: float, opponent_rating: float) -> float: 
    exponent = -(team_rating - opponent_rating)/400
    return 1/(1 + np.power(10, exponent))

def create_initial_elo(initial_elo_score, team_names):
    keys = team_names
    values = initial_elo_score*np.ones(len(keys))
    elo_scores_dict = dict(zip(keys, values))
    
    return elo_scores_dict


'''

train_df['home_start_rating'] = 0
train_df['away_start_rating'] = 0
train_df['home_end_rating'] = 0
train_df['away_end_rating'] = 0

def win_prob(team_rating: float, opponent_rating: float) -> float:
    return 1/(1 + np.power(10, -(team_rating - opponent_rating)/400))   # 1/(1 + 10^(-diff/400))

keys = team_names
values = 1500*np.ones(len(keys))
elo_scores_dict = dict(zip(keys, values))
current_season = train_df.iloc[0,:]['numeric_season']
k_multiplier = 1

k = 24
alpha = 0.8
home_adv = 50
mov_exp = 1.0
mov_bias = 0
auto_coeff = 0.9
auto_bias = 0.1
k_mult=False


# iterate over every row of dataframe
for row_index in range(len(train_df)):
    
    # check if season change
    if current_season < train_df.loc[row_index,'numeric_season']:
        elo_scores_dict.update( (k, alpha*v + (1-alpha)*1500) for k, v in elo_scores_dict.items() )
        current_season = train_df.loc[row_index,'numeric_season']
    
    # get elo values of teams
    home_original_elo = elo_scores_dict[train_df.loc[row_index,'home_team']]
    away_original_elo = elo_scores_dict[train_df.loc[row_index,'away_team']]

    train_df.loc[row_index,'home_start_rating'] = home_original_elo
    train_df.loc[row_index,'away_start_rating'] = away_original_elo

    # calculate expected scores (probs)
    home_prob = win_prob(home_original_elo + home_adv, away_original_elo)
    away_prob = win_prob(away_original_elo, home_original_elo + home_adv)
    
    # Calculate new elo values
    if k_mult:
        elo_diff = home_original_elo + home_adv - away_original_elo
        mov = np.abs(train_df.loc[row_index,'home_score'] - train_df.loc[row_index,'away_score'])
        k_multiplier = np.power(mov + mov_bias, mov_exp)/(auto_bias + auto_coeff*elo_diff)

    home_updated_elo = home_original_elo + k*k_multiplier*(train_df.loc[row_index,'home_win'] - home_prob)
    away_updated_elo = away_original_elo + k*k_multiplier*(train_df.loc[row_index,'away_win'] - away_prob)

    train_df.loc[row_index,'home_end_rating'] = home_updated_elo
    train_df.loc[row_index,'away_end_rating'] = away_updated_elo

    # update ELO values
    elo_scores_dict[train_df.loc[row_index,'home_team']] = home_updated_elo
    elo_scores_dict[train_df.loc[row_index,'away_team']] = away_updated_elo
'''