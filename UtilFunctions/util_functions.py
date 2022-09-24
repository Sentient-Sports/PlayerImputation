'''
Utility functions for tracking testings
'''
import pandas as pd
import UtilFunctions.pitch_control as mpc
import UtilFunctions.plot_functions as pf
#import EPV
import numpy as np
from matplotlib import pyplot as plt

#Get all the data based on the game number
def get_dataframes(game_num):
    events_df = pd.read_csv('data/events/'+game_num+'/events_df.csv')
    tracking_df = pd.read_csv('data/tracking/'+game_num+'/full_tracking_df.csv')
    home_df = pd.read_csv('data/tracking/'+game_num+'/home_players.csv')
    away_df = pd.read_csv('data/tracking/'+game_num+'/away_players.csv')
    
    home_df['player_id'] = pd.Series([str(h) for h in home_df['player_id']])
    away_df['player_id'] = pd.Series([str(a) for a in away_df['player_id']])
    return events_df, tracking_df, home_df, away_df

#Get all the data based on the game number
def get_suwon_dataframes(game_num):
    events_df = pd.read_csv('data/Suwon_FC/events/'+game_num+'/events_df.csv')
    tracking_df = pd.read_csv('data/Suwon_FC/tracking/'+game_num+'/full_tracking_df.csv')
    home_df = pd.read_csv('data/Suwon_FC/tracking/'+game_num+'/home_players.csv')
    away_df = pd.read_csv('data/Suwon_FC/tracking/'+game_num+'/away_players.csv')
    formation_df = pd.read_csv('data/Suwon_FC/formations/'+game_num+'/formation.csv')
    
    home_df['player_id'] = pd.Series([str(h) for h in home_df['player_id']])
    away_df['player_id'] = pd.Series([str(a) for a in away_df['player_id']])
    return events_df, tracking_df, home_df, away_df, formation_df

#Takes events dataframe in BePro coordinates format and converts coordinates to EPTS format to match tracking data (0,0) to (105, 68)
def convert_bepro_to_EPTS(events_df):
    #Flip x and y coordinates, and drop events where team is unknown
    events_df[['x','y','relative_event_x','relative_event_y']] = events_df[['y','x','relative_event_y','relative_event_x']]
    events_df = events_df.dropna(axis=0,subset=['team_id']).reset_index(drop=True)
    
    #Get Team id for kickoff, and work out if that team is playing left to right or right to left. Then swap accordingly. This is so that the orientation of the events is right as it flips for each team in possession. Tracking data has the ball always in its real position, where events will flip based on possession team.
    
    #******
    #SOMETIMES THE EVENTS ARE FLIPPED COMPARED TO THE TRACKING. THIS IS CURRENTLY BEING MANUALLY RECONFIGURED BY TESTING IN THE TRACKING_TESTS FILE
    #******
    tid = events_df.loc[0]['team_id']
    if events_df.loc[1]['x'] < events_df.loc[0]['x']:
        l = True
    else:
        l = False
        
    if l == True:  
        events_df.loc[(events_df['event_period'] =='FIRST_HALF') & (events_df['team_id'] != tid),['x','y','relative_event_x','relative_event_y']] = 1-events_df.loc[(events_df['event_period'] =='FIRST_HALF') & (events_df['team_id'] != tid),['x','y','relative_event_x','relative_event_y']] 
        events_df.loc[(events_df['event_period'] =='SECOND_HALF') & (events_df['team_id'] == tid),['x','y','relative_event_x','relative_event_y']] = 1-events_df.loc[(events_df['event_period'] =='SECOND_HALF') & (events_df['team_id'] == tid),['x','y','relative_event_x','relative_event_y']] 
    else:
        events_df.loc[(events_df['event_period'] =='FIRST_HALF') & (events_df['team_id'] == tid),['x','y','relative_event_x','relative_event_y']] = 1 - events_df.loc[(events_df['event_period'] =='FIRST_HALF') & (events_df['team_id'] == tid),['x','y','relative_event_x','relative_event_y']] 
        events_df.loc[(events_df['event_period'] =='SECOND_HALF') & (events_df['team_id'] != tid),['x','y','relative_event_x','relative_event_y']] = 1 - events_df.loc[(events_df['event_period'] =='SECOND_HALF') & (events_df['team_id'] != tid),['x','y','relative_event_x','relative_event_y']] 
    
    events_df[['x','relative_event_x']] = events_df[['x','relative_event_x']] * 105
    events_df[['y','relative_event_y']] = events_df[['y','relative_event_y']] * 68
    return events_df

#Get the team id of each team in the game given an events df
def get_teams(events_df, home_df):
    if str(int(events_df.loc[0]['player_id'])) in home_df['player_id'].values:
        home_team = events_df.loc[0]['team_id']
        away_team = events_df['team_id'].unique()[1]
    else:
        home_team = events_df['team_id'].unique()[1]
        away_team = events_df.loc[0]['team_id']
    return home_team,away_team

#Convert event half to tracking half
def event_period_to_tracking(period):
    if period == 'FIRST_HALF':
        return 0
    elif period == 'SECOND_HALF':
        return 1
    
#Convert tracking half to event half
def tracking_period_to_event(period):
    if period == 0:
        return 'FIRST_HALF'
    elif period == 1:
        return 'SECOND_HALF'

#Gets the index of the row in the tracking dataframe which links to an event time based on the closest time
def get_tracking_index_from_event_time(tracking_df, event_time, period):
    t_df = tracking_df[tracking_df['period_id'] == event_period_to_tracking(period)]
    
    if event_period_to_tracking(period) == 1:
        len_FH = tracking_df[tracking_df['period_id'] == 0].shape[0]
        t_df = t_df.reset_index(drop=True)
        df_sort = (t_df.iloc[(t_df['event_time']-event_time).abs().argsort()[0]]).name + len_FH
    else:
        df_sort = (t_df.iloc[(t_df['event_time']-event_time).abs().argsort()[0]]).name
    return df_sort

#Get index of row which links to event time from events dataframe
def get_event_index_from_event_time(events_df, event_time, period):
    e_df = events_df[events_df['event_period'] == tracking_period_to_event(period)]
    
    if tracking_period_to_event(period) == 'SECOND_HALF':
        len_FH = events_df[events_df['period_id'] == 'FIRST HALF'].shape[0]
        e_df = e_df.reset_index(drop=True)
        df_sort = (e_df.iloc[(e_df['event_time']-event_time).abs().argsort()[0]]).name + len_FH
    else:
        df_sort = (e_df.iloc[(e_df['event_time']-event_time).abs().argsort()[0]]).name
    return df_sort

# Some of the following code is extracted from the Laurie on Tracking Github page.

#Get physical statistics of a team from their tracking by calculating distance covered from their speed over time.
def get_player_physical_statistics_for_team(team_df, tracking_df, fps):
    team_summary = pd.DataFrame(index=team_df['name'])
    minutes = []
    distance = []
    walking = []
    jogging = []
    running = []
    sprinting = []
    
    for player in team_df['player_id']:
        # search for first and last frames that we have a position observation for each player (when a player is not on the pitch positions are NaN)
        mins_column = 'player_' + player + '_x' # use player x-position coordinate
        dist_column = 'player_' + player + '_speed'
        player_minutes = (tracking_df[mins_column].last_valid_index() - tracking_df[mins_column].first_valid_index() + 1 ) / fps / 60. # convert to minutes
        player_distance = tracking_df[dist_column].sum()/fps/1000 # this is the sum of the distance travelled from one observation to the next (1/25 = 40ms) in km.
        
         # walking (less than 2 m/s)
        player_walking_distance = tracking_df.loc[tracking_df[dist_column] < 2, dist_column].sum()/fps/1000
        # jogging (between 2 and 4 m/s)
        player_jogging_distance = tracking_df.loc[ (tracking_df[dist_column] >= 2) & (tracking_df[dist_column] < 4), dist_column].sum()/fps/1000
        # running (between 4 and 7 m/s)
        player_running_distance = tracking_df.loc[ (tracking_df[dist_column] >= 4) & (tracking_df[dist_column] < 7), dist_column].sum()/fps/1000
        # sprinting (greater than 7 m/s)
        player_sprinting_distance = tracking_df.loc[ tracking_df[dist_column] >= 7, dist_column].sum()/25./1000
    
        minutes.append(player_minutes)
        distance.append( player_distance )
        walking.append( player_walking_distance )
        jogging.append( player_jogging_distance )
        running.append( player_running_distance )
        sprinting.append( player_sprinting_distance )
        
    team_summary['Minutes Played'] = minutes
    team_summary['Distance [km]'] = distance
    team_summary['Walking [km]'] = walking
    team_summary['Jogging [km]'] = jogging
    team_summary['Running [km]'] = running
    team_summary['Sprinting [km]'] = sprinting
    team_summary = team_summary.sort_values(['Minutes Played'], ascending=False)
    return team_summary

#Get goalkeepers for each team, useful for calling offsides (may cause issue if team has multiple goalkeepers in squad)
def get_goalkeepers(home_df, away_df):
    return [home_df[home_df['position'] == 'Goalie']['player_id'].values[0], away_df[away_df['position'] == 'Goalie']['player_id'].values[0]]

#Get tracking DFs for just the home players and the away players seperately
def get_home_away_players_tracking(tracking_df, home_df, away_df):
    home_players = ['event_time'] + list(('player_' + home_df['player_id'] + '_x')) + list(('player_' + home_df['player_id'] + '_y')) + list(('player_' + home_df['player_id'] + '_vx')) + list(('player_' + home_df['player_id'] + '_vy')) + list(('player_' + home_df['player_id'] + '_speed'))
    away_players = ['event_time'] + list(('player_' + away_df['player_id'] + '_x')) + list(('player_' + away_df['player_id'] + '_y')) + list(('player_' + away_df['player_id'] + '_vx')) + list(('player_' + away_df['player_id'] + '_vy')) + list(('player_' + away_df['player_id'] + '_speed'))
    home_tracking = tracking_df.loc[:, tracking_df.columns.isin(home_players)]
    away_tracking = tracking_df.loc[:, tracking_df.columns.isin(away_players)]
    return home_tracking, away_tracking

#Find play direction at an event
def find_player_direction_for_event(event_num, events_df, tracking_df, gk_numbers, home_df):
    event = events_df.iloc[event_num]
    event_team = event['team_id']
    event_time = event['event_time']
    event_period = event['event_period']
    ht, at = get_teams(events_df, home_df)
    if event_team == ht:
        gk = gk_numbers[0]
    elif event_team == at:
        gk = gk_numbers[1]
    tracking_frame = get_tracking_index_from_event_time(tracking_df, event_time, event_period)
    gk_pos = tracking_df.loc[tracking_frame]['player_'+gk+'_x']
    if gk_pos > 52.5:
        play_dir = -1
    else:
        play_dir = 1
    return play_dir

#Find play direction at closest event to tracking frame
def find_player_direction_for_tracking_frame(t_id, events_df, tracking_df, gk_numbers, home_df):
    tracking = tracking_df.iloc[t_id]
    event_time = tracking['event_time']
    event_period = tracking['period_id']
    event = events_df.iloc[get_event_index_from_event_time(events_df, event_time, event_period)]
    print(event)
    event_team = event['team_id']
    ht, at = get_teams(events_df, home_df)
    if event_team == ht:
        gk = gk_numbers[0]
    elif event_team == at:
        gk = gk_numbers[1]
    gk_pos = tracking_df.loc[t_id]['player_'+gk+'_x']
    if gk_pos > 52.5:
        play_dir = -1
    else:
        play_dir = 1
    return play_dir

#Calculates and shows pitch control
def get_pitch_control(event_num, events_df, tracking_df, home_df, away_df, home_team, away_team, params, goalkeepers):
    home_tracking, away_tracking = get_home_away_players_tracking(tracking_df, home_df, away_df)
    PPCF,xgrid,ygrid = mpc.generate_pitch_control_for_event(event_num, events_df, tracking_df, home_tracking, away_tracking, home_team, away_team, params, goalkeepers, field_dimen = (105.,68.,), n_grid_cells_x = 50)
    pf.plot_pitchcontrol_for_event(event_num, events_df, tracking_df, home_df, away_df, PPCF, annotate=True )
    return PPCF

#Calculate pass probabilities and list the most risky passes
def classify_pass_probabilities(events_df, home_team, tracking_df, home_df, away_df, params, goalkeepers):
    home_passes = events_df[(events_df['team_id'] == home_team) & (events_df['event_types_0_eventType'] == 'Pass') & (events_df['event_types_0_outcome'] == 'Successful')]

    pass_success_probability = []
    tracking_home, tracking_away = get_home_away_players_tracking(tracking_df, home_df, away_df)

    for i, row in home_passes.iterrows():
        pass_start_pos = np.array([row['x'],row['y']])
        pass_end_pos = np.array([row['relative_event_x'],row['relative_event_y']])
        pass_time = row['event_time']
        pass_period = row['event_period']
        tracking_index = get_tracking_index_from_event_time(tracking_df, pass_time, pass_period)

        attacking_players = mpc.initialise_players(tracking_home.loc[tracking_index],'Home',params,goalkeepers[0])
        defending_players = mpc.initialise_players(tracking_away.loc[tracking_index],'Away',params,goalkeepers[1])
        Patt,_ = mpc.calculate_pitch_control_at_target(pass_end_pos, attacking_players, defending_players, pass_start_pos, params)

        pass_success_probability.append((i,Patt))
        
    pass_success_probability = sorted(pass_success_probability, key=lambda x:x[1])
    fig,ax = plt.subplots()
    ax.hist( [p[1] for p in pass_success_probability], np.arange(0,1.1,0.1))    
    ax.set_xlabel('Pass success probability')
    ax.set_ylabel('Frequency')
    return pass_success_probability[:5]

#Get EPV for an event and plot the EPV and Pitch Control
def get_epv_for_event(event_num, events_df, tracking_df, home_team, away_team, home_df, away_df, params, goalkeepers, epv_grid):
    tracking_home, tracking_away = get_home_away_players_tracking(tracking_df, home_df, away_df)
    EEPV_added, EPV_diff = EPV.calculate_epv_added(event_num, events_df, tracking_df, tracking_home, tracking_away, home_df, home_team, away_team, goalkeepers, epv_grid, params)
    PPCF,xgrid,ygrid = mpc.generate_pitch_control_for_event(event_num, events_df, tracking_df, tracking_home, tracking_away, home_team, away_team, params, goalkeepers, field_dimen = (105.,68.,), n_grid_cells_x = 50)
    fig,ax = pf.plot_EPV_for_event( event_num, events_df, tracking_df, home_df, away_df, tracking_home, tracking_away, PPCF, epv_grid, goalkeepers, annotate=True, autoscale=True )
    pf.plot_pitchcontrol_for_event(event_num, events_df, tracking_df, home_df, away_df, PPCF, annotate=True )
    return EEPV_added

#List the 5 highest added Expected Value passes
def get_highest_EEPV_passes(events_df, tracking_df, home_df, away_df, home_team, away_team, goalkeepers, epv_grid, params):
    tracking_home, tracking_away = get_home_away_players_tracking(tracking_df, home_df, away_df)
    all_passes = events_df[(events_df['event_types_0_eventType'] == 'Pass') & (events_df['event_types_0_outcome'] == 'Successful')]
    value_added = []
    for pa,row in all_passes.iterrows():
        EEPV_added, EPV_diff = EPV.calculate_epv_added(pa, events_df, tracking_df, tracking_home, tracking_away, home_df, home_team, away_team, goalkeepers, epv_grid, params)
        #PPCF,xgrid,ygrid = mpc.generate_pitch_control_for_event(event_num, events_df, tracking_df, home_tracking, away_tracking, home_team, away_team, params, goalkeepers, field_dimen = (105.,68.,), n_grid_cells_x = 50)
        value_added.append( (pa,EEPV_added,EPV_diff ) )
    
    valued_passes = sorted(value_added, key = lambda x: x[1], reverse=True)
    return valued_passes[:5]

#Get player positions from BEPro coords
def coords_to_pos(x,y):
    if (x<1/4) & (y > 5/6):
        pos='LF'
    elif (1/4<=x<3/4) & (y > 5/6):
        pos='CF'
    elif (x>=3/4) & (y>5/6):
        pos='RF'
    elif (x<1/4) & (2/3 < y <= 5/6):
        pos='LW'
    elif (1/4<=x<3/4) & (2/3 < y <=5/6):
        pos='CAM'
    elif (x>=3/4) & (2/3 < y <= 5/6):
        pos='RW'
    elif (x < 1/4) & (1/2 < y <= 2/3):
        pos='LM'
    elif (1/4 <=x<3/4) & (1/2 < y<=2/3):
        pos='CM'
    elif (x>=3/4) & (1/2 < y <= 2/3):
        pos='RM'
    elif (1/4 <= x < 3/4) & (1/3 < y <= 1/2):
        pos='CDM'
    elif (x < 1/4) & (1/3 < y <= 1/2):
        pos='LWB'
    elif (x >= 3/4) & (1/3 < y <= 1/2):
        pos='RWB'
    elif (x < 1/4) & (1/6 < y <= 1/3):
        pos = 'LB'
    elif (1/4 <= x < 3/4) & (1/6 < y <= 1/3):
        pos = 'CB'
    elif (x >= 3/4) & (1/6 < y <= 1/3):
        pos = 'RB'
    elif (1/4 <= x < 3/4) & (y <= 1/6):
        pos = 'GK'
    return pos
