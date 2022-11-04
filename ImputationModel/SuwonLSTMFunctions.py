import UtilFunctions.util_functions as util_functions
import torch
from torch import nn
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from torch.utils.data import Dataset

def get_tracking_indexes(events_df, tracking_df):
    tracking_indexes = [util_functions.get_tracking_index_from_event_time(tracking_df, e, p) for e,p in zip(events_df['event_time'],events_df['event_period'])]
    return tracking_indexes

def get_player_events_x_data(events_df, player):
    #Gets times player is on the ball, and fills it so its previously seen value
    ball_x = events_df['x']
    events_FH = events_df[events_df['event_period']=='FIRST_HALF']
    events_SH = events_df[events_df['event_period']=='SECOND_HALF']
    FH_indexes = events_FH.index
    SH_indexes = events_SH.index
    prev_player_x_FH = pd.Series(np.where(events_FH['player_id'] == player, events_FH['x'], np.nan),index=FH_indexes).ffill().bfill()
    next_player_x_FH = pd.Series(np.where(events_FH['player_id'] == player, events_FH['x'], np.nan),index=FH_indexes).bfill().ffill()
    prev_player_x_SH = pd.Series(np.where(events_SH['player_id'] == player, events_SH['x'], np.nan),index=SH_indexes).ffill().bfill()
    next_player_x_SH = pd.Series(np.where(events_SH['player_id'] == player, events_SH['x'], np.nan),index=SH_indexes).bfill().ffill()
    prev_player_x = pd.concat([prev_player_x_FH,prev_player_x_SH],axis=0)
    next_player_x = pd.concat([next_player_x_FH,next_player_x_SH],axis=0)
    return pd.DataFrame({'ballx': ball_x, 'prev_player_x': prev_player_x,'next_player_x':next_player_x})

def get_player_events_y_data(events_df, player):
    #Gets times player is on the ball, and fills it so its previously seen value
    ball_y = events_df['y']
    events_FH = events_df[events_df['event_period']=='FIRST_HALF']
    events_SH = events_df[events_df['event_period']=='SECOND_HALF']
    FH_indexes = events_FH.index
    SH_indexes = events_SH.index
    prev_player_y_FH = pd.Series(np.where(events_FH['player_id'] == player, events_FH['y'], np.nan),index=FH_indexes).ffill().bfill()
    next_player_y_FH = pd.Series(np.where(events_FH['player_id'] == player, events_FH['y'], np.nan),index=FH_indexes).bfill().ffill()
    prev_player_y_SH = pd.Series(np.where(events_SH['player_id'] == player, events_SH['y'], np.nan),index=SH_indexes).ffill().bfill()
    next_player_y_SH = pd.Series(np.where(events_SH['player_id'] == player, events_SH['y'], np.nan),index=SH_indexes).bfill().ffill()
    prev_player_y = pd.concat([prev_player_y_FH,prev_player_y_SH],axis=0)
    next_player_y = pd.concat([next_player_y_FH,next_player_y_SH],axis=0)
    time_since_last_pred = events_df['event_time'].diff().fillna(0)
    prev_player_time = []
    next_player_time = []
    
    for i in range(len(events_df)):
        prev_events = events_df.loc[0:i]
        prev_player = prev_events[prev_events['player_id'] == player]
        next_events = events_df.loc[i+1:]
        next_player = next_events[next_events['player_id'] == player]
        if len(prev_player) > 0:         
            last_player_time = events_df.iloc[i]['event_time'] - prev_player['event_time'].iloc[-1]
        else:
            last_player_time = float(1000000)

                #Gets next seen position of the player using the first value in next_player, and the time difference between this event and current event   
        if (len(next_player) > 0):
            if (next_player['event_time'].iloc[0] >= events_df.iloc[i]['event_time']):
                event_time_g = next_player['event_time'].iloc[0] - events_df.iloc[i]['event_time']
            else:
                event_time_g = float(1000000)
        else:
            event_time_g = float(1000000)
        prev_player_time.append(last_player_time)
        next_player_time.append(event_time_g)
    return pd.DataFrame({'bally': ball_y,'prev_player_y': prev_player_y,'next_player_y':next_player_y, 'time_since_last_pred':time_since_last_pred, 'prev_player_time': prev_player_time,'next_player_time': next_player_time})

def get_player_categorical_data(events_df, player, team_df, formation):
    position = formation[str(player)]#list(team_df[team_df['player_id'] == str(player)]['position_name'].values) * len(events_df)
    event_type = events_df['event_types_0_eventType']
    event_id = events_df['id']
    match_id = events_df['match_id']
    event_time = events_df['event_time']
    team_on_ball = events_df['player_id'].isin([float(f) for f in team_df['player_id']])
    player_on_ball = pd.Series(np.where(events_df['player_id'] == player, True, False))
    return pd.DataFrame({'event_id':event_id,'match_id':match_id,'event_time':event_time,'player_id':player,'position':position,'event_type':event_type,'team_on_ball':team_on_ball, 'player_on_ball':player_on_ball})

def get_embedding(category_vec):
    num_classes = len(category_vec.unique()) #Will need modifying when whole dataset is built
    emb_size = math.floor(math.sqrt(num_classes))
    le = preprocessing.LabelEncoder()
    cat = le.fit_transform(category_vec)
    embedding = nn.Embedding(num_classes, emb_size)
    embedded_classes = embedding(torch.tensor(cat))
    return embedded_classes

def get_embedding_tensor_for_game(categories):
    cats = torch.tensor([])
    for cat in categories:
        new_cat = get_embedding(categories[cat])
        cats = torch.cat((cats,new_cat),axis=1)
    return cats

#FLIPS THE DATA SO THAT PLAYERS ARE ALWAYS SHOOTING FROM LEFT TO RIGHT, THIS IS DONE SO THAT THE MODEL CAN LEARN EASIER AND NOT GET CONFUSED BY TEAMS SHOOTING IN DIFFERENT DIRECTIONS
def get_data(events_df, tracking_df, player, team_df, team_gk, tracking_indexes, goal_diff, formation):
    ltr_mask_x = np.where(tracking_df['player_'+str(team_gk)+'_x'][tracking_indexes].reset_index(drop=True) > 52.5, 105, 0)
    ltr_mask_y = np.where(tracking_df['player_'+str(team_gk)+'_x'][tracking_indexes].reset_index(drop=True) > 52.5, 68, 0)
    
    input_x_data = abs(get_player_events_x_data(events_df,player) - ltr_mask_x[:, None])
    input_y_data = abs(get_player_events_y_data(events_df,player) - ltr_mask_y[:, None])
    
    cats = get_player_categorical_data(events_df,player, team_df, formation)
    cats = pd.concat([cats, goal_diff],axis=1)
    
    input_data = pd.concat([input_x_data,input_y_data, cats], axis=1) #torch.cat((input_x_data, input_y_data,cats),1)
    input_data['av_player_x'] = np.mean(input_data[input_data['player_on_ball'] == True]['ballx'])
    input_data['av_player_y'] = np.mean(input_data[input_data['player_on_ball'] == True]['bally'])
    
    label_x = abs(np.array(tracking_df['player_'+str(player)+'_x'][tracking_indexes])  - ltr_mask_x)
    label_y = abs(np.array(tracking_df['player_'+str(player)+'_y'][tracking_indexes])  - ltr_mask_y)
    label_data = np.array(list(zip(label_x,label_y)))
    
    nan_rows = np.where((np.isnan(label_data)).all(axis=1))[0]
    label_data = np.delete(label_data, nan_rows, axis=0)
    input_data.drop(input_data.index[nan_rows], inplace=True)
    
    return input_data, label_data

def preprocess_data(input_data, cats, label_data):
    emb_cats = get_embedding_tensor_for_game(cats)
    scaler = MinMaxScaler(feature_range=(0, 1))
    time_scaler = MinMaxScaler(feature_range=(0, 1))
    input_data_normalized = scaler.fit_transform(input_data.loc[:, (input_data.columns != "time_since_last_pred") | (input_data.columns != "prev_player_time") | (input_data.columns != "next_player_time")])
    input_data_time = time_scaler.fit_transform(np.array(input_data[['time_since_last_pred','prev_player_time','next_player_time']]).reshape(-1, 3))
    input_data_normalized = np.concatenate((input_data_normalized,input_data_time),axis=1)
    label_data_normalized = scaler.fit_transform(label_data.reshape(-1, 2))
    input_data_normalized = torch.cat((torch.tensor(input_data_normalized),emb_cats),1)
    return input_data_normalized, label_data_normalized, scaler

#Gets sequences of the previous and next x values for input data
def split_sequences(sorted_whole_input_df,input_sequences, output_sequence, timestamps, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    ts = list()
    indexes = list()
    pid = sorted_whole_input_df['player_id']
    for i in range(-2,len(input_sequences)):
            
        # find the end of the input, output sequence
        end_ix = i + n_steps_in + n_steps_out 
        out_end_ix = end_ix - n_steps_out
        if out_end_ix == len(input_sequences): break
        swid = pid[pid == pid.iloc[out_end_ix]].reset_index()
        swid_new_index = swid.loc[swid['index'] == out_end_ix].index.values[0]
        if (swid_new_index < 2) | (swid_new_index > len(swid)-3):
            seq_indexes = [int(swid.loc[swid_new_index]['index'])] * 5
        else:
            seq_indexes = swid.loc[swid_new_index-2:swid_new_index+2]['index'].values
        seq_x = input_sequences[seq_indexes]
        seq_y = output_sequence[out_end_ix]
        timestamp = timestamps[seq_indexes]
        X.append(seq_x), y.append(seq_y), ts.append(timestamp), indexes.append(out_end_ix)
        
    return X, y, ts, indexes

#Calculate the goal difference at moments in a game
def get_goal_diff(events_df, team_id):
    gc = ((events_df['event_types_0_eventType'].str.contains("Goal Conceded")) & (events_df['team_id'] == team_id)).cumsum()
    gs = ((events_df['event_types_0_eventType'].str.contains("Goal Conceded")) & (events_df['team_id'] != team_id)).cumsum()
    return gs-gc

#Get full feature set data for each game in dataset
def get_game_data(events_df, tracking_df, home_df, away_df, goalkeepers, formation):
    tracking_indexes = get_tracking_indexes(events_df, tracking_df)
    whole_input = pd.DataFrame()
    whole_label = np.empty((0,2), float)
    home_team,away_team = util_functions.get_teams(events_df, home_df)
    
    home_goal_diff = pd.Series(get_goal_diff(events_df,home_team),name='goal_diff')
    away_goal_diff = pd.Series(get_goal_diff(events_df,away_team),name='goal_diff')
    
    for player in home_df['player_id']:
        if 'player_'+str(player)+'_x' in tracking_df.columns:
            input_data, label_data = get_data(events_df,tracking_df, int(player), home_df, goalkeepers[0],tracking_indexes, home_goal_diff, formation)
            whole_input = whole_input.append(input_data)
            whole_label = np.append(whole_label,label_data,axis=0)

    for player in away_df['player_id']:
        if 'player_'+str(player)+'_x' in tracking_df.columns:
            input_data, label_data = get_data(events_df,tracking_df, int(player), away_df, goalkeepers[1],tracking_indexes, away_goal_diff, formation)
            whole_input = whole_input.append(input_data)
            whole_label = np.append(whole_label,label_data,axis=0)

    whole_cat_input = whole_input[['position','event_type','team_on_ball','player_on_ball','goal_diff']]
    whole_num_input = whole_input[whole_input.columns[~whole_input.columns.isin(['event_id','match_id','event_time','player_id','position','event_type','team_on_ball','player_on_ball','goal_diff'])]]
    return whole_num_input, whole_cat_input, whole_label, tracking_df.loc[tracking_indexes][:],events_df, whole_input

#Sort the input data by event, and then team, and then position
def custom_sort(dataframe, whole_label):
    position_dict = {'GK': 0, 'LB': 1, 'LWB' : 2, 'CB': 3,'RB':4,'RWB':5,'CDM':6,'CM':7,'CAM':8,'LW':9,'LF':10,'RW':11,'RF':12,'CF':13} 
    whole_df = pd.DataFrame()
    dataframe = dataframe.reset_index(drop=True).sort_values('event_id')
    for e in dataframe['event_id'].unique():
        events = dataframe[(dataframe['event_id'] == e)]
        if(len(events) != 22):
            continue
        team_on_ball = events[(events['team_on_ball'] == True)]
        team_off_ball = events[(events['team_on_ball'] == False)]
        team_on_ball = team_on_ball.sort_values(by='position', key=lambda x: x.map(position_dict))
        team_off_ball = team_off_ball.sort_values(by='position', key=lambda x: x.map(position_dict))
        on_off = pd.concat([team_on_ball,team_off_ball])
        whole_df = pd.concat([whole_df,on_off])
    whole_label = whole_label[whole_df.index.values]
    return whole_df, whole_label

#Get train and test split
def get_train_test_split(sorted_input, X_ss, y_mm, ts, fold):
    train = list(set(sorted_input.index) - set(fold))
    X_train = [X_ss[i] for i in train]
    X_test = [X_ss[i] for i in fold]
    X_train_ts = [ts[i] for i in train]
    X_test_ts = [ts[i] for i in fold]

    y_train = torch.tensor([y_mm[i] for i in train])
    y_test = torch.tensor([y_mm[i] for i in fold])
    return X_train, X_test, y_train, y_test, X_train_ts, X_test_ts

#Convert into chunks of 22 players and sequences of 5 with all features
class series_data(Dataset):
    def __init__(self,x,y,t,feature_num):
        self.x = torch.stack(x).reshape(int(len(x)/22),22,5,feature_num)
        self.y = torch.tensor(y,dtype=torch.float32).reshape(int(len(x)/22),22,2)
        self.t = torch.stack(t).reshape(int(len(x)/22),22,5)
        self.len = int(len(x)/22)

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx],self.t[idx]
  
    def __len__(self):
        return self.len
    