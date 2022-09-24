import UtilFunctions.util_functions as util_functions
import torch
from torch import nn
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

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
    return pd.DataFrame({'bally': ball_y,'prev_player_y': prev_player_y,'next_player_y':next_player_y, 'time_since_last_pred':time_since_last_pred})

def get_player_categorical_data(events_df, player, team_df, formation):
    position = formation[str(player)]#list(team_df[team_df['player_id'] == str(player)]['position_name'].values) * len(events_df)
    event_type = events_df['event_types_0_eventType']
    team_on_ball = events_df['player_id'].isin([float(f) for f in team_df['player_id']])
    player_on_ball = pd.Series(np.where(events_df['player_id'] == player, True, False))
    return pd.DataFrame({'position':position,'event_type':event_type,'team_on_ball':team_on_ball, 'player_on_ball':player_on_ball})

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
    input_data_normalized = scaler.fit_transform(input_data.loc[:, input_data.columns != "time_since_last_pred"])
    #input_data_time = time_scaler.fit_transform(np.array(input_data['time_since_last_pred']).reshape(-1, 1))
    #input_data_normalized = np.concatenate((input_data_normalized,input_data_time),axis=1)
    label_data_normalized = scaler.fit_transform(label_data.reshape(-1, 2))
    input_data_normalized = torch.cat((torch.tensor(input_data_normalized),emb_cats),1)
    return input_data_normalized, label_data_normalized, scaler

#Gets sequences of the last x values for input data
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix][-1]
        X.append(seq_x), y.append(seq_y)
    return X, y