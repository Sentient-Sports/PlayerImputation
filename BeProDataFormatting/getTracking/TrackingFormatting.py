'''
Contains formatting functions used in the BeProGetTracking python file
'''
import getTracking.load_epts_into_pandas as load_epts_into_pandas
import pandas as pd
import scipy.signal as signal
import numpy as np

#Gets tracking data and metadata from EPTS files for a tracking game and outputs tracking df and team dataframes
def tracking_files_to_df(txt, xml):
    metadata, tracking_df = load_epts_into_pandas.main(xml,txt) #Uses EPTS Python file to get xml and txt file as a dataframe
    #tracking_df.loc[:,tracking_df.columns.str.contains('velocity')] = tracking_df.loc[:,tracking_df.columns.str.contains('velocity')].apply(lambda xs: [float(math.nan) if x == 'N/A' else float(x) for x in xs])
    
    #Assumes First Team in MetaData is always the home team
    home_df = pd.DataFrame([[p.player_id, p.name, str(p.attributes.get('position')),p.jersey_no] for p in metadata.teams[0].players], columns = ['player_id','name','position', 'shirt'])
    away_df = pd.DataFrame([[p.player_id, p.name, str(p.attributes.get('position')),p.jersey_no] for p in metadata.teams[1].players], columns = ['player_id','name','position', 'shirt'])
    return tracking_df, home_df, away_df

#Combines first half and second half and also stores event time to link to the event data
def combine_FH_SH_tracking(data_fh, data_sh):
    data_fh['event_time'] = (data_fh['frame_id']-300)/30*1000 #250 for padding, 25 for FPS for demo game
    data_fh['period_id'] = 0
    data_sh['event_time'] = (data_sh['frame_id']-300)/30*1000 + (45*60*1000)
    data_sh['period_id'] = 1
    data_full = pd.concat([data_fh, data_sh], ignore_index=True)
    return data_full

#Calculates player velocities in x and y direction and their total speed for each timestamp in the tracking data
#Adds there columns to a tracking dataframe, and takes in a tracking dataframe. Applies and filter and smoothing to calculate velocity
#Used from Laurie on Tracking Github: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
def calc_player_velocities(tracking_df, home_df, away_df, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """
    Parameters
    -----------
        tracking_df: the tracking DataFrame for home or away team
        home_df, away_df: Team dataframes
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returns
    -----------
       tracking_df : the tracking DataFrame with columns for speed in the x & y direction and total speed added
    """

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = tracking_df['event_time'].diff() / 1000
    
    # index of first frame in second half
    second_half_idx = tracking_df['period_id'].idxmax(1)
    
    print(tracking_df.columns)
    player_ids = list(home_df['player_id'].values) + list(away_df['player_id'].values)
    # estimate velocities for players in team
    player_ids = [p for p in player_ids if ('player_' + p + '_x') in tracking_df.columns]
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        tracking_df['player_'+player+"_x"] = pd.Series([float(p) for p in tracking_df['player_'+player+"_x"]])
        tracking_df['player_'+player+"_y"] = pd.Series([float(p) for p in tracking_df['player_'+player+"_y"]])
        vx = tracking_df['player_'+player+"_x"].diff() / dt
        vy = tracking_df['player_'+player+"_y"].diff() / dt

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        tracking_df['player_'+player + "_vx"] = vx
        tracking_df['player_'+player + "_vy"] = vy
        tracking_df['player_'+player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return tracking_df