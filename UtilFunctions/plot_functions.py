''' 
Pitch plotting functions, using code from Laurie on Tracking Github 
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
'''
from mplsoccer import Pitch
from matplotlib import animation
from matplotlib import pyplot as plt
import pandas as pd
import math
import UtilFunctions.util_functions as ut
import numpy as np

def plot_pitch():
    pitch = Pitch(pitch_type='uefa', pitch_length = 105, pitch_width=68,axis=True,label=True)
    fig,ax = pitch.draw()
    return fig,ax

"""
Gets the positions of all the players for a particular frame in the game
"""
def animate(i, tracking_df, home_df, away_df, ball, away, home):
    ball.set_data(float(tracking_df['ball_x'][i]),float(tracking_df['ball_y'][i]))
    x_coords = tracking_df.loc[i].filter(like='_x')
    y_coords = tracking_df.loc[i].filter(like='_y')
    
    home_ids_x = ['player_'+h+'_x' for h in home_df['player_id'].values]
    away_ids_x = ['player_'+a+'_x' for a in away_df['player_id'].values]
    home_ids_y = ['player_'+h+'_y' for h in home_df['player_id'].values]
    away_ids_y = ['player_'+a+'_y' for a in away_df['player_id'].values]
    home_x_coords = pd.to_numeric(x_coords.filter(items=home_ids_x))
    home_y_coords = pd.to_numeric(y_coords.filter(items=home_ids_y))
    away_x_coords = pd.to_numeric(x_coords.filter(items=away_ids_x))
    away_y_coords = pd.to_numeric(y_coords.filter(items=away_ids_y))
    home.set_data(home_x_coords, home_y_coords)
    away.set_data(away_x_coords, away_y_coords)
    return ball, home, away

"""
Animates the football game using animate function by continously plotting the player locations for each row of tracking data at 25fps
"""
def generate_video(tracking_df, home_df, away_df): 
    
    fig,ax = plot_pitch()
    marker_kwargs = {'marker':'o','markeredgecolor':'black','linestyle':'None'}
    ball, = ax.plot([],[],ms=6,markerfacecolor='w',zorder=3, **marker_kwargs)
    away, = ax.plot([],[],ms=10,markerfacecolor='r', **marker_kwargs)
    home, = ax.plot([],[],ms=10,markerfacecolor='b', **marker_kwargs)
    anim = animation.FuncAnimation(fig,animate,frames=len(tracking_df),interval=40,blit=True, fargs = (tracking_df,home_df,away_df, ball, away, home))
    anim.save('tracking.mp4',fps=25)
    
"""
Plot a frame of the game using a row of tracking data
"""
def plot_frame(tracking_row, home_df, away_df, figax=None, team_colors=('r','b'), include_player_velocities=True, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=True ):
    
    if figax is None: # create new pitch 
        fig,ax = plot_pitch()
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
        
    #Get the coordinates for the home and away players in the tracking game    
    x_coords = tracking_row.filter(like='_x')
    y_coords = tracking_row.filter(like='_y')    
        
    home_ids_x = ['player_'+h+'_x' for h in home_df['player_id']]
    away_ids_x = ['player_'+a+'_x' for a in away_df['player_id']]
    home_ids_y = ['player_'+h+'_y' for h in home_df['player_id']]
    away_ids_y = ['player_'+a+'_y' for a in away_df['player_id']]
    home_x_coords = pd.to_numeric(x_coords.filter(items=home_ids_x))
    home_y_coords = pd.to_numeric(y_coords.filter(items=home_ids_y))
    away_x_coords = pd.to_numeric(x_coords.filter(items=away_ids_x))
    away_y_coords = pd.to_numeric(y_coords.filter(items=away_ids_y))
     
    #Loop over the home team coordinates and plot the players, along with shirt names and player velocities
    count=0
    for x,y in zip(home_x_coords, home_y_coords):
        ax.plot(x,y,team_colors[0]+'o',MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha)
        if not (math.isnan(x)) & annotate:
            ax.text( x+0.5, y+0.5, home_df['shirt'][count], fontsize=10, color=team_colors[0]  )
        if not (math.isnan(x)) & include_player_velocities:
            try:
                ax.quiver( x, y, tracking_row['player_'+home_df['player_id'][count]+'_vx'], tracking_row['player_'+home_df['player_id'][count]+'_vy'], color=team_colors[0] , scale_units='inches', scale=10.,width=0.0025,headlength=5,headwidth=3,alpha=PlayerAlpha)
            except: 
                print('No Vel')
        count+=1
    
    #Loop over the away team coordinates and plot the players, along with shirt names and player velocities
    count=0
    for x,y in zip(away_x_coords, away_y_coords):
        ax.plot(x,y,team_colors[1]+'o',MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha)
        if not (math.isnan(x)) & annotate:
            ax.text( x+0.5, y+0.5, away_df['shirt'][count], fontsize=10, color=team_colors[1]  )
        if not (math.isnan(x)) & include_player_velocities:
            try:
                ax.quiver( x, y, tracking_row['player_'+away_df['player_id'][count]+'_vx'], tracking_row['player_'+away_df['player_id'][count]+'_vy'], color=team_colors[1] , scale_units='inches', scale=10.,width=0.0025,headlength=5,headwidth=3,alpha=PlayerAlpha)
            except:
                print("No Vel")
        count+=1
    
    #Plot the ball location
    ax.plot( tracking_row['ball_x'], tracking_row['ball_y'], 'ko', MarkerSize=10, alpha=1.0, LineWidth=0, color='green')
 
    return fig,ax

#Plot a sequence of events by plotting where the event took place, and an arrow of where the ball is going if that data is available through relative_event
def plot_events(events, figax=None, indicators = ['Marker','Arrow'], color='r', marker_style = 'o', alpha = 0.8, annotate=False):

    if figax is None: # create new pitch 
        fig,ax = plot_pitch()
    else: # overlay on a previously generated pitch
        fig,ax = figax 
        
    #print(events)    
    for i,row in events.iterrows():
        if 'Marker' in indicators:
            ax.plot(  row['x'], row['y'], color+marker_style, alpha=alpha )
        if not math.isnan(row['relative_event_x']) & ('Arrow' in indicators):
            ax.annotate("", xy=row[['relative_event_x','relative_event_y']], xytext=row[['x','y']], alpha=alpha, arrowprops=dict(alpha=alpha,width=0.5,headlength=4.0,headwidth=4.0,color=color),annotation_clip=False)
        if annotate:
            textstring = row['event_types_0_eventType'] + ': ' + str(row['player_id'])
            ax.text( row['x'], row['y'], textstring, fontsize=10, color=color)
    return fig,ax

#Plots the pitch control at a given event
def plot_pitchcontrol_for_event( event_id, events, tracking_df, home_df, away_df, PPCF, alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (105.0,68)):

    #Gets the event time and team on the ball, and the index in tracking dataframe to plot the frame based on event time
    pass_frame = events.loc[event_id]['event_time']
    pass_team = events.loc[event_id]['team_id']
    pass_half = events.loc[event_id]['event_period']
    tracking_index = ut.get_tracking_index_from_event_time(tracking_df, pass_frame, pass_half)
    #print(tracking_index)
    
    # plot frame and event
    fig,ax = plot_pitch()
    print(tracking_df.loc[tracking_index][['ball_x','ball_y']])
    print(events.loc[event_id:event_id])
    plot_frame( tracking_df.loc[tracking_index], home_df, away_df, figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
    
    # plot pitch control surface, chooses color to plot based on team on the ball
    if pass_team==ut.get_teams(events, home_df)[0]:
        cmap = 'bwr'
    else:
        cmap = 'bwr_r'
    
    #Plots the surface, extent represents the dimensions of the field, uses the color map specified
    #Flips the pitch control vector to match how pitch is plotted (first values on the bottom working up)
    ax.imshow(np.flipud(PPCF), extent=(0, field_dimen[0], 0., field_dimen[1]),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)

    return fig,ax

#Plot EPV grid
def plot_EPV(EPV,field_dimen=(105.0,68),attack_direction=1):
    if attack_direction==-1:
        # flip direction of grid if team is attacking right->left
        EPV = np.fliplr(EPV)
    ny,nx = EPV.shape
    # plot a pitch
    fig,ax = plot_pitch()
    # overlap the EPV surface
    ax.imshow(EPV, extent=(0., field_dimen[0], 0, field_dimen[1]),vmin=0.0,vmax=0.6,cmap='Blues',alpha=0.6)
    
#Plot EPV for Event
def plot_EPV_for_event( event_id, events, tracking_df, home_df, away_df, tracking_home, tracking_away, PPCF, EPV, gk_numbers, alpha = 0.7, include_player_velocities=True, annotate=False, autoscale=0.1, contours=False, field_dimen = (105.0,68)):
    """ plot_EPV_for_event( event_id, events,  tracking_home, tracking_away, PPCF, EPV, alpha, include_player_velocities, annotate, autoscale, contours, field_dimen)
    
    Plots the EPVxPitchControl surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        EPV: Expected Possession Value surface. EPV is the probability that a possession will end with a goal given the current location of the ball. 
             The EPV surface is saved in the FoT github repo and can be loaded using Metrica_EPV.load_EPV_grid()
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        autoscale: If True, use the max of surface to define the colorscale of the image. If set to a value [0-1], uses this as the maximum of the color scale.
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """    

    # pick a pass at which to generate the pitch control surface
    pass_frame = events.loc[event_id]['event_time']
    pass_team = events.loc[event_id]['team_id']
    pass_half = events.loc[event_id]['event_period']
    tracking_index = ut.get_tracking_index_from_event_time(tracking_df, pass_frame, pass_half)
    
    # plot frame and event
    fig,ax = plot_pitch()
    plot_frame( tracking_df.loc[tracking_index], home_df, away_df, figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
       
    # plot pitch control surface
    if pass_team==ut.get_teams(events, home_df)[0]:
        cmap = 'Reds'
        lcolor = 'r'
    else:
        cmap = 'Blues'
        lcolor = 'b'
    EPV = np.fliplr(EPV) if ut.find_player_direction_for_event(event_id, events, tracking_df, gk_numbers, home_df) == -1 else EPV
    EPVxPPCF = PPCF*EPV
    
    if autoscale is True:
        vmax = np.max(EPVxPPCF)*2.
    elif autoscale>=0 and autoscale<=1:
        vmax = autoscale
    else:
        assert False, "'autoscale' must be either {True or between 0 and 1}"
        
    ax.imshow(np.flipud(EPVxPPCF), extent=(0, field_dimen[0], 0, field_dimen[1]),interpolation='spline36',vmin=0.0,vmax=vmax,cmap=cmap,alpha=0.7)
    
    if contours:
        ax.contour( EPVxPPCF,extent=(0, field_dimen[0], 0, field_dimen[1]),levels=np.array([0.75])*np.max(EPVxPPCF),colors=lcolor,alpha=1.0)
    
    return fig,ax
