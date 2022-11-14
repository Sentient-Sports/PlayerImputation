"""
Utility functions for tracking, imputation and experiments code
"""
import pandas as pd
import UtilFunctions.plot_functions as pf
import UtilFunctions.pitch_control as ipc

params = ipc.default_model_params()
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from mplsoccer import Pitch
from statsmodels.stats.weightstats import DescrStatsW
from pandera.typing import DataFrame, Series
from typing import Callable, List, Optional, Tuple
l=8
w=6

"""
Load and convert datasets
"""

# Get all the data based on the game number
def get_dataframes(game_num):
    events_df = pd.read_csv("data/events/" + game_num + "/events_df.csv")
    tracking_df = pd.read_csv("data/tracking/" + game_num + "/full_tracking_df.csv")
    home_df = pd.read_csv("data/tracking/" + game_num + "/home_players.csv")
    away_df = pd.read_csv("data/tracking/" + game_num + "/away_players.csv")

    home_df["player_id"] = pd.Series([str(h) for h in home_df["player_id"]])
    away_df["player_id"] = pd.Series([str(a) for a in away_df["player_id"]])
    return events_df, tracking_df, home_df, away_df


# Get all the data based on the game number
def get_suwon_dataframes(game_num):
    events_df = pd.read_csv("data/Suwon_FC/events/" + game_num + "/events_df.csv")
    tracking_df = pd.read_csv(
        "data/Suwon_FC/tracking/" + game_num + "/full_tracking_df.csv"
    )
    home_df = pd.read_csv("data/Suwon_FC/tracking/" + game_num + "/home_players.csv")
    away_df = pd.read_csv("data/Suwon_FC/tracking/" + game_num + "/away_players.csv")
    formation_df = pd.read_csv(
        "data/Suwon_FC/formations/" + game_num + "/formation.csv"
    )

    home_df["player_id"] = pd.Series([str(h) for h in home_df["player_id"]])
    away_df["player_id"] = pd.Series([str(a) for a in away_df["player_id"]])
    return events_df, tracking_df, home_df, away_df, formation_df


# Takes events dataframe in BePro coordinates format and converts coordinates to EPTS format to match tracking data (0,0) to (105, 68)
def convert_bepro_to_EPTS(events_df):
    # Flip x and y coordinates, and drop events where team is unknown
    events_df[["x", "y", "relative_event_x", "relative_event_y"]] = events_df[
        ["y", "x", "relative_event_y", "relative_event_x"]
    ]
    events_df = events_df.dropna(axis=0, subset=["team_id"]).reset_index(drop=True)

    # Get Team id for kickoff, and work out if that team is playing left to right or right to left. Then swap accordingly. This is so that the orientation of the events is right as it flips for each team in possession. Tracking data has the ball always in its real position, where events will flip based on possession team.

    # ******
    # SOMETIMES THE EVENTS ARE FLIPPED COMPARED TO THE TRACKING. THIS IS CURRENTLY BEING MANUALLY RECONFIGURED BY TESTING IN THE TRACKING_TESTS FILE
    # ******
    tid = events_df.loc[0]["team_id"]
    if events_df.loc[1]["x"] < events_df.loc[0]["x"]:
        l = True
    else:
        l = False

    if l == True:
        events_df.loc[
            (events_df["event_period"] == "FIRST_HALF") & (events_df["team_id"] != tid),
            ["x", "y", "relative_event_x", "relative_event_y"],
        ] = (
            1
            - events_df.loc[
                (events_df["event_period"] == "FIRST_HALF")
                & (events_df["team_id"] != tid),
                ["x", "y", "relative_event_x", "relative_event_y"],
            ]
        )
        events_df.loc[
            (events_df["event_period"] == "SECOND_HALF")
            & (events_df["team_id"] == tid),
            ["x", "y", "relative_event_x", "relative_event_y"],
        ] = (
            1
            - events_df.loc[
                (events_df["event_period"] == "SECOND_HALF")
                & (events_df["team_id"] == tid),
                ["x", "y", "relative_event_x", "relative_event_y"],
            ]
        )
    else:
        events_df.loc[
            (events_df["event_period"] == "FIRST_HALF") & (events_df["team_id"] == tid),
            ["x", "y", "relative_event_x", "relative_event_y"],
        ] = (
            1
            - events_df.loc[
                (events_df["event_period"] == "FIRST_HALF")
                & (events_df["team_id"] == tid),
                ["x", "y", "relative_event_x", "relative_event_y"],
            ]
        )
        events_df.loc[
            (events_df["event_period"] == "SECOND_HALF")
            & (events_df["team_id"] != tid),
            ["x", "y", "relative_event_x", "relative_event_y"],
        ] = (
            1
            - events_df.loc[
                (events_df["event_period"] == "SECOND_HALF")
                & (events_df["team_id"] != tid),
                ["x", "y", "relative_event_x", "relative_event_y"],
            ]
        )

    events_df[["x", "relative_event_x"]] = events_df[["x", "relative_event_x"]] * 105
    events_df[["y", "relative_event_y"]] = events_df[["y", "relative_event_y"]] * 68
    return events_df


# Get the team id of each team in the game given an events df
def get_teams(events_df, home_df):
    if str(int(events_df.loc[0]["player_id"])) in home_df["player_id"].values:
        home_team = events_df.loc[0]["team_id"]
        away_team = events_df["team_id"].unique()[1]
    else:
        home_team = events_df["team_id"].unique()[1]
        away_team = events_df.loc[0]["team_id"]
    return home_team, away_team


# Convert event half to tracking half
def event_period_to_tracking(period):
    if period == "FIRST_HALF":
        return 0
    elif period == "SECOND_HALF":
        return 1


# Convert tracking half to event half
def tracking_period_to_event(period):
    if period == 0:
        return "FIRST_HALF"
    elif period == 1:
        return "SECOND_HALF"


# Gets the index of the row in the tracking dataframe which links to an event time based on the closest time
def get_tracking_index_from_event_time(tracking_df, event_time, period):
    t_df = tracking_df[tracking_df["period_id"] == event_period_to_tracking(period)]

    if event_period_to_tracking(period) == 1:
        len_FH = tracking_df[tracking_df["period_id"] == 0].shape[0]
        t_df = t_df.reset_index(drop=True)
        df_sort = (
            t_df.iloc[(t_df["event_time"] - event_time).abs().argsort()[0]]
        ).name + len_FH
    else:
        df_sort = (t_df.iloc[(t_df["event_time"] - event_time).abs().argsort()[0]]).name
    return df_sort


# Get index of row which links to event time from events dataframe
def get_event_index_from_event_time(events_df, event_time, period):
    e_df = events_df[events_df["event_period"] == tracking_period_to_event(period)]

    if tracking_period_to_event(period) == "SECOND_HALF":
        len_FH = events_df[events_df["period_id"] == "FIRST HALF"].shape[0]
        e_df = e_df.reset_index(drop=True)
        df_sort = (
            e_df.iloc[(e_df["event_time"] - event_time).abs().argsort()[0]]
        ).name + len_FH
    else:
        df_sort = (e_df.iloc[(e_df["event_time"] - event_time).abs().argsort()[0]]).name
    return df_sort


# Get goalkeepers for each team, useful for calling offsides (may cause issue if team has multiple goalkeepers in squad)
def get_goalkeepers(home_df, away_df):
    return [
        home_df[home_df["position"] == "Goalie"]["player_id"].values[0],
        away_df[away_df["position"] == "Goalie"]["player_id"].values[0],
    ]


# Get tracking DFs for just the home players and the away players seperately
def get_home_away_players_tracking(tracking_df, home_df, away_df):
    home_players = (
        ["event_time"]
        + list(("player_" + home_df["player_id"] + "_x"))
        + list(("player_" + home_df["player_id"] + "_y"))
        + list(("player_" + home_df["player_id"] + "_vx"))
        + list(("player_" + home_df["player_id"] + "_vy"))
        + list(("player_" + home_df["player_id"] + "_speed"))
    )
    away_players = (
        ["event_time"]
        + list(("player_" + away_df["player_id"] + "_x"))
        + list(("player_" + away_df["player_id"] + "_y"))
        + list(("player_" + away_df["player_id"] + "_vx"))
        + list(("player_" + away_df["player_id"] + "_vy"))
        + list(("player_" + away_df["player_id"] + "_speed"))
    )
    home_tracking = tracking_df.loc[:, tracking_df.columns.isin(home_players)]
    away_tracking = tracking_df.loc[:, tracking_df.columns.isin(away_players)]
    return home_tracking, away_tracking


# Find play direction at an event
def find_player_direction_for_event(
    event_num, events_df, tracking_df, gk_numbers, home_df
):
    event = events_df.iloc[event_num]
    event_team = event["team_id"]
    event_time = event["event_time"]
    event_period = event["event_period"]
    ht, at = get_teams(events_df, home_df)
    if event_team == ht:
        gk = gk_numbers[0]
    elif event_team == at:
        gk = gk_numbers[1]
    tracking_frame = get_tracking_index_from_event_time(
        tracking_df, event_time, event_period
    )
    gk_pos = tracking_df.loc[tracking_frame]["player_" + gk + "_x"]
    if gk_pos > 52.5:
        play_dir = -1
    else:
        play_dir = 1
    return play_dir


# Find play direction at closest event to tracking frame
def find_player_direction_for_tracking_frame(
    t_id, events_df, tracking_df, gk_numbers, home_df
):
    tracking = tracking_df.iloc[t_id]
    event_time = tracking["event_time"]
    event_period = tracking["period_id"]
    event = events_df.iloc[
        get_event_index_from_event_time(events_df, event_time, event_period)
    ]
    print(event)
    event_team = event["team_id"]
    ht, at = get_teams(events_df, home_df)
    if event_team == ht:
        gk = gk_numbers[0]
    elif event_team == at:
        gk = gk_numbers[1]
    gk_pos = tracking_df.loc[t_id]["player_" + gk + "_x"]
    if gk_pos > 52.5:
        play_dir = -1
    else:
        play_dir = 1
    return play_dir


# Calculates and shows pitch control
def get_pitch_control(
    event_num,
    events_df,
    tracking_df,
    home_df,
    away_df,
    home_team,
    away_team,
    params,
    goalkeepers,
):
    home_tracking, away_tracking = get_home_away_players_tracking(
        tracking_df, home_df, away_df
    )
    PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
        event_num,
        events_df,
        tracking_df,
        home_tracking,
        away_tracking,
        home_team,
        away_team,
        params,
        goalkeepers,
        field_dimen=(105.0, 68.0,),
        n_grid_cells_x=50,
    )
    pf.plot_pitchcontrol_for_event(
        event_num, events_df, tracking_df, home_df, away_df, PPCF, annotate=True
    )
    return PPCF


# Get player positions from BePro coordinates
def coords_to_pos(x, y):
    if (x < 1 / 4) & (y > 5 / 6):
        pos = "LF"
    elif (1 / 4 <= x < 3 / 4) & (y > 5 / 6):
        pos = "CF"
    elif (x >= 3 / 4) & (y > 5 / 6):
        pos = "RF"
    elif (x < 1 / 4) & (2 / 3 < y <= 5 / 6):
        pos = "LW"
    elif (1 / 4 <= x < 3 / 4) & (2 / 3 < y <= 5 / 6):
        pos = "CAM"
    elif (x >= 3 / 4) & (2 / 3 < y <= 5 / 6):
        pos = "RW"
    elif (x < 1 / 4) & (1 / 2 < y <= 2 / 3):
        pos = "LM"
    elif (1 / 4 <= x < 3 / 4) & (1 / 2 < y <= 2 / 3):
        pos = "CM"
    elif (x >= 3 / 4) & (1 / 2 < y <= 2 / 3):
        pos = "RM"
    elif (1 / 4 <= x < 3 / 4) & (1 / 3 < y <= 1 / 2):
        pos = "CDM"
    elif (x < 1 / 4) & (1 / 3 < y <= 1 / 2):
        pos = "LWB"
    elif (x >= 3 / 4) & (1 / 3 < y <= 1 / 2):
        pos = "RWB"
    elif (x < 1 / 4) & (1 / 6 < y <= 1 / 3):
        pos = "LB"
    elif (1 / 4 <= x < 3 / 4) & (1 / 6 < y <= 1 / 3):
        pos = "CB"
    elif (x >= 3 / 4) & (1 / 6 < y <= 1 / 3):
        pos = "RB"
    elif (1 / 4 <= x < 3 / 4) & (y <= 1 / 6):
        pos = "GK"
    return pos


"""
Functions used in Experiments notebook
"""

# Convert specific roles to generic roles
def convert_positions_to_generic(results_df):
    results_df.loc[results_df["position"] == "GK", "position"] = "Goalkeeper"
    results_df.loc[results_df["position"] == "CDM", "position"] = "Central Defender"
    results_df.loc[results_df["position"] == "CB", "position"] = "Central Defender"
    results_df.loc[results_df["position"] == "RB", "position"] = "Wide Defender"
    results_df.loc[results_df["position"] == "RWB", "position"] = "Wide Defender"
    results_df.loc[results_df["position"] == "LB", "position"] = "Wide Defender"
    results_df.loc[results_df["position"] == "LWB", "position"] = "Wide Defender"
    results_df.loc[results_df["position"] == "CM", "position"] = "Central Midfielder"
    results_df.loc[results_df["position"] == "RM", "position"] = "Wide Midfielder"
    results_df.loc[results_df["position"] == "LM", "position"] = "Wide Midfielder"
    results_df.loc[results_df["position"] == "CAM", "position"] = "Central Attacker"
    results_df.loc[results_df["position"] == "CF", "position"] = "Central Attacker"
    results_df.loc[results_df["position"] == "LW", "position"] = "Wide Attacker"
    results_df.loc[results_df["position"] == "RW", "position"] = "Wide Attacker"
    results_df.loc[results_df["position"] == "LF", "position"] = "Wide Attacker"
    results_df.loc[results_df["position"] == "RF", "position"] = "Wide Attacker"
    return results_df

# Convert specific roles to generic roles
def convert_positions_to_generic_sloan(results_df):
    results_df.loc[results_df["position"] == "GK", "position"] = "Goalkeeper"
    results_df.loc[results_df["position"] == "CDM", "position"] = "Center Midfielder"
    results_df.loc[results_df["position"] == "CB", "position"] = "Center Back"
    results_df.loc[results_df["position"] == "RB", "position"] = "Wing Back"
    results_df.loc[results_df["position"] == "RWB", "position"] = "Wing Back"
    results_df.loc[results_df["position"] == "LB", "position"] = "Wing Back"
    results_df.loc[results_df["position"] == "LWB", "position"] = "Wing Back"
    results_df.loc[results_df["position"] == "CM", "position"] = "Center Midfielder"
    results_df.loc[results_df["position"] == "RM", "position"] = "Winger"
    results_df.loc[results_df["position"] == "LM", "position"] = "Winger"
    results_df.loc[results_df["position"] == "CAM", "position"] = "Center Midfielder"
    results_df.loc[results_df["position"] == "CF", "position"] = "Striker"
    results_df.loc[results_df["position"] == "LW", "position"] = "Winger"
    results_df.loc[results_df["position"] == "RW", "position"] = "Winger"
    results_df.loc[results_df["position"] == "LF", "position"] = "Winger"
    results_df.loc[results_df["position"] == "RF", "position"] = "Winger"
    return results_df


# Get standard errors for distance errors when calculating by role
def get_standard_errors_all_directions(errors_df, sample_nums):
    errors_df["x_se"] = (errors_df["x_dist"] / np.sqrt(sample_nums)) * 1.959
    errors_df["y_se"] = (errors_df["y_dist"] / np.sqrt(sample_nums)) * 1.959
    errors_df["se"] = (errors_df["dist"] / np.sqrt(sample_nums)) * 1.959
    return errors_df


# Get time since last seen
def get_time_since_seen(results_df):
    for player in results_df["player_id"].unique():
        curr_player = results_df.loc[
            results_df["player_id"] == player,
            ["player_id", "time_since_last_pred", "player_on_ball"],
        ]
        l = []
        count = 10000
        for i, row in curr_player.iterrows():
            if row["player_on_ball"] == True:
                count = 0
            else:
                count += row["time_since_last_pred"] / 1000
            l.append(count)
        results_df.loc[curr_player.index, "time_since_seen"] = l

    for player in results_df["player_id"].unique():
        curr_player = results_df.loc[
            results_df["player_id"] == player,
            ["player_id", "time_since_last_pred", "player_on_ball"],
        ].reset_index()
        l = []
        count = 10000
        for i, row in curr_player.iterrows():
            next_player_events = curr_player.iloc[i:]
            try:
                next_seen_player_event = next_player_events[
                    next_player_events["player_on_ball"] == True
                ]
                time_next_seen = (
                    curr_player.iloc[i : next_seen_player_event.index[0]][
                        "time_since_last_pred"
                    ].sum()
                    / 1000
                )
            except:
                time_next_seen = 20000
            l.append(time_next_seen)
        results_df.loc[curr_player["index"], "time_next_seen"] = l
    return results_df


# Get the rolling average error across time periods in a match
def get_rolling_average_error(error_time, match_ids):
    rolling_means = []
    rolling_stds = []
    rolling_times = []

    # For each second in a game
    for i in range(60, 5650):
        means = []
        weights = []

        # Get the rolling average error over last 5 minutes for each game and then average these cross games using a weighted average
        for mid in match_ids:
            mid_temp = error_time[
                (error_time["match_id"] == mid)
                & (error_time.index > i - 300)
                & (error_time.index < i)
            ]["dist"]
            mid_mean = mid_temp.mean()
            mid_len = len(mid_temp)
            means.append(mid_mean)
            weights.append(mid_len)
        means = np.array(means)
        weights = np.array(weights)
        means = means[~np.isnan(means)]
        weights = weights[weights > 0]
        weighted_stats = DescrStatsW(means, weights=weights, ddof=0)
        weighted_mean = weighted_stats.mean
        weighted_std = weighted_stats.std
        rolling_means.append(weighted_mean)
        rolling_stds.append(weighted_std)
        rolling_times.append(i)

    # Save in dataframe and return
    means_df = pd.DataFrame()
    means_df["times"] = np.array(rolling_times) / 60
    means_df["means"] = rolling_means
    means_df["std"] = rolling_stds
    return means_df

# Get the rolling average error across time periods in a match for 1 half
def get_rolling_average_error_period1(error_time, match_ids,half):
    rolling_means = []
    rolling_stds = []
    rolling_times = []
    
    if half == 1:
        a = 60
        b = 2850
    else:
        a=2750
        b=5850
        
    # For each second in a game
    for i in range(a, b):
        means = []
        weights = []

        # Get the rolling average error over last 5 minutes for each game and then average these cross games using a weighted average
        for mid in match_ids:
            mid_temp = error_time[
                (error_time["match_id"] == mid)
                & (error_time.index > i - 300)
                & (error_time.index < i)
            ]["dist"]
            mid_mean = mid_temp.mean()
            mid_len = len(mid_temp)
            means.append(mid_mean)
            weights.append(mid_len)
        means = np.array(means)
        weights = np.array(weights)
        means = means[~np.isnan(means)]
        weights = weights[weights > 0]
        weighted_stats = DescrStatsW(means, weights=weights, ddof=0)
        weighted_mean = weighted_stats.mean
        weighted_std = weighted_stats.std
        rolling_means.append(weighted_mean)
        rolling_stds.append(weighted_std)
        rolling_times.append(i)

    # Save in dataframe and return
    means_df = pd.DataFrame()
    means_df["times"] = np.array(rolling_times) / 60
    means_df["means"] = rolling_means
    means_df["std"] = rolling_stds
    return means_df


# Calculate distance covered of players using predicted locations and compare to the actual results
def distance_covered_metric(agent_imputer_dist_covered, close_events):
    ps = []
    ms = []
    pred_dist = []
    act_dist = []
    mins = []
    position = []

    # Loop through players and matches for each player, store their difference between predictions (distance covered) and mins played
    for p in agent_imputer_dist_covered["player_id"].unique():
        for m in agent_imputer_dist_covered[
            agent_imputer_dist_covered["player_id"] == p
        ]["match_id"].unique():
            ms.append(m)
            ps.append(p)
            ai_player = agent_imputer_dist_covered[
                (agent_imputer_dist_covered["match_id"] == m)
                & (agent_imputer_dist_covered["player_id"] == p)
            ].copy()
            ai_player["pred_dist"] = np.sqrt(
                (abs((ai_player["pred_x"].diff()).fillna(0)) ** 2)
                + (abs((ai_player["pred_y"].diff()).fillna(0)) ** 2)
            )
            ai_player["act_dist"] = np.sqrt(
                (abs((ai_player["act_x"].diff()).fillna(0)) ** 2)
                + (abs((ai_player["act_y"].diff()).fillna(0)) ** 2)
            )

            # Remove events within 1s of eachother (optional)
            if close_events == True:
                ai_player = ai_player[
                    (ai_player["event_time"].diff() > 1)
                    & (ai_player["event_type"] != "Goal Conceded")
                ]
            mins.append(ai_player["time_since_last_pred"].sum() / 1000 / 60)
            pred_dist.append(ai_player["pred_dist"].sum())
            act_dist.append(ai_player["act_dist"].sum())
            position.append(ai_player.iloc[0]["position"])
    dist_df = pd.DataFrame()
    dist_df["player"] = ps
    dist_df["match"] = ms
    dist_df["mins"] = mins
    dist_df["position"] = position
    dist_df["pred_dist"] = pred_dist
    dist_df["act_dist"] = act_dist
    dist_df["error"] = (
        abs((np.array(pred_dist) - np.array(act_dist)) / np.array(pred_dist)) * 100
    )
    # Get rid of games where players played under 20 mins, and normalize to 90 mins
    dist_df = dist_df[dist_df["mins"] > 20]
    dist_df["dist_per_90"] = (dist_df["pred_dist"] / dist_df["mins"]) * 90
    dist_df["dist_act_per_90"] = (dist_df["act_dist"] / dist_df["mins"]) * 90
    dist_df["err_p_90"] = (
        abs(
            (np.array(dist_df["dist_per_90"]) - np.array(dist_df["dist_act_per_90"]))
            / np.array(dist_df["dist_per_90"])
        )
        * 100
    )
    return dist_df


# Generate Pitch Control values
def all_pitch_control(action, locs, params, plot):
    PPCFa, _, _ = ipc.generate_pitch_control_for_event(
        action,
        locs,
        params,
        field_dimen=(105.0, 68.0,),
        n_grid_cells_x=32,
        offsides=False,
    )
    return PPCFa


# Generate Pitch Control for predictor models
def calculate_model_pitch_control(results_df, event_id, actual):
    event_df = results_df[results_df["event_id"] == event_id].copy()
    if actual == False:
        event_df["x"] = event_df["pred_x"].copy()
        event_df["y"] = event_df["pred_y"].copy()
    else:
        event_df["x"] = event_df["act_x"].copy()
        event_df["y"] = event_df["act_y"].copy()
    event_df.loc[event_df["team_on_ball"] == False, "x"] = (
        105 - event_df.loc[event_df["team_on_ball"] == False, "x"]
    )
    event_df.loc[event_df["team_on_ball"] == False, "y"] = (
        68 - event_df.loc[event_df["team_on_ball"] == False, "y"]
    )
    PPCFa = all_pitch_control(event_df.iloc[0], event_df, params, False)
    fig, ax = ipc.plot_pitchcontrol_for_event(PPCFa, event_df.iloc[0], event_df)
    fig.set_size_inches(12,8)
    return PPCFa, ax


# Generate Player Heatmaps
def get_player_heatmaps(model_df, player_id, match_id, position):
    plt.clf()
    player_df = model_df[
        (model_df["player_id"] == player_id) & (model_df["match_id"] == match_id)
    ]
    pitch = Pitch(pitch_type='uefa', pitch_length = 105, pitch_width=68,pitch_color='grass')
    fig1,ax1 = pitch.draw()
    sns.kdeplot(
        player_df["act_x"],
        player_df["act_y"],
        shade="True",
        label="Actual",
        legend=True,
    )
    ax1.set_title("Actual Heatmap for " + position,size=20)
    pitch = Pitch(pitch_type='uefa', pitch_length = 105, pitch_width=68,pitch_color='grass')
    fig2,ax2 = pitch.draw()
    sns.kdeplot(
        player_df["pred_x"],
        player_df["pred_y"],
        shade="True",
        label="Prediction",
        legend=True,
        color="red",
    )
    ax2.set_title("Predicted Heatmap for " + position,size=20)
    return fig1,ax1,fig2,ax2


#Plot estimated players
def plot_estimated_players(events, event_id):
    event_ids = events.drop_duplicates(['event_id']).reset_index()
    curr_event = events[events['event_id'] == event_id].copy().reset_index(drop=True)
    ev_index = event_ids[event_ids['event_id'] == event_id].index
    pitch = Pitch(pitch_type='uefa', pitch_length = 105, pitch_width=68, pitch_color='grass')
    fig,ax = pitch.draw()
    fig.set_size_inches(12,6)
    nums = [1,2,3,4,5,6,7,8,9,10,11]
    nums2 = [12,13,14,15,16,17,18,19,20,21,22]
    ax.scatter(curr_event['pred_x'].head(11),curr_event['pred_y'].head(11),s=250, label='Predicted Positions',color='navy')
    #ax.scatter(105-curr_event['pred_x'].tail(11),68-curr_event['pred_y'].tail(11),s=250, label='Predicted Positions Team2',color='navy')
    ax.scatter(curr_event['act_x'].head(11),curr_event['act_y'].head(11),s=250, label='Actual Positions',color='white')
    ax.scatter(curr_event[curr_event['player_on_ball']==True]['act_x'],curr_event[curr_event['player_on_ball']==True]['act_y'],s=100,c='black',label='Ball Position')
    ax.annotate("", xy=event_ids.loc[ev_index+1][['ballx','bally']].values.flatten(), xytext=event_ids.loc[ev_index][['ballx','bally']].values.flatten(), alpha=0.8, arrowprops=dict(alpha=0.9,width=3,headlength=12.0,headwidth=12.0,color='k'),annotation_clip=False)
    for i,txt in enumerate(nums):
        ax.annotate(txt, [(curr_event['pred_x']).iloc[i],(curr_event['pred_y']).iloc[i]+1],size=16)
        ax.annotate(txt, [(curr_event['act_x']).iloc[i],(curr_event['act_y']).iloc[i]+1],size=16)
    #for i,txt in enumerate(nums):
    #    ax.annotate(txt, [(105-curr_event['pred_x']).iloc[i+11],(68-curr_event['pred_y']).iloc[i+11]+1],size=20)
        
    ax.legend(loc=2,prop={'size':18})
    return fig,ax

def _get_cell_indexes(
    x: Series[float], y: Series[float], l: int = l, w: int = w
) -> Tuple[Series[int], Series[int]]:
    xi = x.divide(105).multiply(l)
    yj = y.divide(68).multiply(w)
    xi = xi.astype('int64').clip(0, l - 1)
    yj = yj.astype('int64').clip(0, w - 1)
    return xi, yj

def _get_flat_indexes(x: Series[float], y: Series[float], l: int =l, w: int = w) -> Series[int]:
    xi, yj = _get_cell_indexes(x, y, l, w)
    return yj.rsub(w - 1).mul(l).add(xi)

def get_grid_points(l,w):
    dx = 105.0/l
    dy = 68.0/w
    xgrid = np.arange(l)*dx + dx/2.
    ygrid = np.arange(w)*dy + dy/2.
    ygrid = ygrid[::-1]
    
    pos = []
    for i in range( len(ygrid) ):
        for j in range( len(xgrid) ):
            target_position = np.array( [xgrid[j], ygrid[i]] )
            pos.append(target_position)
    return xgrid,ygrid,pos

"""
Other possibly useful functions
"""

# Get physical statistics of a team from their tracking by calculating distance covered from their speed over time.
# Code from Friends of Tracking Github - https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking.git
def get_player_physical_statistics_for_team(team_df, tracking_df, fps):
    team_summary = pd.DataFrame(index=team_df["name"])
    minutes = []
    distance = []
    walking = []
    jogging = []
    running = []
    sprinting = []

    for player in team_df["player_id"]:
        # search for first and last frames that we have a position observation for each player (when a player is not on the pitch positions are NaN)
        mins_column = "player_" + player + "_x"  # use player x-position coordinate
        dist_column = "player_" + player + "_speed"
        player_minutes = (
            (
                tracking_df[mins_column].last_valid_index()
                - tracking_df[mins_column].first_valid_index()
                + 1
            )
            / fps
            / 60.0
        )  # convert to minutes
        player_distance = (
            tracking_df[dist_column].sum() / fps / 1000
        )  # this is the sum of the distance travelled from one observation to the next (1/25 = 40ms) in km.

        # walking (less than 2 m/s)
        player_walking_distance = (
            tracking_df.loc[tracking_df[dist_column] < 2, dist_column].sum()
            / fps
            / 1000
        )
        # jogging (between 2 and 4 m/s)
        player_jogging_distance = (
            tracking_df.loc[
                (tracking_df[dist_column] >= 2) & (tracking_df[dist_column] < 4),
                dist_column,
            ].sum()
            / fps
            / 1000
        )
        # running (between 4 and 7 m/s)
        player_running_distance = (
            tracking_df.loc[
                (tracking_df[dist_column] >= 4) & (tracking_df[dist_column] < 7),
                dist_column,
            ].sum()
            / fps
            / 1000
        )
        # sprinting (greater than 7 m/s)
        player_sprinting_distance = (
            tracking_df.loc[tracking_df[dist_column] >= 7, dist_column].sum()
            / 25.0
            / 1000
        )

        minutes.append(player_minutes)
        distance.append(player_distance)
        walking.append(player_walking_distance)
        jogging.append(player_jogging_distance)
        running.append(player_running_distance)
        sprinting.append(player_sprinting_distance)

    team_summary["Minutes Played"] = minutes
    team_summary["Distance [km]"] = distance
    team_summary["Walking [km]"] = walking
    team_summary["Jogging [km]"] = jogging
    team_summary["Running [km]"] = running
    team_summary["Sprinting [km]"] = sprinting
    team_summary = team_summary.sort_values(["Minutes Played"], ascending=False)
    return team_summary
