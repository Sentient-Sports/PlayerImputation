import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import math

#Loads input dataframe into baseline format and eccodes values to be appropriately used by XGBoost model. Gets the specific fold
def get_baseline_data(whole_df, whole_labels, fold):
    train = list(set(whole_df.index) - set(fold))
    folded_input_train = whole_df.loc[train]
    folded_input_test = whole_df.loc[fold]
    folded_label_train = whole_labels[train]
    folded_label_test = whole_labels[fold]
    
    sorted_wi_num = folded_input_train[['ballx','prev_player_x','next_player_x','bally','prev_player_y','next_player_y','av_player_x','av_player_y','time_since_last_pred','player_id','event_id','player_on_ball','prev_player_time','next_player_time']]
    sorted_wi_cat = folded_input_train[['position','event_type','team_on_ball','player_on_ball','goal_diff']]
    sorted_wi_num_test = folded_input_test[['ballx','prev_player_x','next_player_x','bally','prev_player_y','next_player_y','av_player_x','av_player_y','time_since_last_pred','player_id','event_id','player_on_ball','prev_player_time','next_player_time']]
    sorted_wi_cat_test = folded_input_test[['position','event_type','team_on_ball','player_on_ball','goal_diff']]

    xg_categories = pd.DataFrame([])
    xg_categories['position'] = pd.Categorical(sorted_wi_cat['position']).codes
    xg_categories['event_type'] = pd.Categorical(sorted_wi_cat['event_type']).codes
    xg_categories['team_on_ball'] = pd.Categorical(sorted_wi_cat['team_on_ball']).codes
    xg_categories['player_on_ball'] = pd.Categorical(sorted_wi_cat['player_on_ball']).codes
    xg_categories['goal_diff'] = pd.Categorical(sorted_wi_cat['goal_diff']).codes
    
    xg_categories_test = pd.DataFrame([])
    xg_categories_test['position'] = pd.Categorical(sorted_wi_cat_test['position']).codes
    xg_categories_test['event_type'] = pd.Categorical(sorted_wi_cat_test['event_type']).codes
    xg_categories_test['team_on_ball'] = pd.Categorical(sorted_wi_cat_test['team_on_ball']).codes
    xg_categories_test['player_on_ball'] = pd.Categorical(sorted_wi_cat_test['player_on_ball']).codes
    xg_categories_test['goal_diff'] = pd.Categorical(sorted_wi_cat_test['goal_diff']).codes
    
    base_x_train = sorted_wi_num
    base_x_train_cat = sorted_wi_cat
    base_x_test = sorted_wi_num_test
    base_x_test_cat = sorted_wi_cat_test
    base_y_train = np.array(folded_label_train)
    base_y_test = np.array(folded_label_test)
    xg_cat_train = xg_categories
    xg_cat_test = xg_categories_test
    return base_x_train,base_x_train_cat,base_x_test,base_x_test_cat,base_y_train,base_y_test,xg_cat_train,xg_cat_test

#BASELINE 1: Gets the average location between the previous and next location of the player
def baseline_1(x_test, y_test):
    pred_x = np.array((x_test['prev_player_x']+x_test['next_player_x']) / 2).reshape(len(x_test),1)
    pred_y = np.array((x_test['prev_player_y']+x_test['next_player_y']) / 2).reshape(len(pred_x),1)
    x_test['pred_x'] = pred_x
    x_test['pred_y'] = pred_y
    preds = np.hstack((pred_x,pred_y))
    base_dist_error = np.mean([np.linalg.norm(preds[i]-y_test[i]) for i in range(len(pred_x))])
    print("X Error:", np.mean(abs(y_test[:,0].reshape(len(pred_x),1) - pred_x)))
    print("Y Error:", np.mean(abs(y_test[:,1].reshape(len(pred_y),1) - pred_y)))
    return base_dist_error, x_test

#BASELINE 2: Gets the time-scaled average location between the previous and next location of the player
def baseline_2(x_test, y_test):
    x_test = x_test.sort_values(['player_id','event_id']).reset_index()
    known_preds = (x_test[x_test['player_on_ball'] == True]).index
    base_2_pred_x = []
    base_2_pred_y = []
    base_2_pred_x = base_2_pred_x + list(x_test['prev_player_x'][:known_preds[0]+1].values)
    base_2_pred_y = base_2_pred_y + list(x_test['prev_player_y'][:known_preds[0]+1].values)
    
    #Loop through predictions, find the sum of time since last seen and next seen to fill out predicted positions between two known locations
    for i in range(1,len(known_preds)):
        events_since_last_seen = x_test[known_preds[i-1]:known_preds[i]+1]
        cum_time = events_since_last_seen['time_since_last_pred'].head(-1).cumsum()
        sum_time = events_since_last_seen['time_since_last_pred'].sum()
        new_preds_x = list(((cum_time/sum_time) * (abs(x_test.loc[known_preds[i-1]]['prev_player_x'] - x_test.loc[known_preds[i]]['prev_player_x']))).values)
        if x_test.loc[known_preds[i-1]]['prev_player_x'] < x_test.loc[known_preds[i]]['prev_player_x']:
            ps_x = new_preds_x + x_test.loc[known_preds[i-1]]['prev_player_x']
        else:
            ps_x = abs(new_preds_x - x_test.loc[known_preds[i-1]]['prev_player_x'])

        new_preds_y = list(((cum_time/sum_time) * (abs(x_test.loc[known_preds[i-1]]['prev_player_y'] - x_test.loc[known_preds[i]]['prev_player_y']))).values)
        if x_test.loc[known_preds[i-1]]['prev_player_y'] < x_test.loc[known_preds[i]]['prev_player_y']:
            ps_y = new_preds_y + x_test.loc[known_preds[i-1]]['prev_player_y']
        else:
            ps_y = abs(new_preds_y - x_test.loc[known_preds[i-1]]['prev_player_y'])

        if np.isnan(ps_x).sum() > 0:
            ps_x = x_test[known_preds[i-1]:known_preds[i]]['prev_player_x']
            ps_y = x_test[known_preds[i-1]:known_preds[i]]['prev_player_y']

        base_2_pred_x = base_2_pred_x + list(ps_x)
        base_2_pred_y = base_2_pred_y + list(ps_y)
    base_2_pred_x = base_2_pred_x + list(x_test['prev_player_x'][known_preds[-1]+1:].values)
    base_2_pred_y = base_2_pred_y + list(x_test['prev_player_y'][known_preds[-1]+1:].values)
    x_test['predx'] = base_2_pred_x
    x_test['predy'] = base_2_pred_y
    x_test = x_test.sort_values('index')
    preds = [[a,b] for a,b in zip(x_test['predx'],x_test['predy'])]
    print("X Error:", np.mean(abs(y_test[:,0].reshape(len(x_test),1) - np.array(x_test['predx']).reshape(len(x_test),1))))
    print("Y Error:", np.mean(abs(y_test[:,1].reshape(len(x_test),1) - np.array(x_test['predy']).reshape(len(x_test),1))))
    base_dist_error = np.mean([math.dist(preds[i],y_test[i]) for i in range(len(base_2_pred_x))])
    return x_test,base_dist_error

#Baseline 3 is in the code already

#XGBoost Baseline
def all_xgboost_regression_model(x_train,y_train,x_test,y_test):
    model = MultiOutputRegressor(XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.8, subsample=0.3, learning_rate = 0.015, max_depth = 7, gamma = 5, n_estimators = 400))
    model.fit(x_train, y_train)
    predictions = pd.DataFrame(model.predict(x_test), columns=['X','Y'])
    train_preds = pd.DataFrame(model.predict(x_train), columns=['X','Y'])
    return predictions, train_preds, model

def xg_boost_baseline(base_x_train, xg_cat_train, base_x_test, xg_cat_test, base_y_train, base_y_test):
    base_x_train = base_x_train.drop('player_on_ball',axis=1)
    base_x_test = base_x_test.drop('player_on_ball',axis=1)
    xg_x_train = pd.concat([base_x_train.reset_index(drop=True),xg_cat_train.reset_index(drop=True)],axis=1)
    xg_x_test = pd.concat([base_x_test.reset_index(drop=True),xg_cat_test.reset_index(drop=True)],axis=1)
    xg_test_preds,xg_train_preds,model = all_xgboost_regression_model(xg_x_train,base_y_train,xg_x_test,base_y_test)
    xg_train_preds = np.array(xg_train_preds)
    xg_test_preds = np.array(xg_test_preds)
    print("Training")
    print("X: ", np.mean(abs(xg_train_preds[:,0].reshape(len(xg_train_preds),1)-base_y_train[:,0].reshape(len(xg_train_preds),1))))
    print("Y: ", np.mean(abs(xg_train_preds[:,1].reshape(len(xg_train_preds),1)-base_y_train[:,1].reshape(len(xg_train_preds),1))))
    print("Error: ", np.mean([math.dist(xg_train_preds[i],base_y_train[i]) for i in range(len(xg_train_preds))]))
    print("Testing")
    print("X: ", np.mean(abs(xg_test_preds[:,0].reshape(len(xg_test_preds),1)-base_y_test[:,0].reshape(len(xg_test_preds),1))))
    print("Y: ", np.mean(abs(xg_test_preds[:,1].reshape(len(xg_test_preds),1)-base_y_test[:,1].reshape(len(xg_test_preds),1))))
    print("Error: ", np.mean([math.dist(xg_test_preds[i],base_y_test[i]) for i in range(len(xg_test_preds))]))
    return np.mean([math.dist(xg_test_preds[i],base_y_test[i]) for i in range(len(xg_test_preds))]),xg_test_preds


