import matplotlib.pyplot as plt
import numpy as np

#EXPERIMENT 1: ERROR PER ROLE
def create_role_error_plot(errors_df,std_train,std_test,positions):
    plt.style.reload_library()
    plt.style.use(['science','no-latex'])
    with plt.style.context(['science','no-latex','bright']):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,4))
        ax3.bar(range(len(errors_df)),errors_df['train_dist'], label='Train',width = 0.3,yerr=std_train['se'],ecolor='k', capsize=3)
        ax3.bar([r+0.3 for r in range(len(errors_df))],errors_df['test_dist'], label='Test',width = 0.3,yerr=std_test['se'],ecolor='k', capsize=3)
        ax1.bar(range(len(errors_df)),errors_df['train_x_dist'], label='Train',width = 0.3,yerr=std_train['x_se'],ecolor='k', capsize=3)
        ax1.bar([r+0.3 for r in range(len(errors_df))],errors_df['test_x_dist'], label='Test',width = 0.3,yerr=std_test['x_se'],ecolor='k', capsize=3)
        ax2.bar(range(len(errors_df)),errors_df['train_y_dist'], label='Train',width = 0.3,yerr=std_train['y_se'],ecolor='k', capsize=3)
        ax2.bar([r+0.3 for r in range(len(errors_df))],errors_df['test_y_dist'], label='Test',width = 0.3,yerr=std_test['y_se'],ecolor='k', capsize=3)
        ax1.set_xticks(np.array(range(len(errors_df))) + 0.1 / 2)
        ax2.set_xticks(np.array(range(len(errors_df))) + 0.1 / 2)
        ax3.set_xticks(np.array(range(len(errors_df))) + 0.1 / 2)
        ax1.set_xlabel('')
        ax1.set_xticklabels(sorted(positions['position'].unique()), rotation= 80,size=10)
        ax2.set_xticklabels(sorted(positions['position'].unique()), rotation= 80,size=10)
        ax3.set_xticklabels(sorted(positions['position'].unique()), rotation= 80,size=10)
        ax1.set_yticks(range(int(np.ceil(errors_df['test_dist'].max()))+1))
        ax2.set_yticks(range(int(np.ceil(errors_df['test_dist'].max()))+1))
        ax3.set_yticks(range(int(np.ceil(errors_df['test_dist'].max()))+1))
        ax3.set_title('Mean Euclidean Distance Error (metres)', fontsize=10)
        ax1.set_title('Mean X Distance Error (metres)', fontsize=10)
        ax2.set_title('Mean Y Distance Error (metres)', fontsize=10)
        ax1.legend(loc=2)
    return fig


#Experiment 2: ERROR OVER OBSERVATION
def create_error_across_observation_time_plot(agent_imputer,lstm_results,gnn_results,xgboost_results):
    bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    means_ai = []
    std_ai = []
    means_lstm = []
    std_lstm = []
    means_gnn = []
    std_gnn = []
    means_xg = []
    std_xg = []
    means_ai_next = []
    std_ai_next = []
    means_lstm_next = []
    std_lstm_next = []
    means_gnn_next = []
    std_gnn_next = []
    means_xg_next = []
    std_xg_next = []
    
    #Get the mean and confidence interval of the errors for each bin (e.g. between 1 and 2 seconds since last observation), for all models
    for i in range(len(bins)):
        means_ai.append(agent_imputer[(agent_imputer['time_since_seen'] < bins[i]) & (agent_imputer['time_since_seen'] >= bins[i-1])]['dist'].mean())
        std_ai.append((agent_imputer[(agent_imputer['time_since_seen'] < bins[i]) & (agent_imputer['time_since_seen'] >= bins[i-1])]['dist'].std() / np.sqrt(len(agent_imputer[(agent_imputer['time_since_seen'] < bins[i]) & (agent_imputer['time_since_seen'] >= bins[i-1])])))*1.959)
        means_xg.append(xgboost_results[(xgboost_results['time_since_seen'] < bins[i]) & (xgboost_results['time_since_seen'] >= bins[i-1])]['dist'].mean())
        std_xg.append((xgboost_results[(xgboost_results['time_since_seen'] < bins[i]) & (xgboost_results['time_since_seen'] >= bins[i-1])]['dist'].std() / np.sqrt(len(xgboost_results[(xgboost_results['time_since_seen'] < bins[i]) & (xgboost_results['time_since_seen'] >= bins[i-1])])))*1.959)
        means_gnn.append(gnn_results[(gnn_results['time_since_seen'] < bins[i]) & (gnn_results['time_since_seen'] >= bins[i-1])]['dist'].mean())
        std_gnn.append((gnn_results[(gnn_results['time_since_seen'] < bins[i]) & (gnn_results['time_since_seen'] >= bins[i-1])]['dist'].std() / np.sqrt(len(gnn_results[(gnn_results['time_since_seen'] < bins[i]) & (gnn_results['time_since_seen'] >= bins[i-1])])))*1.959)
        means_lstm.append(lstm_results[(lstm_results['time_since_seen'] < bins[i]) & (lstm_results['time_since_seen'] >= bins[i-1])]['dist'].mean())
        std_lstm.append((lstm_results[(lstm_results['time_since_seen'] < bins[i]) & (lstm_results['time_since_seen'] >= bins[i-1])]['dist'].std() / np.sqrt(len(lstm_results[(lstm_results['time_since_seen'] < bins[i]) & (lstm_results['time_since_seen'] >= bins[i-1])])))*1.959)
        means_ai_next.append(agent_imputer[(agent_imputer['time_next_seen'] < bins[i]) & (agent_imputer['time_next_seen'] >= bins[i-1])]['dist'].mean())
        std_ai_next.append((agent_imputer[(agent_imputer['time_next_seen'] < bins[i]) & (agent_imputer['time_next_seen'] >= bins[i-1])]['dist'].std() / np.sqrt(len(agent_imputer[(agent_imputer['time_next_seen'] < bins[i]) & (agent_imputer['time_next_seen'] >= bins[i-1])])))*1.959)
        means_xg_next.append(xgboost_results[(xgboost_results['time_next_seen'] < bins[i]) & (xgboost_results['time_next_seen'] >= bins[i-1])]['dist'].mean())
        std_xg_next.append((xgboost_results[(xgboost_results['time_next_seen'] < bins[i]) & (xgboost_results['time_next_seen'] >= bins[i-1])]['dist'].std() / np.sqrt(len(xgboost_results[(xgboost_results['time_next_seen'] < bins[i]) & (xgboost_results['time_next_seen'] >= bins[i-1])])))*1.959)
        means_gnn_next.append(gnn_results[(gnn_results['time_next_seen'] < bins[i]) & (gnn_results['time_next_seen'] >= bins[i-1])]['dist'].mean())
        std_gnn_next.append((gnn_results[(gnn_results['time_next_seen'] < bins[i]) & (gnn_results['time_next_seen'] >= bins[i-1])]['dist'].std() / np.sqrt(len(gnn_results[(gnn_results['time_next_seen'] < bins[i]) & (gnn_results['time_next_seen'] >= bins[i-1])])))*1.959)
        means_lstm_next.append(lstm_results[(lstm_results['time_next_seen'] < bins[i]) & (lstm_results['time_next_seen'] >= bins[i-1])]['dist'].mean())
        std_lstm_next.append((lstm_results[(lstm_results['time_next_seen'] < bins[i]) & (lstm_results['time_next_seen'] >= bins[i-1])]['dist'].std() / np.sqrt(len(lstm_results[(lstm_results['time_next_seen'] < bins[i]) & (lstm_results['time_next_seen'] >= bins[i-1])])))*1.959)
        
    with plt.style.context(['science','no-latex','bright']):
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,4))
        
    #Create 2 figures for last and next seen observation, setup for each model
    with plt.style.context(['science','no-latex','bright']):
        ax1.errorbar(bins, means_ai, yerr=std_ai, fmt="x",capsize=2,ls='solid',label='Agent Imputer',lw=2)
        ax1.errorbar(bins, means_gnn, yerr=std_gnn, fmt="x",capsize=2,ls='solid',label='GNN',lw=2)
        ax1.errorbar(bins, means_lstm, yerr=std_lstm, fmt="x",capsize=2,ls='solid',label='Time-Aware LSTM',lw=2)
        ax1.errorbar(bins, means_xg, yerr=std_xg, fmt="x",capsize=2,ls='solid',label='XGBoost',lw=2)
        ax1.set_ylabel('Mean Euclidean Distance Error (m)',size=15)
        ax1.set_xlabel('Time Since Last Observed (seconds)',size=15)
        ax1.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
        ax1.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
        ax1.set_xticklabels([0,2,4,6,8,10,12,14,16,18,20], fontsize=15)
        ax1.set_yticklabels([0,1,2,3,4,5,6,7,8,9,10], fontsize=15)
        ax2.errorbar(bins, means_ai_next, yerr=std_ai_next, fmt="x",capsize=2,ls='solid',label='Agent Imputer',lw=2)
        ax2.errorbar(bins, means_gnn_next, yerr=std_gnn_next, fmt="x",capsize=2,ls='solid',label='GNN',lw=2)
        ax2.errorbar(bins, means_lstm_next, yerr=std_lstm_next, fmt="x",capsize=2,ls='solid',label='Time-Aware LSTM',lw=2)
        ax2.errorbar(bins, means_xg_next, yerr=std_xg_next, fmt="x",capsize=2,ls='solid',label='XGBoost',lw=2)
        ax2.set_xlabel('Time until Next Observed (seconds)',size=15)
        ax2.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
        ax2.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
        ax2.set_xticklabels([0,2,4,6,8,10,12,14,16,18,20], fontsize=15)
        ax2.set_yticklabels([0,1,2,3,4,5,6,7,8,9,10], fontsize=15)
        ax2.legend(fontsize=17)
    return fig


#Experiment 3: Get distance error over game periods
def plot_error_over_game_periods(means_time_df):
    fig = plt.figure(figsize=(8,4))
    plt.plot(means_time_df['times'],means_time_df['means'])
    plt.fill_between(means_time_df['times'],means_time_df['means']-(means_time_df['std']/2),means_time_df['means']+(means_time_df['std']/2),alpha=0.3)
    plt.xlabel('Minutes Played',fontsize=13)
    plt.ylabel('Mean Euclidean Distance Error (m)',fontsize=13)
    plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95])
    plt.yticks([5.5,6,6.5,7,7.5])
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.xlim([0,95])
    return fig