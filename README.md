# Spatiotemporal Estimation of Player Positioning using Soccer Event Data

## Paper Description
Player tracking data has the potential to drive value for clubs and present new research opportunities in soccer analytics (e.g., physical metrics, space analysis and pitch control). However, this data is extremely expensive due to the advanced data collection process, meaning it is unaffordable to the vast majority of clubs. Therefore, in this work, we present a model to impute snapshots of player tracking data from event-based data which is far cheaper and more widespread. This model consists of a graph network of long short-term memory (LSTM) models and hence captures both the spatial and temporal structure of player positioning. Finally we apply imputed player positions to off-ball analyses such as pitch control and player physical metrics.

## Data 
This work and resources for this work were supported and provided by Bepro Group Ltd. This data consists of 34 matches of Events and Tracking data for a team in the K League 1. To access this data, enquire about it's availability at enquiries@bepro11.com.

## Versions 
Created using Python 3.8.8, Anaconda installation with a few other libraries such as `pandas` and `mplsoccer`.

## Folders 
`BeProDataFormatting`: Contains notebooks used to get the required dataset in the required files to run the model

`UtilFunctions`: Contains python files with functions used in other notebooks

`ImputationModel`: Contains the notebook which runs the imputation model

## Instructions
1. Create a folder called `data/Suwon_FC` within the repository
2. Create an `events` folder, and run the `BeProGetEvents` notebook using API calls from BePro to generate the event dataset for each game
3. Create a `tracking` folder, and put each supplied tracking dataset into files for each game in chronological order. Next, run the `BeProGetTracking` notebook to get data in csv format.
4. Run the `PlayerPositionImputation` notebook to run the model and extract results for imputed players. 
5. If you would like to use your own dataset to train the model, do so by creating it in the same format as current data files.
6. If you would like to impute data using final model, run the `PlayerPositionImputation` notebook and save the final model

## *** WORK IN PROGRESS - NEEDS TIDYING UP AND FURTHER CHANGES BEFORE COMPLETION ***
