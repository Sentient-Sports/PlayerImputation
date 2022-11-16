# Spatiotemporal Estimation of Player Positioning using Soccer Event Data

## Paper Description
Player tracking data has the potential to drive value for clubs and present new research opportunities in soccer analytics (e.g., physical metrics, space analysis and pitch control). However, this data is extremely expensive due to the advanced data collection process, meaning it is unaffordable to the vast majority of clubs. Therefore, in this work, we present a model to impute snapshots of player tracking data from event-based data which is far cheaper and more widespread. This model consists of a graph network of long short-term memory (LSTM) models and hence captures both the spatial and temporal structure of player positioning. Finally we apply imputed player positions to off-ball analyses such as pitch control and player physical metrics.

![image](https://user-images.githubusercontent.com/96203800/202048553-573ad414-2654-445e-a4e8-122bcc213153.png)

## Data 
This work and resources for this work were supported and provided by Bepro Group Ltd. This data consists of 34 matches of Events and Tracking data for a team in the K League 1. To access this data, enquire about it's availability at enquiries@bepro11.com.

## Versions 
Created using Python 3.8.8 and Anaconda. All packages used along with versions are contained within the `requirements.txt` file. For model training, we suggest using Google Colab to utilize their GPU service.

## Folders 
`BeProDataFormatting`: Contains notebooks used to get the required dataset in the required files to run the model.

`UtilFunctions`: Contains python files with functions used in other notebooks.

`ImputationModel`: Contains the notebook which runs the imputation model. This also contains all model architectures.

`Experiments`: Contains the model experiments and applications using imputed player positions

`ModelResults` and `data`: Data directories containing data when receiving correct data and running the notebooks

## Instructions
1. Create a folder called `Suwon_FC` within the `data` repository.
2. Create an `events` folder, and run the `BeProDataFormatting/BeProGetEvents.ipynb` notebook using API calls from Bepro to generate the event dataset for each game.
3. Create a `tracking` folder, and put each supplied tracking dataset into files for each game in chronological order. Next, run the `BeProDataFormatting/BeProGetTracking.ipynb` notebook to get data in csv format.
4. Run the `ImputationModel/PlayerPositionImputation.ipynb` notebook to run the model and extract results for imputed players. 
5. Save the imputed positions of players for test events using the functions at the bottom of the `ImputationModel/PlayerPositionImputation.ipynb` notebook. This will save model results in csv format in the `ModelResults` directory.
6. Run applications of the model and model experiemnts in the `Experiments\ModelApplications.ipynb` notebook
