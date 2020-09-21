## Problem overview:
Recent Covid-19 Pandemic has raised alarms over one of the most overlooked area to focus: Healthcare 
Management. While healthcare management has various use cases for using data science, patient length of 
stay is one critical parameter to observe and predict if one wants to improve the efficiency of the 
healthcare management in a hospital. 

This parameter helps hospitals to identify patients of high LOS risk (patients who will stay longer) at 
the time of admission. Once identified, patients with high LOS risk can have their treatment plan 
optimized to miminize LOS and lower the chance of staff/visitor infection. Also, prior knowledge of 
LOS can aid in logistics such as room and bed allocation planning.

Suppose you have been hired as Data Scientist of HealthMan – a not for profit organization dedicated 
to manage the functioning of Hospitals in a professional and optimal manner.
The task is to accurately predict the Length of Stay for each patient on case by case basis so that the 
Hospitals can use this information for optimal resource allocation and better functioning. 
The length of stay is divided into 11 different classes ranging from 0-10 days to more than 100 days.



## Data Description :
Train.zip contains 1 csv alongside the data dictionary that contains definitions for each variable
train.csv – File containing features related to patient, hospital and Length of stay on case basis
train_data_dict.csv – File containing the information of the features in train file



## Evaluation Metric:
The evaluation metric for this hackathon is 100*Accuracy Score.


## Public and Private split
The public leaderboard is based on 40% of test data, while final rank would be decided on remaining 
60% of test data (which is private leaderboard)



## Guidelines for Final Submission:
Please ensure that your final submission includes the following:
Solution file containing the predicted Length of stay every case_id in the test set
Code file for reproducing the submission, note that it is mandatory to submit your code for a valid 
final submission