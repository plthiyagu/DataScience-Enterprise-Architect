## Problem overview:
Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely 
helpful for the company because it can then accordingly plan its communication strategy to reach out 
to those customers and optimise its business model and revenue. 
Now, in order to predict, whether the customer would be interested in Vehicle insurance, you have 
information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), 
Policy (Premium, sourcing channel) etc.


## Data Description :

* **id**	Unique ID for the customer
* **Gender**	Gender of the customer
* **Age**   Age of the customer
* **Driving_License**	 Customer has a DL or not
* **Region_Code**	Unique code for the region of the customer
* **Previously_Insured**  Customer already has Vehicle Insurance or not  
* **Vehicle_Age**	Age of the Vehicle 
* **Vehicle_Damage** Vehicle previously damaged or not
* **Annual_Premium**	The amount customer needs to pay as premium in the year
* **Policy_Sales_Channel**	Anonymised Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.
* **Vintage**	Number of Days, Customer has been associated with the company
* **Response**	Customer is interested or mot





## Evaluation Metric:
The evaluation metric for this hackathon is ROC_AUC score.



## Public and Private split
The public leaderboard is based on 40% of test data, while final rank would be decided on 
remaining 60% of test data (which is private leaderboard)




## Guidelines for Final Submission:
Please ensure that your final submission includes the following:
Solution file containing the predicted Length of stay every case_id in the test set
Code file for reproducing the submission, note that it is mandatory to submit your code for a valid 
final submission