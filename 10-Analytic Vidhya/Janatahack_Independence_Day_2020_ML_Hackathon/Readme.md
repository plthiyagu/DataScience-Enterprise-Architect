## Problem overview
Topic Modeling for Research Articles
Researchers have access to large online archives of scientific articles. As a consequence, 
finding relevant articles has become more difficult. Tagging or topic modelling provides a way 
to give token of identification to research articles which facilitates recommendation and search process.

Given the abstract and title for a set of research articles, predict the topics for each article 
included in the test set. Note that a research article can possibly have more than 1 topic. 
The research article abstracts and titles are sourced from the following 6 topics: 

1. Computer Science
2. Physics
3. Mathematics
4. Statistics
5. Quantitative Biology
6. Quantitative Finance

## Data Description :

* **ID** Unique ID for each article
* **TITLE** Title of the research article
* **ABSTRACT**	 Abstract of the research article
* **Computer Science** Whether article belongs to topic computer science (1/0)
* **Physics** Whether article belongs to topic physics (1/0)
* **Mathematics** Whether article belongs to topic Mathematics (1/0)
* **Statistics** Whether article belongs to topic Statistics (1/0)
* **Quantitative Biology** Whether article belongs to topic Quantitative Biology (1/0)
* **Quantitative Finance** Whether article belongs to topic Quantitative Finance (1/0)


## Evaluation Metric
Submissions are evaluated on micro F1 Score between the predicted and observed topics for 
each article in the test set



## Public and Private split
Test reviews are further divided into Public (40%) and Private (60%)
Your initial responses will be checked and scored on the Public data.
The final rankings would be based on your private score which will be published once the competition is over.

 

## Guidelines for Final Submission
Please ensure that your final submission includes the following:

Solution file containing the predicted 1/0 for each of the 6 topics for every research article in the test set
Code file for reproducing the submission, note that it is mandatory to submit your code for a valid final submission