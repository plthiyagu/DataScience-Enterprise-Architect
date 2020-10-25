## Problem overview
A leading pet adoption agency is planning to create a virtual tour experience for their customers 
showcasing all animals that are available in their shelter. To enable this tour experience, you 
are required to build a Machine Learning model that determines type and breed of the animal based 
on its physical attributes and other factors.

## Data Description :

* **pet_id** Unique Pet Id
* **issue_date** Date on which the pet was issued to the shelter
* **listing_date** date when the pet arrived at the shelter
* **condition** condition of the pet
* **color_type** color of the pet
* **length(m)** length of the pet (in meter)
* **height(cm)** height of the pet (in centimeter)
* **X1,X2** anonymous columns
* **breed_category** breed categoty of the pet(target variable)
* **pet_category** category of the pet(target variable)


## Evaluation Metric
Submissions are evaluated on the average of the F1 scores of both the target variables.


