## Analyze AB Test Results
Within the framework this project, we tried to understand whether the company should implement a new page or keep the old page with following:

- **Probability based approach**
- **A/B test**
- **Regression approach**

##### Requirements
Python 3.6 (or higher)<br>
matplotlib 2.1 (or higher)<br>
numpy 1.10 (or higher)<br>
pandas 0.20 (or higher) <br>
statsmodels 0.8.0

### Probability based approach:
- We will try to find probability of an individual receiving the new page or an old page and what are the chances of those.

### A/B test:
- In A/B test we set up our hypothesis to test if new page results in better conversion or not
- We will simulate our user groups with respect to conversions
- We will find the the p_value
- We will also use an alternative approach to validate / double check our results and decide whether to reject the null hypothesis

### Regression Approach:

- We shall look at exploring two possible outcomes. Whether new page is better or not.
- By further adding geographic location of the users, we will attempt to find if any specific country had an impact on conversion
