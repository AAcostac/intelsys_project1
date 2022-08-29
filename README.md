# Predicting data science job salaries with linear regression

The program in this folder, ```ds_salaries.py```, uses a multivariate linear regression to calculate an estimate of a data science position salary from a few of that position's characteristics.

The model is an [Ordinary least squares Linear Regression from the Scikit learn framework](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) and the dataset is taken from [Kaggle](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries) which in turn cites [ai-jobs.net](ai-jobs.net) as the aggregator.

The job characteristics used to predict its salary are:
- **Job title**: name of the position, one out of Data Scientist, Data Engineer, Data Analyst, and Machine Learning Engineer

- **Level of experience**: degree of competence in the position, possible values are entry level, intermediate, senior, and executive level.

- **Remote ratio**: overall amount of work done remotely expressed as one of three categories: no remote work, partially remote, and fully remote.

- **Company size**: the number of people working for the company, divided into three categories: less than 50, between 50 and 250, and more than 250. 

## How to run
- Download this repository running ```git clone https://github.com/AAcostac/intelsys_project1``` on a console
- Move into the local copy of the repository
- Run the command ```python ds_salaries.py```
- You will be prompted to input the job characteristics used on the prediction, input valid values for all of them.
- After completing all input the prediction will be displayed in USD and MXN.
- Optionally input "Yes" at the final prompt to input values for a prediction again. At this point use any other input to finish the program.

## Notes
- The program uses the following libraries, you may have to install them should they not be in your system already:
    <ul>
        <li>Pandas</li>
        <li>Matplotlib</li>
        <li>Scikit learn</li>
    <ul> 