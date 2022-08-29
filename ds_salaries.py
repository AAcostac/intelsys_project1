# Predict data science job salary with linear regression
# By Adolfo Acosta Castro

# Dataset comes from https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries where the uploader in turn credits ai-jobs.net for aggregating the data

import os
import pandas as pd

# <----------- Load salary data ----------->
SALARIES_PATH = os.path.join("datasets", "archive")

def load_salaries_data(salaries_data_path=SALARIES_PATH):
    csv_path = os.path.join(salaries_data_path, "ds_salaries.csv")
    return pd.read_csv(csv_path)

ds_salaries = load_salaries_data()


# <----------- Clean data ----------->

# The first column in the data set are the row indices and thus is not needed
columns_to_delete = ds_salaries.columns[[0]]
ds_clean = ds_salaries.drop(columns_to_delete, axis=1)

# Remove the instances that aren't full-time (FT), there are too few of the other employment types in the data
ds_clean = ds_clean.drop(ds_clean[ds_clean["employment_type"] != "FT"].index)

# Remove the instances that have EX as experience level, there are too few of this experience level in the data
ds_clean = ds_clean.drop(ds_clean[ds_clean["experience_level"] == "EX"].index)

# Remove outliers with salaries above 300k
ds_clean = ds_clean.drop(ds_clean[ds_clean["salary_in_usd"] > 300000].index)

# Remove job titles with less than 40 instances
ds_clean = ds_clean.groupby("job_title").filter(lambda x: len(x) > 40)


# <----------- Prepare features for model fitting ----------->
predictor_names = ["job_title", "experience_level", "remote_ratio", "company_size"]
data = ds_clean[predictor_names]

target_name = "salary_in_usd"
target = ds_clean[target_name]


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

experience_preprocessor = OrdinalEncoder(categories=[['EN', 'MI', 'SE']])
comp_size_preprocessor = OrdinalEncoder(categories=[['S', 'M', 'L']])
nominal_preprocessor = OneHotEncoder()
numerical_preprocessor = StandardScaler()


from sklearn.compose import ColumnTransformer

exp_attributes = ["experience_level"]
comp_size_attributes = ["company_size"]
nom_attributes = ["job_title"]
num_attributes = ["remote_ratio"]

preprocessor = ColumnTransformer([
    ('experience-encoder', experience_preprocessor, exp_attributes),
    ('company-size-encoder', comp_size_preprocessor, comp_size_attributes),
    ('one-hot-encoder', nominal_preprocessor, nom_attributes),
    ('standard_scaler', numerical_preprocessor, num_attributes)])

data_prepared = preprocessor.fit_transform(data)


# <----------- Separate into train and test data ----------->
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_prepared, target, test_size=0.2, random_state=424)


# <----------- Fit linear regression ----------->
from sklearn import linear_model

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model
reg.fit(X_train, y_train)


# <----------- Check performace ----------->
from sklearn.metrics import mean_absolute_error

# Performance on training data
predicted_y = reg.predict(X_train)
train_mae = mean_absolute_error(y_train, predicted_y)

from sklearn.metrics import mean_absolute_error

# Performance on test data
predicted_y = reg.predict(X_test)
test_mae = mean_absolute_error(y_test, predicted_y)

print("Info on the model:")
print(f"This linear regression model has a MAE of {test_mae:.2f}")
print(f"and it's coefficients are: {reg.coef_}")

# <----------- Make predictions ----------->
from random import randrange

def validate_input(user_input, possible_values):
    return user_input in possible_values

def print_fail_mesage():
    messages = ["No, that's not right", "Wait, that's illegal", "No. Try again"]
    print(messages[randrange(len(messages))])

def print_dict(dict):
    for key in dict:
        print(f"\t{key}) {dict[key]}")

def get_integer_input(description, prompt, options, usekey):
    print(description)
    print_dict(options)
    while True:
        value = input(prompt)
        if value.isdigit() and validate_input(int(value), options.keys()):
            if usekey:
                return value
            return options[int(value)]
        else:
            print_fail_mesage()

def get_string_input(description, prompt, options):
    print(description)
    print_dict(options)
    while True:
        value = input(prompt)
        if validate_input(value, options.keys()):
            return value
        else:
            print_fail_mesage()

jobs = {
    1: "Data Scientist",
    2: "Data Engineer",
    3: "Data Analyst",
    4: "Machine Learning Engineer",
}

exp_levels = {
    "EN": "Entry-level",
    "MI": "Mid-level / Intermediate",
    "SE": "Senior-level / Expert",
}

remote_ratios = {
    0: "No remote work",
    50: "Partially remote",
    100: "Fully remote"
}

comp_sizes = {
    "S": "Small (<50 employees)",
    "M": "Medium (<250 employees)",
    "L": "Large (>250 employees)",
}


print("\nPredict data science job salaries (based on data from ai-jobs.net)!!!")
print("We're gonna need 4 values to make the prediction: job title, experience level, ratio of remote work, and company size")

while True:

    # Get job title
    job_title = get_integer_input("\nThe data science job can be one of these: ",
                                    "Input the data science job number: ", 
                                    jobs, usekey = False)

    # Get experience level
    exp_level = get_string_input("\nThe experience level can be one of these: ",
                                    "Input the experience level code: ", 
                                    exp_levels)

    # Get ratio of remote work
    remote_ratio = get_integer_input("\nThe overall amount of work done remotely can be: ",
                                    "Input the remote ratio ammount: ", 
                                    remote_ratios, usekey = True)
        
    # Get company size
    comp_size = get_string_input("\nThe company size can be one of these: ",
                                    "Input the company size code: ", 
                                    comp_sizes)
    
    # Process user input for prediction
    input_data = pd.DataFrame([(job_title, exp_level, remote_ratio, comp_size)], columns = predictor_names)
    input_data_prepared = preprocessor.transform(input_data)

    # Make prediction
    prediction = reg.predict(input_data_prepared)

    USD_TO_MXN = 20.10
    print(f"\nThe predicted salary is ${prediction[0]:.2f} USD or ${(prediction[0]*USD_TO_MXN):.2f} MXN")

    repeat = input("\nMake another prediction? (Yes/No): ")
    if not repeat.lower() in ["yes", "y", "yas", "yeah"]: 
        break

print("Happy data-sciencing!!!")