import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Read the Wave1_cleaned_noids data
survey_W1 = pd.read_csv("C:\\Users\\olive\\Desktop\\UW Files\\_Research Project\\PacTrans Survey Analysis\\PacTrans_Covid_Survey_Waves\\Wave1_cleaned_noids.csv",
                        na_values=["", "NA"])

# Variable names
VarName_W1 = survey_W1.columns

survey_W1 = survey_W1.rename(columns={VarName_W1[37]: "Public_Transit_Usage_W1",
                                      VarName_W1[101]: "Gender_W1",
                                      VarName_W1[102]: "Age_W1",
                                      VarName_W1[103]: "Race_W1",
                                      VarName_W1[104]: "Household_Size_W1",
                                      VarName_W1[105]: "Annual_Income_W1",
                                      VarName_W1[106]: "Education_W1",
                                      VarName_W1[108]: "Zip_Code_W1",
                                      VarName_W1[110]: "User_ID_W1",
                                      VarName_W1[111]: "Waves_W1"})

# Remove rows with NA values in specified columns
survey_W1 = survey_W1.dropna(subset=["Public_Transit_Usage_W1", "Race_W1", "Annual_Income_W1", "Household_Size_W1"])

# Group transit usage responses
survey_W1["Public_Transit_Usage_W1_grouped"] = np.select([survey_W1["Public_Transit_Usage_W1"] == "Never",
                                                          survey_W1["Public_Transit_Usage_W1"] == "Once a month or less",
                                                          survey_W1["Public_Transit_Usage_W1"].isin(
                                                              ["A few times a month", "1-2 days a week", "3-4 days a week", "Everyday"])],
                                                         ["Never", "Infrequent", "Frequent"])

# Convert the column to CategoricalDtype with specified categories and order
transit_categories = ['Never', 'Infrequent', 'Frequent']
survey_W1["Public_Transit_Usage_W1_grouped"] = pd.Categorical(survey_W1["Public_Transit_Usage_W1_grouped"],
                                                              categories=transit_categories, ordered=True)

# TODO delete following 2 lines
# # Factorize transit usage responses
# survey_W1["Public_Transit_Usage_W1_factor"] = pd.factorize(survey_W1["Public_Transit_Usage_W1_grouped"])[0]

# Group race responses
survey_W1["Race_W1_grouped"] = np.select([survey_W1["Race_W1"].isin(["American Indian or Alaska Native"]),
                                          survey_W1["Race_W1"].isin(["Asian", "Indian", "Asian-American", "subcontinent of Asia-India/punjab"]),
                                          survey_W1["Race_W1"].isin(["Black or African American"]),
                                          survey_W1["Race_W1"].isin(["Hispanic or Latino", "Latina", "Latino", "Mexican", "Chicana (indigenous Mexican and Hispanic)", "Hispanic"]),
                                          survey_W1["Race_W1"].isin(["Mixed", "biracial ", "Bi-racial", "Mix ", "Mixed race: asian and white",
                                                                    "mixed race", "Mixed unknown", "Mixed asian/white", "Multi racial ", "mixed white and American Indian",
                                                                    "Mixed, asian and white ", "Mixed - asian and white (yâ€™all should really make these options multi select)",
                                                                    "Multi", "Multiracial", "Muli-racial", "Asian Hispanic ", "Mixed asian/white", "Asian and white", "asian and white",
                                                                    "White and Asian", "White and Filipino ", "White and Native American", "Caucasian/American Indian",
                                                                    "Mixed race: asian and white", "White/South Asian"]),
                                          survey_W1["Race_W1"].isin(["Native Hawaiian or Other Pacific Islander"]),
                                          survey_W1["Race_W1"].isin(["White", "Italian", "Italian American", "Italian American ", "Middle Eastern", "Middle Eastern ", "Syrian", "Caucasian, why does your survey have â€œwhiteâ€\u009d? Pay attention! Black Lives Matter! "]),
                                          survey_W1["Race_W1"].isin(["Prefer not to answer", "Unknown", "Does it matter?", "Choose not to answer", "HUMAN", "Human", "Unknown, presumed white", "Prefer not to disclose", "Light skinned Jew "])],
                                         ["American Indian or Alaska Native", "Asian", "Black or African American", "Hispanic or Latino",
                                          "Mixed/Bi-racial", "Native Hawaiian or Other Pacific Islander", "White", "Prefer not to answer/Unknown"])

# Display table for Race_W1_grouped
print(pd.Series(survey_W1["Race_W1_grouped"]).value_counts())

# # Display the count of unique values in Household_Size_W1
# print("count of unique Household Size responses: ", len(survey_W1["Household_Size_W1"].unique()))


# Function to calculate the number of people in each household
def calculate_household_size(entry):
    # non-household categories
    nh_categories = ["Cat or Dog", "Non-household members (e.g. relatives, roommates)"]

    if entry == "None of the above" or set(entry.split(';')) <= set(nh_categories):
        return 1
    else:
        # Remove 'Cat or Dog' and 'Non-household members'
        categories = [category.strip() for category in entry.split(';') if category.strip() not in nh_categories]
        return len(categories)


# Apply the function to the specified column
survey_W1["Household_Num_W1"] = survey_W1["Household_Size_W1"].apply(calculate_household_size)

# Display table for Household_Num_W1
print(pd.Series(survey_W1["Household_Num_W1"]).value_counts())

# Create table for monthly 2020 WA State median income levels
WA_med_income_2020 = pd.DataFrame({"Household_Size": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   "Monthly_Median_Income_2020":
                                          [4237, 5541, 6845, 8149, 9452, 10756, 11001, 11246, 11491, 11736]})

# Add column in WA_med_income_2020_mo for annual income and annual income thresholds
WA_med_income_2020["Annual_Median_Income_2020"] = WA_med_income_2020["Monthly_Median_Income_2020"] * 12
WA_med_income_2020["Low_Income_Thresh_2020"] = WA_med_income_2020["Annual_Median_Income_2020"] * 0.667
WA_med_income_2020["High_Income_Thresh_2020"] = WA_med_income_2020["Annual_Median_Income_2020"] * 2


# TODO: figure out why 4 rows are missing after running this function
# convert Annual_Income_W1 from range to a single value in middle of range
def convert_annual_income(entry):
    if entry == "Prefer not to answer":
        return np.nan
    else:
        # Extract the numbers from the string
        numbers = re.findall(r'\d{1,3}(?:,\d{3})*', entry)
        # Convert the numbers to integers
        numbers = [int(number.replace(',', '')) for number in numbers]
        numbers = [int(number) for number in numbers]
        # Check if there are two numbers in the list (indicating a range)
        if len(numbers) == 2:
            # Calculate the average of the range
            return np.mean(numbers)
        else:
            # If there is only one number, return that number
            return numbers[0]


# apply convert_annual_income to Annual_Income_W1
survey_W1["Annual_Income_Avg_W1"] = survey_W1["Annual_Income_W1"].apply(convert_annual_income)

# Display table for Annual_Income_Avg_W1
print(pd.Series(survey_W1["Annual_Income_Avg_W1"]).value_counts())

# print number of all, not just unique, values in Annual_Income_Avg_W1. This should equal 1310.
print("count of all Annual Income responses: ", len(survey_W1["Annual_Income_Avg_W1"]))


# TODO: figure out why 4 rows are missing after running this function
# Function to categorize income based on thresholds
def categorize_income(row):
    household_size = row['Household_Num_W1']

    # Look up the thresholds based on household size
    low_thresh = WA_med_income_2020.loc[WA_med_income_2020['Household_Size'] == household_size, 'Low_Income_Thresh_2020'].values[0]
    high_thresh = WA_med_income_2020.loc[WA_med_income_2020['Household_Size'] == household_size, 'High_Income_Thresh_2020'].values[0]

    if row['Annual_Income_Avg_W1'] < low_thresh:
        return 'Low'
    elif row['Annual_Income_Avg_W1'] > high_thresh:
        return 'High'
    else:
        return 'Middle'

# Apply the categorize_income function to create the 'Income_Category' column
survey_W1['Income_Category_W1'] = survey_W1.apply(lambda row: categorize_income(row), axis=1)

# Create binary variable for low-income
survey_W1['Low_Income'] = np.where(survey_W1['Income_Category_W1'] == 'Low', 1, 0)

# Create binary variable for high-income
survey_W1['High_Income'] = np.where(survey_W1['Income_Category_W1'] == 'High', 1, 0)

# Create binary variable for white respondents
survey_W1['White'] = np.where(survey_W1['Race_W1_grouped'] == 'White', 1, 0)

# Create binary variable for non-white respondents
survey_W1['Non_White'] = np.where(survey_W1['Race_W1_grouped'] != 'White', 1, 0)

# print bar charts of independent and dependent variables
# transit_usage_chart = survey_W1["Public_Transit_Usage_W1_grouped"].value_counts().plot(kind='bar')

transit_usage_counts = survey_W1["Public_Transit_Usage_W1_grouped"].value_counts()
# Define the desired order of bars
desired_order = ["Never", "Infrequent", "Frequent"]
# Use the reindex method to set the order
transit_usage_counts = transit_usage_counts.reindex(desired_order)
transit_usage_chart = transit_usage_counts.plot(kind='bar')
# Add data labels on each bar
for index, value in enumerate(transit_usage_counts):
    plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
plt.tight_layout()
# rotate x-axis labels to be horizontal
plt.xticks(rotation=0)
plt.show()

race_response_counts = survey_W1["Race_W1_grouped"].value_counts()
race_chart = race_response_counts.plot(kind='bar')
# Add data labels on each bar
for index, value in enumerate(race_response_counts):
    plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
plt.tight_layout
plt.xticks(rotation=45, ha='right')
plt.show()

income_category_counts = survey_W1["Income_Category_W1"].value_counts()
# Define the desired order of bars
desired_order = ["Low", "Middle", "High"]
# Use the reindex method to set the order
income_category_counts = income_category_counts.reindex(desired_order)
income_chart = income_category_counts.plot(kind='bar')
# Add data labels on each bar
for index, value in enumerate(income_category_counts):
    plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
# rotate x-axis labels to be horizontal
plt.xticks(rotation=0)
plt.show()

# TODO create stacked bar chart for low, middle, high income with race categories

# Display table for ordered logit inputs
print(survey_W1["Public_Transit_Usage_W1_grouped"].dtype)
print(pd.Series(survey_W1["Low_Income"]).value_counts())
print(pd.Series(survey_W1["High_Income"]).value_counts())
print(pd.Series(survey_W1["White"]).value_counts())
print(pd.Series(survey_W1["Non_White"]).value_counts())

# Create ordered logit model
mod_log = OrderedModel(survey_W1["Public_Transit_Usage_W1_grouped"],
                       (survey_W1[["Low_Income", "Non_White"]]),
                       distr='logit')
res_log = mod_log.fit(method='bfgs', disp=False)
print(res_log.summary())
print("loglikelihood of model without explanatory variables: ", res_log.llnull)
print("chi-squared probability of getting a log-likelihood ratio statistic greater than llr: ", res_log.llr_pvalue)
print(res_log.llr)
print(res_log.llf)

params = res_log.params
conf = res_log.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))
print(np.exp(res_log.params))

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

f = open('myreg.tex', 'w')
f.write(beginningtex)
f.write(res_log.summary().as_latex())
f.write(endtex)
f.close()
