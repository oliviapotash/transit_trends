import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel

###################################################################################################################

#       Data Cleaning and Preprocessing

###################################################################################################################


# Read each CSV file into a DataFrame
Wave1_cleaned_noids = pd.read_csv('C:\\Users\\olive\\Desktop\\UW Files\\_Research Project\\PacTrans Survey Analysis\\PacTrans_Covid_Survey_Waves\\Wave1_cleaned_noids.csv')
Wave2_fb_cleaned_noids = pd.read_csv('C:\\Users\\olive\\Desktop\\UW Files\\_Research Project\\PacTrans Survey Analysis\\PacTrans_Covid_Survey_Waves\\Wave2_fb_cleaned_noids.csv')
Wave2_prior_cleaned_noids = pd.read_csv('C:\\Users\\olive\\Desktop\\UW Files\\_Research Project\\PacTrans Survey Analysis\\PacTrans_Covid_Survey_Waves\\Wave2_prior_cleaned_noids.csv')
Wave3_fb_cleaned_noids = pd.read_csv('C:\\Users\\olive\\Desktop\\UW Files\\_Research Project\\PacTrans Survey Analysis\\PacTrans_Covid_Survey_Waves\\Wave3_fb_cleaned_noids.csv')
Wave3_prior_cleaned_noids = pd.read_csv('C:\\Users\\olive\\Desktop\\UW Files\\_Research Project\\PacTrans Survey Analysis\\PacTrans_Covid_Survey_Waves\\Wave3_prior_cleaned_noids.csv')
df_list = [Wave1_cleaned_noids, Wave2_fb_cleaned_noids, Wave2_prior_cleaned_noids, Wave3_fb_cleaned_noids, Wave3_prior_cleaned_noids]

# create list of column name pairs
rename_list = [['In.a.typical.month.BEFORE.the.outbreak..e.g..in.January...how.often.did.you......work.from.home..', 'wfh_before'],
               ['In.a.typical.month.BEFORE.the.pandemic..e.g..in.January.2020...how.often.did.you......work.from.home..', 'wfh_before'],
               ['In.the.past.month..i.e..DURING.the.outbreak...how.often.did.you......work.from.home..', 'wfh_during'],
               ['In.the.past.month..i.e..DURING.the.pandemic...how.often.did.you......work.from.home..', 'wfh_during'],
               ['In.the.past.month..how.often.did.you......work.from.home..', 'wfh_during'],
               ['Sometime.in.the.future..when.COVID.19.will.no.longer.be.a.threat..how.often.do.you.think.you.will......work.from.home..', 'wfh_future'],
               ['In.the.past.month..i.e..DURING.the.outbreak...how.often.did.you.use.the.following.modes.of.transportation...Public.Transit..Bus..Light.Rail..Streetcar..', 'transit_usage_during'],
               ['In.the.past.month..i.e..DURING.the.pandemic...how.often.did.you.use.the.following.modes.of.transportation...Public.Transit..Bus..Light.Rail..Streetcar..', 'transit_usage_during'],
               ['In.the.past.month..how.often.did.you.use.the.following.modes.of.transportation...Public.Transit..Bus..Light.Rail..Streetcar..', 'transit_usage_during'],
               ['Sometime.in.the.future..when.COVID.19.will.no.longer.be.a.threat..how.often.do.you.think.you.will.use.the.following.modes.of.transportation...Public.Transit..Bus..Light.Rail..Streetcar..', 'transit_usage_future'],
               ['What.is.your.annual.household.income.level..', 'annual_hh_income'],
               ['What.is.your.annual.household.income.level.', 'annual_hh_income'],
               ['Which.of.the.following.currently.live.with.you...Check.all.that.apply..', 'hh_size']]

# iterate over list of dataframes
for df in df_list:
    # iterate over list of column name pairs
    for old_col, new_col in rename_list:
        # rename columns
        df.rename(columns={old_col: new_col}, inplace=True)

# join annual_hh_income column from Wave1_cleaned_noids to Wave2_prior_cleaned_noids and Wave3_prior_cleaned_noids based on user_id
Wave2_prior_cleaned_noids = Wave2_prior_cleaned_noids.merge(Wave1_cleaned_noids[['user_id', 'annual_hh_income']], on='user_id', how='left')
Wave3_prior_cleaned_noids = Wave3_prior_cleaned_noids.merge(Wave1_cleaned_noids[['user_id', 'annual_hh_income']], on='user_id', how='left')

# join hh_size column from Wave1_cleaned_noids to Wave2_prior_cleaned_noids and Wave3_prior_cleaned_noids based on user_id
Wave2_prior_cleaned_noids = Wave2_prior_cleaned_noids.merge(Wave1_cleaned_noids[['user_id', 'hh_size']], on='user_id', how='left')
Wave3_prior_cleaned_noids = Wave3_prior_cleaned_noids.merge(Wave1_cleaned_noids[['user_id', 'hh_size']], on='user_id', how='left')

# Identify common columns
common_columns = list(set(Wave1_cleaned_noids.columns) & set(Wave2_fb_cleaned_noids.columns) & set(Wave2_prior_cleaned_noids.columns) & set(Wave3_fb_cleaned_noids.columns) & set(Wave3_prior_cleaned_noids.columns))

# Filter out unique columns and reorder columns in each DataFrame
Wave1_cleaned_noids = Wave1_cleaned_noids[common_columns]
Wave2_fb_cleaned_noids = Wave2_fb_cleaned_noids[common_columns]
Wave2_prior_cleaned_noids = Wave2_prior_cleaned_noids[common_columns]
Wave3_fb_cleaned_noids = Wave3_fb_cleaned_noids[common_columns]
Wave3_prior_cleaned_noids = Wave3_prior_cleaned_noids[common_columns]

# Concatenate DataFrames into a single DataFrame
pooled_survey_data = pd.concat([Wave1_cleaned_noids, Wave2_fb_cleaned_noids, Wave2_prior_cleaned_noids, Wave3_fb_cleaned_noids, Wave3_prior_cleaned_noids], ignore_index=True)

# Save the pooled data to a new CSV file
pooled_survey_data.to_csv('pooled_survey_data.csv', index=False)

# Read the pooled data from the new CSV file
pooled_survey_data = pd.read_csv('pooled_survey_data.csv')

# Remove rows with NA values in specified columns
pooled_survey_data = pooled_survey_data.dropna(subset=["wfh_during", "wfh_future", "transit_usage_during", "transit_usage_future", "annual_hh_income", "hh_size"])

###################################################################################################################

#       Household Size Calculation and Income Categorization

###################################################################################################################


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
pooled_survey_data["hh_size"] = pooled_survey_data["hh_size"].apply(calculate_household_size)

# # Display table for hh_size
# print(pd.Series(pooled_survey_data["hh_size"]).value_counts())

# Create table for monthly 2020 WA State median income levels
WA_med_income_2020 = pd.DataFrame({"Household_Size": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   "Monthly_Median_Income_2020":
                                          [4237, 5541, 6845, 8149, 9452, 10756, 11001, 11246, 11491, 11736]})

# Add column in WA_med_income_2020_mo for annual income and annual income thresholds
WA_med_income_2020["Annual_Median_Income_2020"] = WA_med_income_2020["Monthly_Median_Income_2020"] * 12
WA_med_income_2020["Low_Income_Thresh_2020"] = WA_med_income_2020["Annual_Median_Income_2020"] * 0.667
WA_med_income_2020["High_Income_Thresh_2020"] = WA_med_income_2020["Annual_Median_Income_2020"] * 2


# convert annual_hh_income from range to a single value in middle of range
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


# apply convert_annual_income to annual_hh_income column
pooled_survey_data["annual_hh_income_avg"] = pooled_survey_data["annual_hh_income"].apply(convert_annual_income)

# # Display table for annual_hh_income_avg
# print(pd.Series(pooled_survey_data["annual_hh_income_avg"]).value_counts())
#
# # print number of all, not just unique, values in annual_hh_income_avg
# print("count of all Annual Income responses: ", len(pooled_survey_data["annual_hh_income_avg"]))


# Function to categorize income based on thresholds
def categorize_income(row):
    household_size = row['hh_size']

    # Look up the thresholds based on household size
    low_thresh = WA_med_income_2020.loc[WA_med_income_2020['Household_Size'] == household_size, 'Low_Income_Thresh_2020'].values[0]
    high_thresh = WA_med_income_2020.loc[WA_med_income_2020['Household_Size'] == household_size, 'High_Income_Thresh_2020'].values[0]

    if row['annual_hh_income_avg'] < low_thresh:
        return 'Low'
    elif row['annual_hh_income_avg'] > high_thresh:
        return 'High'
    else:
        return 'Middle'


# Apply the categorize_income function to create the 'income_category' column
pooled_survey_data['income_category'] = pooled_survey_data.apply(lambda row: categorize_income(row), axis=1)

# Create binary variable for low-income
pooled_survey_data['Low_Income'] = np.where(pooled_survey_data['income_category'] == 'Low', 1, 0)

# Create binary variable for middle-income
pooled_survey_data['Middle_Income'] = np.where(pooled_survey_data['income_category'] == 'Middle', 1, 0)

# Create binary variable for high-income
pooled_survey_data['High_Income'] = np.where(pooled_survey_data['income_category'] == 'High', 1, 0)

###################################################################################################################

#       Create and sort counts of WFH and Transit Usage, recode parameters to numeric values,
#       and create diff columns (anticipated change) for WFH and Transit Usage

###################################################################################################################

# Return a Series containing counts of unique values and sort by index
wfh_during_counts = pooled_survey_data['wfh_during'].value_counts().reindex(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])
wfh_future_counts = pooled_survey_data['wfh_future'].value_counts().reindex(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])
transit_usage_during_counts = pooled_survey_data['transit_usage_during'].value_counts().reindex(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])
transit_usage_future_counts = pooled_survey_data['transit_usage_future'].value_counts().reindex(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])

ordinal_mapping = {
    'Never': 0,
    'Once a month or less': 1,
    'A few times a month': 2,
    '1-2 days a week': 3,
    '3-4 days a week': 4,
    'Everyday': 5
}

# Recode parameters to numeric values
pooled_survey_data['wfh_during'] = pooled_survey_data['wfh_during'].map(ordinal_mapping)
pooled_survey_data['wfh_future'] = pooled_survey_data['wfh_future'].map(ordinal_mapping)
pooled_survey_data['transit_usage_during'] = pooled_survey_data['transit_usage_during'].map(ordinal_mapping)
pooled_survey_data['transit_usage_future'] = pooled_survey_data['transit_usage_future'].map(ordinal_mapping)

# Create column based on diff between 'wfh_during' and 'wfh_future'
pooled_survey_data['wfh_change_estimated'] = np.where(pooled_survey_data['wfh_during'] == pooled_survey_data['wfh_future'], 'No Change', np.where(pooled_survey_data['wfh_during'] > pooled_survey_data['wfh_future'], 'Decreased WFH', 'Increased WFH'))
# print the counts of unique values in 'wfh_change_estimated'
print(pooled_survey_data['wfh_change_estimated'].value_counts())

# Create column based on diff between transit_usage_during and transit_usage_future
pooled_survey_data['transit_change_estimated'] = np.where(pooled_survey_data['transit_usage_during'] == pooled_survey_data['transit_usage_future'], 'No Change', np.where(pooled_survey_data['transit_usage_during'] > pooled_survey_data['transit_usage_future'], 'Decreased Transit', 'Increased Transit'))
# print the counts of unique values in 'transit_change_estimated'
print(pooled_survey_data['transit_change_estimated'].value_counts())


###################################################################################################################

#       Create visualizations

###################################################################################################################


# # Create side-by-side bar chart for 'wfh_during' and 'wfh_future'
# fig, ax = plt.subplots()
# bar_width = 0.35
# index = np.arange(6)
# wfh_during_bar = ax.bar(index, wfh_during_counts, bar_width, label='In the past month (i.e., DURING the pandemic) how often did you work from home?')
# wfh_future_bar = ax.bar(index + bar_width, wfh_future_counts, bar_width, label='Sometime in the future (when COVID-19 will no longer be a threat) how often do you think you will work from home?')
# ax.set_xlabel('Frequency of WFH')
# ax.set_ylabel('Count')
# ax.set_title('')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])
# # add data labels on each bar
# for index, value in enumerate(wfh_during_counts):
#     plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
# for index, value in enumerate(wfh_future_counts):
#     plt.text(index + bar_width, value + 0.1, str(value), ha='center', va='bottom')
# plt.tight_layout()
# ax.legend()
# plt.show()

# # Create side-by-side bar chart for 'transit_usage_during_counts' and 'transit_usage_future_counts'
# fig, ax = plt.subplots()
# bar_width = 0.35
# index = np.arange(6)
# transit_during_bar = ax.bar(index, transit_usage_during_counts, bar_width, label='In the past month (i.e., DURING the pandemic) how often did you use public transit?')
# transit_future_bar = ax.bar(index + bar_width, transit_usage_future_counts, bar_width, label='Sometime in the future (when COVID-19 will no longer be a threat) how often do you think you will use public transit?')
# ax.set_xlabel('Frequency of Transit Use')
# ax.set_ylabel('Count')
# ax.set_title('')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])
# # add data labels on each bar
# for index, value in enumerate(transit_usage_during_counts):
#     plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
# for index, value in enumerate(transit_usage_future_counts):
#     plt.text(index + bar_width, value + 0.1, str(value), ha='center', va='bottom')
# plt.tight_layout()
# ax.legend()
# plt.show()

# # create bar chart of transit_usage_future
# transit_usage_future_chart = transit_usage_future_counts.plot(kind='bar')
# # add data labels on each bar
# for index, value in enumerate(transit_usage_future_counts):
#     plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
# plt.tight_layout()
# plt.xticks(rotation=0)
# # plt.show()

# # Create cross-tabulation (crosstab)
# cross_tab = pd.crosstab(pooled_survey_data['wfh_future'], pooled_survey_data['transit_usage_future'])
# # Convert index and columns to integers
# cross_tab.index = cross_tab.index.astype(int)
# cross_tab.columns = cross_tab.columns.astype(int)
# # Reverse the order of the rows
# cross_tab = cross_tab.iloc[::-1, :]
# # Create heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(cross_tab, cmap='Blues', annot=True, fmt="d", linewidths=.5)
# plt.title('Heatmap of WFH Future vs Transit Usage Future')
# plt.xlabel('Transit Usage Future')
# plt.ylabel('WFH Future')
# plt.show()

# # Mapping of numeric values to labels
# freq_labels = ['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday']
#
# # Create a figure and axis object for the subplots
# fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey=True)
# # set y-axis limit as 0.5
# axes[0, 0].set_ylim(0, 0.5)
#
# # Iterate over each possible value of 'wfh_future'
# for value, ax in zip(range(6), axes.flatten()):
#     # Filter the data to include only the current value
#     subset = pooled_survey_data[pooled_survey_data['wfh_future'] == value]
#     # Count the number of times each value of 'transit_usage_future' occurs
#     transit_usage_future_subset = subset['transit_usage_future'].value_counts(normalize=True)  # Normalize frequencies
#     # Create a bar chart of the transit_usage_future
#     transit_usage_future_chart = transit_usage_future_subset.plot(kind='bar', ax=ax)
#     # Add data labels on each bar (with normalized counts)
#     for index, val in enumerate(transit_usage_future_subset):
#         ax.text(index, val + 0.01, f'{val:.2f}', ha='center', va='bottom')
#     # Set the x-axis tick locations and labels
#     ax.tick_params(axis='x', labelsize=9)
#     ax.set_xticks(range(len(freq_labels)))
#     ax.set_xticklabels(freq_labels, rotation=0, ha='center')
#     # set y-axis label
#     ax.set_ylabel('Percentage')
#     # set x-axis label
#     ax.set_xlabel('Post-Pandemic Transit Usage Frequency')
#     # Set subplot title with corresponding label
#     ax.set_title('Post-Pandemic WFH Frequency: ' + freq_labels[value])
#
# plt.tight_layout()
# plt.show()

# income_category_counts = pooled_survey_data['income_category'].value_counts()
# # Define the desired order of bars
# desired_order = ["Low", "Middle", "High"]
# # Use the reindex method to set the order
# income_category_counts = income_category_counts.reindex(desired_order)
# income_chart = income_category_counts.plot(kind='bar')
# # Add data labels on each bar
# for index, value in enumerate(income_category_counts):
#     plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
# # rotate x-axis labels to be horizontal
# plt.xticks(rotation=0)
# plt.show()

# # Return a Series containing counts of unique values
# wfh_change_counts = pooled_survey_data['wfh_change_estimated'].value_counts()
# # Use the reindex to set the order
# wfh_change_estimated = wfh_change_counts.reindex(["Decreased WFH", "No Change", "Increased WFH"])
# wfh_change_chart = wfh_change_estimated.plot(kind='bar')
# # Add data labels on each bar
# for index, value in enumerate(wfh_change_counts):
#     plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
# # rotate x-axis labels to be horizontal
# plt.xticks(rotation=0)
# plt.show()

# # Return a Series containing counts of unique values
# transit_change_counts = pooled_survey_data['transit_change_estimated'].value_counts()
# # Use the reindex to set the order
# transit_change_estimated = transit_change_counts.reindex(["Decreased Transit", "No Change", "Increased Transit"])
# transit_change_chart = transit_change_estimated.plot(kind='bar')
# # Add data labels on each bar
# for index, value in enumerate(transit_change_estimated):
#     plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
# # rotate x-axis labels to be horizontal
# plt.xticks(rotation=0)
# plt.show()

###################################################################################################################

#       Create ordered logit model

###################################################################################################################

# create ordered logit model where independent variable is 'wfh_future' and dependent variables are
# 'transit_usage_future', 'Middle_Income', 'High_Income'

mod_log = OrderedModel(pooled_survey_data['wfh_future'],
                       (pooled_survey_data[['transit_usage_future', 'Middle_Income', 'High_Income']]), distr='logit')
res_log = mod_log.fit(method='bfgs', disp=False)
print(res_log.summary())
print("Log-likelihood of model: ", res_log.llf)
print("loglikelihood of model without explanatory variables: ", res_log.llnull)
print("Likelihood ratio chi-squared statistic: ", res_log.llr)
print("chi-squared probability of getting a log-likelihood ratio statistic greater than llr: ", res_log.llr_pvalue)
# compute the f-test for the model
print("F-test for the model: ", res_log.f_test(np.eye(8)))

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

f = open('res_log_cross_sec.tex', 'w')
f.write(beginningtex)
f.write(res_log.summary().as_latex())
f.write(endtex)
f.close()
