import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel

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
               ['Sometime.in.the.future..when.COVID.19.will.no.longer.be.a.threat..how.often.do.you.think.you.will.use.the.following.modes.of.transportation...Public.Transit..Bus..Light.Rail..Streetcar..', 'transit_usage_future']]

# iterate over list of dataframes
for df in df_list:
    # iterate over list of column name pairs
    for old_col, new_col in rename_list:
        # rename columns
        df.rename(columns={old_col: new_col}, inplace=True)

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

wfh_during_counts = pooled_survey_data['wfh_during'].value_counts()
wfh_future_counts = pooled_survey_data['wfh_future'].value_counts()
wfh_during_counts = wfh_during_counts.reindex(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])
wfh_future_counts = wfh_future_counts.reindex(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])

# Create side-by-side bar chart for 'wfh_during' and 'wfh_future' using matplotlib
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(6)
wfh_during_bar = ax.bar(index, wfh_during_counts, bar_width, label='In the past month (i.e., DURING the pandemic) how often did you work from home?')
wfh_future_bar = ax.bar(index + bar_width, wfh_future_counts, bar_width, label='Sometime in the future (when COVID-19 will no longer be a threat) how often do you think you will work from home?')
ax.set_xlabel('Frequency of WFH')
ax.set_ylabel('Count')
ax.set_title('')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])
# add data labels on each bar
for index, value in enumerate(wfh_during_counts):
    plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
for index, value in enumerate(wfh_future_counts):
    plt.text(index + bar_width, value + 0.1, str(value), ha='center', va='bottom')
plt.tight_layout()
ax.legend()
plt.show()

# create bar chart of transit_usage_future
transit_usage_future_counts = pooled_survey_data['transit_usage_future'].value_counts()
transit_usage_future_counts = transit_usage_future_counts.reindex(['Never', 'Once a month or less', 'A few times a month', '1-2 days a week', '3-4 days a week', 'Everyday'])
transit_usage_future_chart = transit_usage_future_counts.plot(kind='bar')
# add data labels on each bar
for index, value in enumerate(transit_usage_future_counts):
    plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
plt.tight_layout()
plt.xticks(rotation=0)
plt.show()

# TODO fix this scatterplot
# create scatterplot with 'transit_usage_future' on x-axis and 'wfh_future' on y-axis
plt.scatter(pooled_survey_data['transit_usage_future'], pooled_survey_data['wfh_future'])
plt.xlabel('Transit Usage in the Future')
plt.ylabel('WFH in the Future')
plt.show()

# Recode parameters to numeric values
pooled_survey_data['wfh_during'] = pooled_survey_data['wfh_during'].replace({'Never': 0, 'Once a month or less': 1, 'A few times a month': 2, '1-2 days a week': 3, '3-4 days a week': 4, 'Everyday': 5})
pooled_survey_data['wfh_future'] = pooled_survey_data['wfh_future'].replace({'Never': 0, 'Once a month or less': 1, 'A few times a month': 2, '1-2 days a week': 3, '3-4 days a week': 4, 'Everyday': 5})
pooled_survey_data['transit_usage_future'] = pooled_survey_data['transit_usage_future'].replace({'Never': 0, 'Once a month or less': 1, 'A few times a month': 2, '1-2 days a week': 3, '3-4 days a week': 4, 'Everyday': 5})

# Create column based on diff between 'wfh_during' and 'wfh_future'
pooled_survey_data['wfh_change_estimated'] = np.where(pooled_survey_data['wfh_during'] == pooled_survey_data['wfh_future'], 'No Change', np.where(pooled_survey_data['wfh_during'] > pooled_survey_data['wfh_future'], 'Decreased WFH', 'Increased WFH'))

# print the counts of unique values in 'wfh_change_estimated'
print(pooled_survey_data['wfh_change_estimated'].value_counts())

# Return a Series containing counts of unique values
wfh_change_counts = pooled_survey_data['wfh_change_estimated'].value_counts()
# Use the reindex to set the order
wfh_change_estimated = wfh_change_counts.reindex(["Decreased WFH", "No Change", "Increased WFH"])
wfh_change_chart = wfh_change_estimated.plot(kind='bar')
# Add data labels on each bar
for index, value in enumerate(wfh_change_counts):
    plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
plt.tight_layout()
# rotate x-axis labels to be horizontal
plt.xticks(rotation=0)
plt.show()

