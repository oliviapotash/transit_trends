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
rename_list = [['In.a.typical.month.BEFORE.the.outbreak..e.g..in.January...how.often.did.you......work.from.home..', 'work_from_home_before'],
               ['In.a.typical.month.BEFORE.the.pandemic..e.g..in.January.2020...how.often.did.you......work.from.home..', 'work_from_home_before'],
               ['In.the.past.month..i.e..DURING.the.outbreak...how.often.did.you......work.from.home..', 'work_from_home_during'],
               ['In.the.past.month..i.e..DURING.the.pandemic...how.often.did.you......work.from.home..', 'work_from_home_during'],
               ['In.the.past.month..how.often.did.you......work.from.home..', 'work_from_home_during'],
               ['Sometime.in.the.future..when.COVID.19.will.no.longer.be.a.threat..how.often.do.you.think.you.will......work.from.home..', 'work_from_home_future'],
               ['Sometime.in.the.future..when.COVID.19.will.no.longer.be.a.threat..how.often.do.you.think.you.will.use.the.following.modes.of.transportation...Public.Transit..Bus..Light.Rail..Streetcar..', 'public_transit_usage_future']]

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

# Recode 'work_from_home_before', 'work_from_home_during', and 'work_from_home_future' to numeric values
pooled_survey_data['work_from_home_during'] = pooled_survey_data['work_from_home_during'].replace({'Never': 0, 'Once a month or less': 1, 'A few times a month': 2, '1-2 days a week': 3, '3-4 days a week': 4, 'Everyday': 5})
pooled_survey_data['work_from_home_future'] = pooled_survey_data['work_from_home_future'].replace({'Never': 0, 'Once a month or less': 1, 'A few times a month': 2, '1-2 days a week': 3, '3-4 days a week': 4, 'Everyday': 5})

# Create column based on 'work_from_home_during' and 'work_from_home_future'
pooled_survey_data['wfh_change_estimated'] = np.where(pooled_survey_data['work_from_home_during'] == pooled_survey_data['work_from_home_future'], 'No Change', np.where(pooled_survey_data['work_from_home_during'] > pooled_survey_data['work_from_home_future'], 'Decreased WFH', 'Increased WFH'))

print(pooled_survey_data['wfh_change_estimated'].value_counts())




