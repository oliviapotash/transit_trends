import pandas as pd
import numpy as np

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
                                      VarName_W1[105]: "Income_W1",
                                      VarName_W1[106]: "Education_W1",
                                      VarName_W1[108]: "Zip_Code_W1",
                                      VarName_W1[110]: "User_ID_W1",
                                      VarName_W1[111]: "Waves_W1"})

# Remove rows with NA values in specified columns
survey_W1 = survey_W1.dropna(subset=["Public_Transit_Usage_W1", "Race_W1", "Income_W1", "Household_Size_W1"])

# Group transit usage responses
survey_W1["Public_Transit_Usage_W1_grouped"] = np.select([survey_W1["Public_Transit_Usage_W1"] == "Never",
                                                          survey_W1["Public_Transit_Usage_W1"] == "Once a month or less",
                                                          survey_W1["Public_Transit_Usage_W1"].isin(["A few times a month", "1-2 days a week", "3-4 days a week"])],
                                                         ["Never", "Infrequent", "Frequent"])

# Display table for Public_Transit_Usage_W1_grouped
print(pd.Series(survey_W1["Public_Transit_Usage_W1_grouped"]).value_counts())

# Factorize transit usage responses
survey_W1["Public_Transit_Usage_W1_factor"] = pd.factorize(survey_W1["Public_Transit_Usage_W1_grouped"])[0]

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

# Display unique values for Household_Size_W1
# print(survey_W1["Household_Size_W1"].unique())

# Display the count of unique values in Household_Size_W1
print("count of unique Household Size responses: ", len(survey_W1["Household_Size_W1"].unique()))


# Function to calculate the number of people in each household
def calculate_household_size(entry):
    if entry == 'None of the above':
        return 1
    else:
        # Split the entry by semicolon and remove 'Cat or Dog' and 'Non-household members'
        categories = [category.strip() for category in entry.split(';') if category.strip() not in ['Cat or Dog', 'Non-household members (e.g. relatives, roommates)']]
        return len(categories)


# Apply the function to the specified column
survey_W1['Household Num'] = survey_W1["Household_Size_W1"].apply(calculate_household_size)

# Display the result for both columns
print(survey_W1[["Household_Size_W1", 'Household Num']])

# Display the count of unique values in Household_Size_W1
print("count of unique Household Size responses: ", len(survey_W1["Household Num"].unique()))
