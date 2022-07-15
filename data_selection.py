### =========================== DATA SELECTION ==================================== ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree


pd.pandas.set_option('display.max_columns', None)

raw_data = pd.read_csv('VAW.csv')


# drop unimportant features
print("Drop unimportant columns from dataset: 'DATAFLOW', 'FREQ: Frequency', 'INDICATOR: Indicator', 'SEX: Sex', 'AGE: Age',"
      "'UNIT_MEASURE: Unit of measure', 'UNIT_MULT: Unit multiplier', 'OBS_STATUS: Observation Status' ")

raw_data_dropped = raw_data.drop(['DATAFLOW', 'FREQ: Frequency', 'INDICATOR: Indicator', 'SEX: Sex', 'AGE: Age',
                                  'UNIT_MEASURE: Unit of measure', 'UNIT_MULT: Unit multiplier', 'OBS_STATUS: Observation Status'], axis=1)


# return a summary of the raw_data_dropped dataFrame
#print("\nReturn a summary of the raw_data_dropped dataFrame: \n")
#print(raw_data_dropped.info())

#print('\n')

# print the number of NaN values per column
#print("The number of NaN values per column is as follows: \n")
#print(raw_data_dropped.isna().sum())

#print('\n')

# print the unique values from each column in the dataframe
print("Return the unique values from each column in the dataset: \n")
for i in raw_data_dropped:
    print("Column: {}\n---------------------------------".format(i))
    print(raw_data_dropped[i].value_counts(), '\n')

print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')
# use the comments to fill empty values in "CONDITION: Women’s condition" column 
print("""Use the comments to fill empty values from "CONDITION: Women’s condition" column: \n""")

total_ever_part = raw_data_dropped[raw_data_dropped["CONDITION: Women’s condition"] == 'EVPART: Ever-partnered']["CONDITION: Women’s condition"]
print("""Total 'EVPART: Ever-partnered' values from "CONDITION: Women’s condition" column (before): """, total_ever_part.count(), '\n')


perpetrator = ['PARTNER: Partner']

ever_part = {'_T: Any': 'EVPART: Ever-partnered'}
for i in perpetrator:
    raw_data_dropped.loc[raw_data_dropped["PERPETRATOR: Perpetrator"] == i, "CONDITION: Women’s condition"] = raw_data_dropped.loc[raw_data_dropped["PERPETRATOR: Perpetrator"] == i,
                                                                                                                          "CONDITION: Women’s condition"].replace(ever_part)

total_ever_part_after = raw_data_dropped[raw_data_dropped["CONDITION: Women’s condition"] == 'EVPART: Ever-partnered']["CONDITION: Women’s condition"]
print("""Total 'EVPART: Ever-partnered' values from "CONDITION: Women’s condition" column (after): """, total_ever_part_after.count(), '\n')


print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

# use the comments to fill empty values in "CONDITION: Women’s condition"
print("""Use the comments to fill empty values in "CONDITION: Women’s condition" column: \n""")

total_ever_partnered = raw_data_dropped[raw_data_dropped["CONDITION: Women’s condition"] == 'EVPART: Ever-partnered']["CONDITION: Women’s condition"]
print("""Total "EVPART: Ever-partnered" values from CONDITION: Women’s condition" column (before use of comments): """, total_ever_partnered.count(), '\n')


comments_age = ['Ever married women 15-49.', 'Ever married women 15-49. Based on 25-49 cases.', 'Ever married women 15-49. Violence experienced committed by a husband or anyone else.',
                'Ever married women 15-49. Violence experienced from any perpetrators.']

ever_part = {'_T: Any': 'EVPART: Ever-partnered'}
for i in comments_age:
    raw_data_dropped.loc[raw_data_dropped["OBS_COMMENT: Comment"] == i, "CONDITION: Women’s condition"] = raw_data_dropped.loc[raw_data_dropped["OBS_COMMENT: Comment"] == i,
                                                                                                                          "CONDITION: Women’s condition"].replace(ever_part)

total_ever_partnered_after = raw_data_dropped[raw_data_dropped["CONDITION: Women’s condition"] == 'EVPART: Ever-partnered']["CONDITION: Women’s condition"]
print("""Total 'EVPART: Ever-partnered' values from "CONDITION: Women’s condition" column (after use of comments): """, total_ever_partnered_after.count(), '\n')


print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

# use the comments to fill empty values in "ACTUALITY: Actuality"
print("""Use the comments to fill empty values in "ACTUALITY: Actuality" column: \n""")

total_last_months = raw_data_dropped[raw_data_dropped["ACTUALITY: Actuality"] == 'ALO12M: At least once in the past 12 months']["ACTUALITY: Actuality"]
print("""Total "ALO12M: At least once in the past 12 months" values from "ACTUALITY: Actuality" column (before use of comments): """, total_last_months.count(), '\n')


comments_age = ['UN Women: Global database on Violence against women. In the past four weeks.']

last_months = {'_T: Any': 'ALO12M: At least once in the past 12 months'}
for i in comments_age:
    raw_data_dropped.loc[raw_data_dropped["OBS_COMMENT: Comment"] == i, "ACTUALITY: Actuality"] = raw_data_dropped.loc[raw_data_dropped["OBS_COMMENT: Comment"] == i,
                                                                                                                          "ACTUALITY: Actuality"].replace(last_months)

total_last_months_after = raw_data_dropped[raw_data_dropped["ACTUALITY: Actuality"] == 'ALO12M: At least once in the past 12 months']["ACTUALITY: Actuality"]
print("""Total 'ALO12M: At least once in the past 12 months' values from "ACTUALITY: Actuality" column (after use of comments): """, total_last_months_after.count(), '\n')


print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

### Drop more unimportant features
print("Now drop also 'OBS_COMMENT: Comment' and 'DATA_SOURCE: Data source' columns from 'raw_data_dropped' dataframe\n")
raw_data_dropped.drop(['OBS_COMMENT: Comment', 'DATA_SOURCE: Data source'], inplace=True, axis=1)

print('Now drop missing values from outcome feature and append it to new dataframe called x_test\n')
x_unknown = raw_data_dropped[raw_data_dropped["OUTCOME: Outcome"] == '_T: Any']
print("Length of x_test: ", len(x_unknown))
final_df = raw_data_dropped.drop(raw_data_dropped.loc[raw_data_dropped["OUTCOME: Outcome"] == '_T: Any'].index)
print("Length of final_df: ", len(final_df))

print('\n----------------------------------------------------------------------------------------------------------------------------------------------------\n')

print('Save results in excel files\n')
# raw_data_dropped.to_excel('all_data_col_dropped.xlsx', index=False)
# x_test.to_excel('x_test.xlsx', index=False)
# final_df.to_excel('final_df.xlsx', index=False)

print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

# return a summary of the final_df dataFrame
print("\nReturn a new summary of the final_df dataFrame: \n")
print(final_df.info())

print('\n')

# print the number of NaN values per column
print("Now, the number of NaN values per column is as follows: \n")
print(final_df.isna().sum())

print('\n')

# print the unique values from each column in the dataframe
print("Now, return again the unique values from each column in the dataset: \n")
for i in final_df:
    print("Column: {}\n---------------------------------".format(i))
    print(final_df[i].value_counts(), '\n')

print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

print("Replace '_T: Any' values in every feature with np.nan value and print isna().sum() of final_df: \n")
final_df.replace('_T: Any', np.nan, inplace=True)
x_unknown.replace('_T: Any', np.nan, inplace=True)

print(final_df.isna().sum())

print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

print('Save results in excel files of final_df dataFrame, after 3 empty columns dropping. \n')
final_df.to_csv('final_df_with_nan.csv', index=False)

print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

print('Plot a simple heatmap to visualize missing data\n')
# plt.figure(figsize=(8, 8))
mis_val_heatmap = sns.heatmap(final_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values in final dataset', fontsize=18)
plt.xlabel('Feautes', fontsize=12)
plt.xticks(rotation=30, wrap=True, fontsize=8)
# mis_val_heatmap.set_xticklabels(final_df.columns, rotation=55, ha="center")
# plt.tight_layout()
plt.show()


print("Now drop also 'RESPONSE: Response', 'HELP_REASON: Reason for searching help', 'HELP_PROVIDER: Help provider' columns from 'raw_data_dropped' dataframe because all of their values are null.\n")
final_df.drop(['RESPONSE: Response', 'HELP_REASON: Reason for searching help', 'HELP_PROVIDER: Help provider'], inplace=True, axis=1)
x_unknown.drop(['RESPONSE: Response', 'HELP_REASON: Reason for searching help', 'HELP_PROVIDER: Help provider'], inplace=True, axis=1)


print('Plot a simple heatmap to visualize missing data after removing empty columns\n')

mis_val_heatmap_new = sns.heatmap(final_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values in final dataset', fontsize=18)
plt.xlabel('Feautes', fontsize=12)
plt.xticks(rotation=30, wrap=True, fontsize=8)
plt.show()


# sns.set_style('whitegrid')
# final_df['VIOLENCE_TYPE: Type of violence'] = final_df['VIOLENCE_TYPE: Type of violence'].fillna('NaN Values')

# Plot Bar chart for Violence-Against-Women dataset for every feature 


for i in final_df.columns:
    if i == 'OBS_VALUE':
        continue
    plt.figure(figsize=(10, 8))
    ax = sns.countplot(x=final_df[i].fillna('NaN Values'), order=final_df[i].fillna('NaN Values').value_counts(ascending=False).index, palette='viridis')
    abs_values = final_df[i].fillna('NaN Values').value_counts(ascending=False).values
    ax.bar_label(container=ax.containers[0], labels=abs_values)
    plt.title("Bar chart for Violence-Against-Women dataset")

    plt.xticks(fontsize=15, rotation=65)
    plt.tight_layout()

    plt.show()

print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

print("Return a final summary of the final_df dataFrame: \n")
print(final_df.info())

# print(final_df.isna().sum())
final_df.to_csv('final_dataset.csv', index=False)
print(final_df.columns)
print(x_unknown.columns)
