import pandas as pd
import numpy as np
!pip install pandas
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Input data 
df = pd.read_csv ('/content/wafer_23012020_041211.csv')
df.head() # To see top 5 importat columns and raws 
df.info() # To get all data type of information 
# Percentage of missing value
def overall_missing_percentage(dk):
    total_missing = dk.isnull().sum().sum()
    total_cells = dk.size
    overall_missing_percentage = (total_missing / total_cells) * 100
    return overall_missing_percentage
# percentage of missing values in our data
percen_missing_value = overall_missing_percentage (df)
print(f"Overall percentage of missing values: {percen_missing_value:.2f}%")
# To check the balance of Target variable
class_distribution= df["Good/Bad"].value_counts()
class_proportion= df["Good/Bad"].value_counts(normalize=True)*100
print(class_proportion)
# We have to substract categorical values because we have to describe
m_df = df.iloc[:,1:591]
# Descriptive Statistics for all columns:
m_df.describe()
# We have created a def function for cleaning our whole data, where It will help to substract the columns whos column has zero standerd deviation and percentage of the missing value per column greater than 30%.
def clean_data (dl):
    zero_std_column = [col for col in dl.columns if dl[col].std() == 0] # Columns which have zero Std.
    missing_percentage = (dl.isnull().sum() / len(dl)) * 100 # Calculate percentage missing values
    high_missing_columns = [col for col in dl.columns if missing_percentage[col] > 30]
    columns_to_drop = set(zero_std_column + high_missing_columns) # combine both columns
    clean_data = dl.drop(columns=columns_to_drop)
    return clean_data
mf_data = clean_data(m_df)
# Check the missing values 
mf_data.isnull().sum().sum()
# Find Outliers
Q1 = mf_data.quantile(0.25)
Q3 = mf_data.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((mf_data < (Q1 - 1.5 * IQR)) | (mf_data > (Q3 + 1.5 * IQR))).any(axis=1)
num_outliers = outlier_mask.sum()
print(f"Number of outliers using IQR: {num_outliers}")
# We have to fill the missing values.So, We have used KNN impoter techniques.
from scipy import stats
from sklearn.impute import KNNImputer
def fill_missing_values(du): # We have made a def function for continuous variable and our main purpose is to fill the missing values by KNNI inputer
  inputer= KNNImputer(n_neighbors=2)
  du_input = inputer.fit_transform(du)
  du = pd.DataFrame(du_input, columns=du.columns)
  return(du)
clean_1 = fill_missing_values (mf_data)
print (clean_1)
# Now check again there is any missing values not 
clean_1.isnull().sum().sum()
# There is any duplicate value have or not 
clean_1.duplicated().sum()
# After cleaning the whole data, We have to standardize the data by using standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(clean_1), columns=clean_1.columns)
df_standardized.head()
# We have to include target variable with main data part
df_standardized ["Good/Bad"]=df["Good/Bad"]
df = df.rename (columns={'Unnamed: 0': 'Wafer_name'})
df_standardized ["Wafer_name"]=df["Wafer_name"]
final_df =df_standardized.copy()
# Slicing our data into two parts, one train data another is test data.
from sklearn.model_selection import train_test_split
final_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)
# Afetr cleaning the data as we have nomalized our data. Now, we can plot normal distribution curve to visual the distributions.
import matplotlib.pyplot as plt
import seaborn as sns
numeric_columns = final_df.columns.tolist() # to get exess all the numeric columns
random_columns = random.sample(numeric_columns, min(50, len(numeric_columns)))
n_cols = 5  # Number of plots per row
n_rows = -(-len(random_columns) // n_cols)  # Calculate rows needed

plt.figure(figsize=(20, n_rows * 4))  # Adjust figure size

# Iterate over the selected columns and plot
for i, column in enumerate(random_columns):
    plt.subplot(n_rows, n_cols, i + 1)  # Create subplot (But I didn't get this line of code)
    sns.histplot(final_df[column].dropna(), kde=True, bins=30, color='blue')
    plt.title(f'{column}', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()