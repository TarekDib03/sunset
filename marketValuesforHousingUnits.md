### Reading Data and Importing Modules


```python
# Change working directory
import os
os.chdir("D:\Rice_Uni_Business_Analytics_Capstone\Data")
```


```python
# Data files with extension .txt
txt_files = [item for item in os.listdir() if item.endswith(".txt")]
txt_files
```




    ['thads2001.txt',
     'thads2003.txt',
     'thads2005.txt',
     'thads2007.txt',
     'thads2009.txt',
     'thads2011.txt',
     'thads2013.txt']




```python
# Columns slected to be used for analysis
usecols = ["CONTROL", "AGE1", "METRO3", "REGION", "LMED", "FMR", "IPOV", "BEDRMS", "BUILT", "STATUS", 
           "TYPE", "VALUE", "NUNITS","ROOMS", "PER", "ZINC2", "ZADEQ", "ZSMHC", "STRUCTURETYPE", 
           "OWNRENT", "UTILITY", "OTHERCOST", "COST06", "COST08","COST12", "COSTMED", "ASSISTED"]
```


```python
# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from scipy import stats
import warnings
```


```python
data_2001 = pd.read_csv("thads2001.txt", sep=",", usecols=usecols)
data_2003 = pd.read_csv("thads2003.txt", sep=",", usecols=usecols)
data_2005 = pd.read_csv("thads2005.txt", sep=",", usecols=usecols)
data_2007 = pd.read_csv("thads2007.txt", sep=",", usecols=usecols)
data_2009 = pd.read_csv("thads2009.txt", sep=",", usecols=usecols)
data_2011 = pd.read_csv("thads2011.txt", sep=",", usecols=usecols)
data_2013 = pd.read_csv("thads2013.txt", sep=",", usecols=usecols)
```

### Data Manipulation and Statistical Analysis


```python
# Initialize a list
size_list = []
# Iterate over the length of txt_files to append the size of each file to the size_list. Select 
# only market values that are $1000 or more
for i in range(len(txt_files)):
    df = pd.read_csv(txt_files[i], sep=",", usecols=usecols)
    df = df[df["VALUE"] >= 1000]
    size_list.append(df.shape[0])
```


```python
# Print the size of each file as formated below
years = range(2001, 2015, 2)
for i in range(len(txt_files)):
    print("The number of housing units that have market values of $1,000 or more in {} is {}"
          .format(years[i], size_list[i]))
```

    The number of housing units that have market values of $1,000 or more in 2001 is 29381
    The number of housing units that have market values of $1,000 or more in 2003 is 33434
    The number of housing units that have market values of $1,000 or more in 2005 is 30514
    The number of housing units that have market values of $1,000 or more in 2007 is 27785
    The number of housing units that have market values of $1,000 or more in 2009 is 31317
    The number of housing units that have market values of $1,000 or more in 2011 is 85050
    The number of housing units that have market values of $1,000 or more in 2013 is 36675
    


```python
# Extract the names of the dataframes
def get_df_name(df):
    '''Extract the name of the data frame. Input of the function is the dataframe df'''
    name =[x for x in globals() if globals()[x] is df][0]
    return name
```


```python
# Change names of the `VALUE` and `STATUS` variables to match the year
def column_name_change(df):
    '''Change the names of the VALUE and STATUS columns. Input is a dataframe'''
    df.rename(columns={"VALUE": "VALUE_{}".format(get_df_name(df)[-4:]), 
                       "STATUS": "STATUS_{}".format(get_df_name(df)[-4:]), 
                       "FMR": "FMR_{}".format(get_df_name(df)[-4:])}, inplace = True)
```


```python
# Test
get_df_name(data_2001)
```




    'data_2001'




```python
dfs = [data_2005, data_2007, data_2009, data_2011, data_2013]
for df_name in dfs:
    column_name_change(df_name)
```


```python
# Now what is the average market value (in $) across all housing units for year 2005
data_2005[data_2005["VALUE_2005"] >= 1000]["VALUE_2005"].mean()
```




    246504.11244019138




```python
# Subsets of the dataframes - market value is $1000 or more
def f(df, x):
    return df[df[x] >= 1000]
```


```python
# A list of the yearly market value variables
yearly_value = ["VALUE_2005", "VALUE_2007", "VALUE_2009", "VALUE_2011", "VALUE_2013"]
# A list of yearly status
yearly_status = ["STATUS_2005", "STATUS_2007", "STATUS_2009", "STATUS_2011", "STATUS_2013"]
# A list of the dataframes of the years 2005 through 2013
dfs = [data_2005, data_2007, data_2009, data_2011, data_2013]

# Initialize the status list
status = []
# Iterate over the yearly data and group it by status to count the number of occupied vs. vacant apartments in these housing 
# units
for i in range(len(yearly_value)):
    status.append(f(dfs[i], yearly_value[i]).groupby([yearly_status[i]])[yearly_status[i]].count())

# Convert the list to a datafram
status_df = pd.DataFrame(status)    # '1': occupied, '3': not occupied
# Rename columns to Occupied and Vacant, respectively
status_df.rename(columns={"'1'": 'Occupied', "'3'": 'Vacant'}, inplace=True)
```


```python
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(16,4))
fig.suptitle('Occupied and Vacant Houses')
axes[0].set_title('Occupied Houses')
axes[1].set_title('Vacant Houses')
sns.barplot(ax = axes[0], x = status_df.index, y = status_df.Occupied)
sns.barplot(ax = axes[1], x = status_df.index, y = status_df.Vacant)
```




    <Axes: title={'center': 'Vacant Houses'}, ylabel='Vacant'>




    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_16_1.png)
    



```python
# Comparison of the average market value between occupied and vacant housing units 
df_2007_sub = f(data_2007[["STATUS_2007", "VALUE_2007"]], "VALUE_2007")
df_2007_sub.groupby("STATUS_2007").mean().round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VALUE_2007</th>
    </tr>
    <tr>
      <th>STATUS_2007</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'1'</th>
      <td>278960.75</td>
    </tr>
    <tr>
      <th>'3'</th>
      <td>289004.49</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Standard deviation
df_2007_sub.groupby("STATUS_2007").std().round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VALUE_2007</th>
    </tr>
    <tr>
      <th>STATUS_2007</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'1'</th>
      <td>317162.77</td>
    </tr>
    <tr>
      <th>'3'</th>
      <td>306203.82</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now mean and standard deviation for 2013
df_2013_sub = f(data_2013[["STATUS_2013", "VALUE_2013"]], "VALUE_2013")
print(df_2013_sub.groupby("STATUS_2013").mean().round(2))
print(df_2013_sub.groupby("STATUS_2013").std().round(2))
```

                 VALUE_2013
    STATUS_2013            
    '1'           249858.55
    '3'           251996.82
                 VALUE_2013
    STATUS_2013            
    '1'           282290.65
    '3'           389653.09
    


```python
# Report the t-statistic from the difference in means test using the 2011 data

# Two-sample independent t-test

# 1. Import stats library from scipy
from scipy import stats

# 2. Prepare data
occupied_2011 = f(data_2011, "VALUE_2011")[f(data_2011, "VALUE_2011")["STATUS_2011"] == "'1'"]["VALUE_2011"]
vacant_2011 = f(data_2011, "VALUE_2011")[f(data_2011, "VALUE_2011")["STATUS_2011"] == "'3'"]["VALUE_2011"]

# 3. Calculate the t-statistic and p-value
t_stat, p_value = stats.ttest_ind(occupied_2011, vacant_2011)
print(round(t_stat, 4), round(p_value, 4))
```

    6.397 0.0
    


```python
# Now the t-statistic from the difference in means test using the 2013 data
occupied_2013 = f(data_2013, "VALUE_2013")[f(data_2013, "VALUE_2013")["STATUS_2013"] == "'1'"]["VALUE_2013"]
vacant_2013 = f(data_2013, "VALUE_2013")[f(data_2013, "VALUE_2013")["STATUS_2013"] == "'3'"]["VALUE_2013"]

# 3. Calculate the t-statistic and p-value
t_stat_2013, p_value_2013 = stats.ttest_ind(occupied_2013, vacant_2013)
print(round(t_stat_2013, 4), round(p_value_2013, 4))
```

    -0.2599 0.7949
    


```python
def diff_in_means_test(df, x, y):
    '''Difference in means test for current market vlaues of occupied vs. vacant housing units.
    Inputs: dataframe df, x: column name (string), y: column name (string) 
    '''
    occupied = f(df, x)[f(df, x)[y] == "'1'"][x]
    vacant = f(df, x)[f(df, x)[y] == "'3'"][x]
    t_stat, p_value = stats.ttest_ind(occupied, vacant)
    return round(t_stat, 4), round(p_value, 4)
```


```python
# Years
years_2 = range(2005, 2015, 2)
# Empty dictionary for the t statistics values
t_stats = {}
# Empty dictionary for the p-values
p_values = {} 
# Iterate over the length of the names of dataframes within the list dfs to populate the above dictionaries
for i in range(len(dfs)):
    t_stats[years_2[i]] = diff_in_means_test(dfs[i], yearly_value[i], yearly_status[i])[0]
    p_values[years_2[i]] = diff_in_means_test(dfs[i], yearly_value[i], yearly_status[i])[1]
```


```python
pd.melt(pd.DataFrame([t_stat])).rename(columns={"variable": "Year", "value": "t_stat"})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>t_stat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6.397</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.melt(pd.DataFrame([p_values])).rename(columns={"variable": "Year", "value": "p_value"})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>0.0416</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007</td>
      <td>0.2609</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009</td>
      <td>0.8465</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013</td>
      <td>0.7949</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Similar to the above, but the test is one sided t-test
def one_sided_ttest(df, x, y):
    occupied = f(df, x)[f(df, x)[y] == "'1'"][x]
    vacant = f(df, x)[f(df, x)[y] == "'3'"][x]
    t_stat, p_value = stats.ttest_ind(occupied, vacant, alternative = 'less')
    return round(t_stat, 4), round(p_value, 4)
```


```python
t_stats_one = {}
p_values_one = {} 

for i in range(len(dfs)):
    p_values_one[years_2[i]] = one_sided_ttest(dfs[i], yearly_value[i], yearly_status[i])[1]
```


```python
pd.melt(pd.DataFrame([p_values_one])).rename(columns={"variable": "Year", "value": "p_value"})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>0.9792</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007</td>
      <td>0.1305</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009</td>
      <td>0.4232</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013</td>
      <td>0.3975</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge all the data frames (2005 through 2013)
yearly_fmr = data_2005[["CONTROL", "FMR_2005"]].merge(
    data_2007[["CONTROL", "FMR_2007"]], on="CONTROL").merge(
    data_2009[["CONTROL", "FMR_2009"]], on="CONTROL").merge(
    data_2011[["CONTROL", "FMR_2011"]], on="CONTROL").merge(
    data_2013[["CONTROL", "FMR_2013"]], on="CONTROL")
```


```python
yearly_fmr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTROL</th>
      <th>FMR_2005</th>
      <th>FMR_2007</th>
      <th>FMR_2009</th>
      <th>FMR_2011</th>
      <th>FMR_2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>'100007130148'</td>
      <td>519</td>
      <td>616</td>
      <td>685</td>
      <td>711</td>
      <td>737</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'100007390148'</td>
      <td>600</td>
      <td>605</td>
      <td>670</td>
      <td>673</td>
      <td>657</td>
    </tr>
    <tr>
      <th>2</th>
      <td>'100007540148'</td>
      <td>788</td>
      <td>807</td>
      <td>897</td>
      <td>935</td>
      <td>988</td>
    </tr>
    <tr>
      <th>3</th>
      <td>'100008700141'</td>
      <td>702</td>
      <td>778</td>
      <td>743</td>
      <td>796</td>
      <td>773</td>
    </tr>
    <tr>
      <th>4</th>
      <td>'100009170148'</td>
      <td>546</td>
      <td>599</td>
      <td>503</td>
      <td>531</td>
      <td>552</td>
    </tr>
  </tbody>
</table>
</div>




```python
fmr_columns = []
warnings.filterwarnings("ignore")
for column in list(yearly_fmr.columns):
    if column.startswith("FMR"):
        fmr_columns.append(column)

yearly_fmr_sub = yearly_fmr

for fmr in fmr_columns:
    yearly_fmr_sub = yearly_fmr_sub[yearly_fmr[fmr] > 0]
yearly_fmr_sub.shape
```




    (26373, 6)




```python
# Test
fmr_min = []
for fmr in fmr_columns:
    fmr_min.append(yearly_fmr_sub[fmr].min())
fmr_min
```




    [360, 387, 427, 424, 421]




```python
fmr_yearly_avgs = []
warnings.filterwarnings("ignore")
for i in range(5):
    fmr_yearly_avgs.append(round(yearly_fmr_sub.describe().loc['mean'][i],2))
fmr_yearly_avgs
```




    [929.04, 977.77, 1063.87, 1116.38, 1151.57]




```python
# Paired ttest
def paired_ttest(x, y):
    t_stat, p_value = stats.ttest_rel(yearly_fmr_sub[x], yearly_fmr_sub[y])
    return round(t_stat, 4), round(p_value, 4)
```


```python
t_stats_paired = []
p_values_paired = []
fmrs = ["FMR_2005", "FMR_2007", "FMR_2009", "FMR_2011", "FMR_2013"]

for i in range(1, len(fmrs)):
    t_stats_paired.append(paired_ttest(fmrs[i], fmrs[i-1])[0])

t_stats_paired
```




    [69.4904, 124.3222, 74.1512, 58.2109]




```python
# OR
for i in range(1, len(fmrs)):
    print("The t-statistic for the difference in means test for the years {} and {} is {}".
          format(fmrs[i-1][-4:], fmrs[i][-4:], paired_ttest(fmrs[i], fmrs[i-1])[0]))
```

    The t-statistic for the difference in means test for the years 2005 and 2007 is 69.4904
    The t-statistic for the difference in means test for the years 2007 and 2009 is 124.3222
    The t-statistic for the difference in means test for the years 2009 and 2011 is 74.1512
    The t-statistic for the difference in means test for the years 2011 and 2013 is 58.2109
    


```python
def percent_increase_fmr(x, y):
    return((yearly_fmr_sub[y] - yearly_fmr_sub[x]) / yearly_fmr_sub[x] * 100)
```


```python
for i in range(1, len(fmrs)):
    print("The average percent increase in fair market rent between {} and {} is {}%".
          format(fmrs[i-1][-4:], fmrs[i][-4:], 
                 round(percent_increase_fmr(fmrs[i-1], fmrs[i]).mean(),2)))
```

    The average percent increase in fair market rent between 2005 and 2007 is 6.21%
    The average percent increase in fair market rent between 2007 and 2009 is 9.28%
    The average percent increase in fair market rent between 2009 and 2011 is 5.09%
    The average percent increase in fair market rent between 2011 and 2013 is 3.77%
    


```python
# lets draw a line chart for the fair market values for the years 2005 through 2013
years = [2005, 2007, 2009, 2011, 2013]
fmr_yearly_data = pd.DataFrame({"year": years, "avg_yearly_fmr": fmr_yearly_avgs})
fmr_yearly_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>avg_yearly_fmr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>929.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007</td>
      <td>977.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009</td>
      <td>1063.87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>1116.38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013</td>
      <td>1151.57</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(6, 3))
warnings.filterwarnings("ignore")
sns.lineplot(data = fmr_yearly_data, x = "year", y = "avg_yearly_fmr", ax=ax)
ax.set_xlim(2003,2015)
ax.set_xticks(range(2005,2015,2))
plt.title("Yearly Average Fair Market Rent")
plt.xlabel("Year")
plt.ylabel("Average Fair Market Rent");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_40_0.png)
    



```python
fvalue, pvalue = stats.f_oneway(yearly_fmr_sub["FMR_2005"], yearly_fmr_sub["FMR_2007"], 
                               yearly_fmr_sub["FMR_2009"], yearly_fmr_sub["FMR_2011"],
                               yearly_fmr_sub["FMR_2013"])
print(fvalue, pvalue)
```

    1706.7912624191479 0.0
    

## Data from 2013 for Linear Regression Analysis - Owned Single Family House Units

#### Define the Variables used in Remaining Analysis for the Year 2013

- CONTROL	Character: A control variable for Housing Unit. Useful to match data across datasets from different years.
- AGE1: Age of head of household
- METRO3 - a character variable with options ‘1’,’2’, ‘3’, ‘4’ or ‘5’ where ‘1’ : Central City, ‘2’, ‘3’, ‘4’, ‘5’ :Others
- REGION	Character ‘1’,’2’, ‘3’ or ‘4’. The four census regions—Northeast, Midwest, South, and West. 
- LMED (Numerical, \$\): Area Median Income
- FMR (Numerical, \$\): Fair Market Monthly Rent
- IPOV (Numerical, \$\): Poverty Income threshold
- BEDRMS (Numerical): Number of Bedrooms in the unit
- BUILT (Numerical): Year the unit was built
- STATUS (Character) ‘1’, ‘3’ 	Occupied or Vacant
- TYPE (Numerical): Structure Type
    - 1 	House, apartment, flat 
    - 2 	Mobile home with no permanent room added 
    - 3 	Mobile home with permanent room added
    - 4 	HU, in nontransient hotel, motel, etc
    - 5 	HU, in permanent transient hotel, motel, etc
    - 6 	HU, in rooming house
    - 7 	Boat or recreation vehicle
    - 9 	HU, not specified above
- VALUE	(Numerical, \$\)): Current market value of unit
- NUNITS (Numerical): Number of Units in Building
- ROOMS	(Numerical): Number of rooms in the unit
- PER (Numerical): Number of persons in Household
- ZINC2 (Numerical): Annual Household income
- ZADEQ	(Character): Adequacy of unit - ‘1’: Adequate, '2’: Moderately Inadequate, '3’: Severely Inadequate, ‘-6’: Vacant - No information
- ZSMHC	(Numerical, \$\): Monthly housing costs. For renters, housing cost is contract rent plus utility costs. For Owners, mortgage is not included
- STRUCTURETYPE (Numerical)	Structure Type
    - 1 	Single Family
    - 2 	2-4 units
    - 3 	5-19 units
    - 4 	20-49 units
    - 5 	50+ units
    - 6 	Mobile Home
    - -9 	Not Known
- OWNRENT (Character), ‘1’,  ‘2’	
    - ‘1’:	Owner: Owner occupied, vacant for sale, and sold but not occupied.
    - ‘2’: 	Rental: Occupied units rented for cash and without payment of cash rent. Vacant for rent, vacant for rent or sale, and rented but not occupied.
- UTILITY (Numerical \$)	Monthly utilities cost (gas, oil, electricity, other fuel, trash collection, and water)
- OTHERCOST	(Numerical \$)	Sum of ‘other monthly costs’ such as Home owners’ or renters’ insurance, Land rent (where distinct from unit rent), Condominium fees (where applicable), Other mobile home fees (where applicable).
- COST06 (Numerical \$): Monthly mortgage payments assuming 6% interest. This applies only to “Owners”.
- COST08 (Numerical \$): Monthly mortgage payments assuming 8% interest. This applies only to “Owners”.
- COST12 (Numerical \$): Monthly mortgage payments assuming 12% interest. This applies only to “Owners”.
- COSTMED	(Numerical \$): Monthly mortgage payments assuming median interest. This applies only to “Owners”.
- ASSISTED (Numerical)	Did the housing unit receive some governmental ‘assistance”? 
    - 0 	Not assisted
    - 1 	Assisted
    - -9 	Not Known

In the proceeding analysis, only data from 2013 will be used. Owned single family housing units will be considered for the rest of the analysis. The data will be used to build a multiple linear regression model that would predict the current market values of housing units based on a set of explanatory features. 


```python
# Read the 2013 into a different dataframe than the one used before
data_2013_df = pd.read_csv("thads2013.txt", sep=",", usecols=usecols)
data_2013_df.shape
```




    (64535, 27)



#### Data Processing and Manipulation


```python
def owned_single_family_housing(df, x = "VALUE", value=1000, type=1, structure_type=1, owned="'1'"):
    '''The function inputs a dataframe, a variable name x to condition over, with default values
    of value of $1000, type=1 (house), structure_type=1 (single family), and owned house'''
    return df[(df[x] >= value) & 
              (df["TYPE"] == type) & 
              (df["STRUCTURETYPE"] == structure_type) &
              (df["OWNRENT"] == owned)].reset_index(drop=True)
owned_single_family_housing_df = owned_single_family_housing(data_2013_df)
```


```python
owned_single_family_housing_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32825 entries, 0 to 32824
    Data columns (total 27 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   CONTROL        32825 non-null  object 
     1   AGE1           32825 non-null  int64  
     2   METRO3         32825 non-null  object 
     3   REGION         32825 non-null  object 
     4   LMED           32825 non-null  int64  
     5   FMR            32825 non-null  int64  
     6   IPOV           32825 non-null  int64  
     7   BEDRMS         32825 non-null  int64  
     8   BUILT          32825 non-null  int64  
     9   STATUS         32825 non-null  object 
     10  TYPE           32825 non-null  int64  
     11  VALUE          32825 non-null  int64  
     12  NUNITS         32825 non-null  int64  
     13  ROOMS          32825 non-null  int64  
     14  PER            32825 non-null  int64  
     15  ZINC2          32825 non-null  int64  
     16  ZADEQ          32825 non-null  object 
     17  ZSMHC          32825 non-null  int64  
     18  STRUCTURETYPE  32825 non-null  int64  
     19  OWNRENT        32825 non-null  object 
     20  UTILITY        32825 non-null  float64
     21  OTHERCOST      32825 non-null  float64
     22  COST06         32825 non-null  float64
     23  COST12         32825 non-null  float64
     24  COST08         32825 non-null  float64
     25  COSTMED        32825 non-null  float64
     26  ASSISTED       32825 non-null  int64  
    dtypes: float64(6), int64(15), object(6)
    memory usage: 6.8+ MB
    


```python
owned_single_family_housing_df.drop(columns=["TYPE", "STRUCTURETYPE", "OWNRENT"], inplace=True)
```


```python
# In the METRO3 (metropolitan status) variable, replace '1' with Central City and '2', '3', '4', or '5'
# with Other
import numpy as np
owned_single_family_housing_df["METRO3"] = np.where(owned_single_family_housing_df.METRO3 == "'1'",
                                                   "central_city", "Other")
```


```python
# STATUS variable: '1': Occupied, '3': Vacant
owned_single_family_housing_df["STATUS"] = np.where(owned_single_family_housing_df.STATUS == "'1'",
                                                   "occupied", "vacant")
```


```python
owned_single_family_housing_df.replace({"REGION":{"'1'": "Northeast", "'2'": "Midwest", 
                                                 "'3'": "South", "'4'": "West"}}, inplace = True)
```


```python
owned_single_family_housing_df.replace({"ZADEQ":{"'1'": "Adequate", "'2'": "Moderately_inadequate",
                    "'3'": "Severely_inadequate", "'-6'": "Vacant_No_Info"}}, inplace = True)
```


```python
owned_single_family_housing_df.replace({"ASSISTED":{0:"Not_Assisted", 1:"Assisted", -9:"Not_known"}}, 
                                        inplace=True)
```


```python
owned_single_family_housing_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE1</th>
      <th>LMED</th>
      <th>FMR</th>
      <th>IPOV</th>
      <th>BEDRMS</th>
      <th>BUILT</th>
      <th>VALUE</th>
      <th>NUNITS</th>
      <th>ROOMS</th>
      <th>PER</th>
      <th>ZINC2</th>
      <th>ZSMHC</th>
      <th>UTILITY</th>
      <th>OTHERCOST</th>
      <th>COST06</th>
      <th>COST12</th>
      <th>COST08</th>
      <th>COSTMED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>3.282500e+04</td>
      <td>32825.0</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>3.282500e+04</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
      <td>32825.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.624158</td>
      <td>68158.426078</td>
      <td>1277.328835</td>
      <td>17184.712628</td>
      <td>3.251942</td>
      <td>1967.431104</td>
      <td>2.573874e+05</td>
      <td>1.0</td>
      <td>6.730967</td>
      <td>2.365362</td>
      <td>8.488523e+04</td>
      <td>1311.889992</td>
      <td>245.215319</td>
      <td>95.374486</td>
      <td>2051.174528</td>
      <td>3045.090902</td>
      <td>2362.079521</td>
      <td>1836.053536</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.027382</td>
      <td>12502.711452</td>
      <td>396.375112</td>
      <td>6657.658062</td>
      <td>0.848657</td>
      <td>26.742731</td>
      <td>2.826902e+05</td>
      <td>0.0</td>
      <td>1.649389</td>
      <td>2.070924</td>
      <td>8.565889e+04</td>
      <td>1087.194787</td>
      <td>123.231149</td>
      <td>104.399267</td>
      <td>1961.990997</td>
      <td>3051.403698</td>
      <td>2302.542808</td>
      <td>1726.568892</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-9.000000</td>
      <td>38500.000000</td>
      <td>421.000000</td>
      <td>-6.000000</td>
      <td>0.000000</td>
      <td>1919.000000</td>
      <td>1.000000e+04</td>
      <td>1.0</td>
      <td>2.000000</td>
      <td>-6.000000</td>
      <td>-1.170000e+02</td>
      <td>-6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>66.459547</td>
      <td>105.075134</td>
      <td>78.538812</td>
      <td>58.101678</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>43.000000</td>
      <td>60060.000000</td>
      <td>1004.000000</td>
      <td>13948.000000</td>
      <td>3.000000</td>
      <td>1950.000000</td>
      <td>1.100000e+05</td>
      <td>1.0</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>3.000000e+04</td>
      <td>550.000000</td>
      <td>165.000000</td>
      <td>41.666667</td>
      <td>1011.974114</td>
      <td>1428.901605</td>
      <td>1141.132406</td>
      <td>919.600112</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.000000</td>
      <td>64810.000000</td>
      <td>1204.000000</td>
      <td>15470.000000</td>
      <td>3.000000</td>
      <td>1970.000000</td>
      <td>1.800000e+05</td>
      <td>1.0</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>6.397400e+04</td>
      <td>1057.000000</td>
      <td>222.666667</td>
      <td>70.000000</td>
      <td>1546.190945</td>
      <td>2263.760874</td>
      <td>1770.776233</td>
      <td>1391.950224</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>66.000000</td>
      <td>74008.000000</td>
      <td>1445.000000</td>
      <td>23401.000000</td>
      <td>4.000000</td>
      <td>1990.000000</td>
      <td>3.000000e+05</td>
      <td>1.0</td>
      <td>8.000000</td>
      <td>4.000000</td>
      <td>1.092000e+05</td>
      <td>1740.000000</td>
      <td>299.000000</td>
      <td>108.333333</td>
      <td>2441.455512</td>
      <td>3624.254012</td>
      <td>2814.497683</td>
      <td>2187.058726</td>
    </tr>
    <tr>
      <th>max</th>
      <td>93.000000</td>
      <td>115300.000000</td>
      <td>3511.000000</td>
      <td>51635.000000</td>
      <td>7.000000</td>
      <td>2013.000000</td>
      <td>2.520000e+06</td>
      <td>1.0</td>
      <td>15.000000</td>
      <td>20.000000</td>
      <td>1.061921e+06</td>
      <td>10667.000000</td>
      <td>1249.000000</td>
      <td>2020.916667</td>
      <td>19261.472577</td>
      <td>28992.600365</td>
      <td>22305.447202</td>
      <td>17155.289494</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Numeric fields that each have values less than 0
x = list(owned_single_family_housing_df
                 .describe()
                 .min()[owned_single_family_housing_df.describe().min() <= 0]
                 .index)
print(x)
```

    ['AGE1', 'IPOV', 'BEDRMS', 'NUNITS', 'PER', 'ZINC2', 'ZSMHC', 'UTILITY', 'OTHERCOST']
    


```python
owned_single_family_housing_df = owned_single_family_housing_df[
    (owned_single_family_housing_df[x] > 0).all(axis=1)]
minimums = pd.DataFrame(owned_single_family_housing_df.describe().min())
min_field_vals = minimums.rename(columns={0: "Minimum"}).reset_index().rename(
    columns={"index": "Field"}).sort_values(by = "Minimum")
min_field_vals.reset_index(inplace=True, drop=True)
```

Now, all the numeric fields have positive values that are 0 or more as shown in the above summary statistics table. All of the numeric fields have a minimum value that is 0 or more.


```python
owned_single_family_housing_df.shape
```




    (30174, 24)



### Exploratory Data Analysis


```python
# Descriptive statistics for the outcome variable "VALUE"
value_descriptive_statistics = pd.DataFrame(owned_single_family_housing_df["VALUE"].describe().round(2)).reset_index()
value_descriptive_statistics.rename(columns={"index": "Statistics", "VALUE": "Value"})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Statistics</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>30174.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>262685.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>std</td>
      <td>281724.78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>10000.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25%</td>
      <td>120000.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50%</td>
      <td>190000.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>75%</td>
      <td>310000.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>max</td>
      <td>2520000.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
value_descriptive_statistics.to_csv("value.csv")
```


```python
# Summary statistics for categorical variables
owned_single_family_housing_df.describe(include='object')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CONTROL</th>
      <th>METRO3</th>
      <th>REGION</th>
      <th>STATUS</th>
      <th>ZADEQ</th>
      <th>ASSISTED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30174</td>
      <td>30174</td>
      <td>30174</td>
      <td>30174</td>
      <td>30174</td>
      <td>30174</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>30174</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>top</th>
      <td>'100003130103'</td>
      <td>Other</td>
      <td>Midwest</td>
      <td>occupied</td>
      <td>Adequate</td>
      <td>Not_known</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>23750</td>
      <td>9015</td>
      <td>30174</td>
      <td>29478</td>
      <td>30174</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, all of the remaining 30,174 housing units are occupied. Therefore, the variable `STATTUS` will not be included in the regression model, because it will not add any information to the final model. In addition, since about 98% (29,478 out of 30,174) of the housing units have adequate space in the response of the Variable `ZADEQ`, it will not be included in the linear regression model. Therefore the variables `STATUS` AND `ZADEQ` will be dropped from the above dataframe.  


```python
plt.figure(figsize=(7,3))
sns.boxplot(owned_single_family_housing_df, x='REGION', y='VALUE')
plt.ylabel("Current Market Value ($)")
plt.xlabel("Region")
plt.title("Distribution of Current Market Value Grouped by Region");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_65_0.png)
    



```python
# Average current market value in each region 
pd.DataFrame(owned_single_family_housing_df.groupby("REGION")["VALUE"].
             mean().round()).sort_values(by="VALUE").reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REGION</th>
      <th>VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Midwest</td>
      <td>186871.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>South</td>
      <td>216831.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Northeast</td>
      <td>328229.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>387586.0</td>
    </tr>
  </tbody>
</table>
</div>



All 4 regions have the same maximum market value of \\$2.52 million, and the same minimum of \\$10,000. However, the average market value ranges between \\$186,871 in the Midwest and \\$387,586 in the West. On average, housing units in the Midwest and South are much cheaper than those in the Northeast and the West. 


```python
# Use pivot table to count the number of housing units each year and estimate their market values
tbl = pd.pivot_table(owned_single_family_housing_df, values=["VALUE"],
            index=['BUILT'], aggfunc={"mean", "count"}).reset_index().round()
# Extract the mean and count from the above tbl datafram
units_stats = tbl["VALUE"]
# Rename columns
units_stats.rename(columns={"mean": "yearly_avg_value", "count": "number_of_units"}, inplace=True)
```


```python
# Concatenate the year to the mean of market values and the count of housing unitrs
units_df = pd.concat([pd.DataFrame(tbl["BUILT"]), units_stats], axis=1)
units_df.rename(columns={"mean": "yearly_avg_value", 
                            "count": "number_of_units", 
                            "BUILT": "year"}, inplace=True)
units_df.sort_values(by="number_of_units").reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>number_of_units</th>
      <th>yearly_avg_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013</td>
      <td>13</td>
      <td>608462.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>128</td>
      <td>362969.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>144</td>
      <td>355764.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>169</td>
      <td>303018.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009</td>
      <td>200</td>
      <td>297150.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2008</td>
      <td>292</td>
      <td>356370.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2002</td>
      <td>376</td>
      <td>302234.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2000</td>
      <td>430</td>
      <td>299395.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2007</td>
      <td>431</td>
      <td>300650.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2001</td>
      <td>443</td>
      <td>318623.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2003</td>
      <td>449</td>
      <td>304321.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2004</td>
      <td>479</td>
      <td>274551.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2006</td>
      <td>483</td>
      <td>308489.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2005</td>
      <td>491</td>
      <td>287515.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1920</td>
      <td>1121</td>
      <td>263390.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1930</td>
      <td>1181</td>
      <td>261431.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1990</td>
      <td>1252</td>
      <td>307388.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1980</td>
      <td>1393</td>
      <td>249311.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1940</td>
      <td>1783</td>
      <td>231032.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1919</td>
      <td>1796</td>
      <td>253207.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1985</td>
      <td>1846</td>
      <td>298754.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1970</td>
      <td>2072</td>
      <td>245594.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1975</td>
      <td>2657</td>
      <td>244212.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1995</td>
      <td>2861</td>
      <td>277407.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1960</td>
      <td>3627</td>
      <td>242677.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1950</td>
      <td>4057</td>
      <td>233823.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
units_df.to_csv("year.csv")
```

From the table above, the yearly average market value barely increases for housing units built between 1940 (\\$231,032) and 1980 (\\$249,311). This accounts for a change of only about an 8% increase in market value for housing units built between 1940 and 1980. For those built between 1919 and 1930, the yearly average market values do not seem to change significantly (~3% increase in market values between 1919 and 1930). Housing units built in 1940 have the lowest average market value of about \\$231,032. The most expensive are housing units built in 2013.

Since there are only 13 housing units built in 2013, and the average market value is almost double the average market value in 2011, housing units built in 2013 will be excluded because the number of units is very low in 2013, and may bias the results!


```python
units_df = units_df[units_df["year"] != 2013]
```


```python
plt.figure(figsize=(12,5))
sns.barplot(units_df, x='year', y='yearly_avg_value')
plt.title("Average Current Market Values of Units Grouped by Year the Units were built")
plt.ylabel("Avearge Current Market Value of Units");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_74_0.png)
    


Current average market values for housing units range between \\$231,032 for those built in 1940 and \\$362,969 built-in 2011, an increase of about 57% in market value in housing units built in 1940. The yearly average market values for housing units built during the global financial crisis (2007 - 2009) fluctuate significantly. For housing units built in 2008, the average market values sit at about \\$356,370. However, for housing units built in 2009, the average market values drop significantly to \\$297,150, i.e. a drop of about 17%. From the above barplot, the average market value is approximately uniformly distributed throughout the years from 1919 to 2012. The average market value per housing unit is estimated to be about \\$262,686. The Variable `BUILT` will not be included in the linear regression model. No time series analysis will be discussed in this report!  


```python
# Create the dummy variables for the categorical variables
region_d = pd.get_dummies(owned_single_family_housing_df['REGION'], dtype=int)
metro_d = pd.get_dummies(owned_single_family_housing_df['METRO3'], dtype=int)
adequacy_d = pd.get_dummies(owned_single_family_housing_df['ZADEQ'], dtype=int)
# Concatenate all the dummy variables into one dataframe
d_df = pd.concat([region_d, metro_d, adequacy_d], axis=1)
d_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Midwest</th>
      <th>Northeast</th>
      <th>South</th>
      <th>West</th>
      <th>Other</th>
      <th>central_city</th>
      <th>Adequate</th>
      <th>Moderately_inadequate</th>
      <th>Severely_inadequate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(5,5))
sns.scatterplot(data = owned_single_family_housing_df, 
                x = owned_single_family_housing_df.COST06, 
                y = owned_single_family_housing_df.VALUE)
sns.scatterplot(data = owned_single_family_housing_df, 
                x = owned_single_family_housing_df.COST08, 
                y = owned_single_family_housing_df.VALUE)
sns.scatterplot(data = owned_single_family_housing_df, 
                x = owned_single_family_housing_df.COST12, 
                y = owned_single_family_housing_df.VALUE)
sns.scatterplot(data = owned_single_family_housing_df, 
                x = owned_single_family_housing_df.COSTMED, 
                y = owned_single_family_housing_df.VALUE)
#ax.legend(loc='lower ',ncol=4, title="Title")
plt.legend(title='Monthly Mortgage Payments', loc='lower right', 
           labels=['6%', '8%', '12%', 'median rate'])
plt.xlabel("Monthly Mortgage Payment")
plt.ylabel("Housing Market Value")
plt.title("Current Market Values of Units vs. Monthly Mortgage Payment");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_77_0.png)
    


As expected, from the scatterplot above, the higher the monthly motgage payment, the higher the market values of the housing units. Though this is true, however, for the purpose of our study, monthly mortgage payments will not add any information to predict the current market value (price) of the housing units. Thus, these variables will not be included in the regression model. 


```python
plt.figure(figsize=(6,3))
sns.scatterplot(data=owned_single_family_housing_df, x="BEDRMS", y="ROOMS")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Number of Rooms")
plt.title("Number of Rooms vs. Number of Bedrooms");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_79_0.png)
    


Number of rooms and number of bedrooms are highly correlated as shown in the above scatterplot, and the high correlation between the 2 variables. The Variable `BEDRMS` will be dropped to avoid multicollinearity issue. 


```python
# Create a total_cost variable as follows:
owned_single_family_housing_df["total_cost"] = owned_single_family_housing_df[
    "UTILITY"] + owned_single_family_housing_df[
    "OTHERCOST"] + owned_single_family_housing_df["ZSMHC"]
owned_single_family_housing_df["ln_value"] = np.log(owned_single_family_housing_df.VALUE)
```


```python
X = owned_single_family_housing_df.drop(columns=["VALUE", "ln_value", "UTILITY", "OTHERCOST", "ZSMHC",
            "UTILITY", "ASSISTED", "CONTROL", "NUNITS", "COST06", "COST08", "COST12", "COSTMED"])
X = pd.concat([X, d_df[['Midwest', 'Northeast', 'South', 'central_city', 'Adequate']]], axis=1)
```


```python
# Drop the Variables REGION, ZADEQ, METRO3, STATUS
X.drop(columns=["REGION", "ZADEQ", "METRO3", "STATUS"], inplace=True)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE1</th>
      <th>LMED</th>
      <th>FMR</th>
      <th>IPOV</th>
      <th>BEDRMS</th>
      <th>BUILT</th>
      <th>ROOMS</th>
      <th>PER</th>
      <th>ZINC2</th>
      <th>total_cost</th>
      <th>Midwest</th>
      <th>Northeast</th>
      <th>South</th>
      <th>central_city</th>
      <th>Adequate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>82</td>
      <td>73738</td>
      <td>956</td>
      <td>11067</td>
      <td>2</td>
      <td>2006</td>
      <td>6</td>
      <td>1</td>
      <td>18021</td>
      <td>915.750000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>55846</td>
      <td>1100</td>
      <td>24218</td>
      <td>4</td>
      <td>1980</td>
      <td>6</td>
      <td>4</td>
      <td>122961</td>
      <td>790.666667</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>55846</td>
      <td>1100</td>
      <td>15470</td>
      <td>4</td>
      <td>1985</td>
      <td>7</td>
      <td>2</td>
      <td>27974</td>
      <td>1601.500000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>67</td>
      <td>55846</td>
      <td>949</td>
      <td>13964</td>
      <td>3</td>
      <td>1985</td>
      <td>6</td>
      <td>2</td>
      <td>32220</td>
      <td>528.666667</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>60991</td>
      <td>988</td>
      <td>18050</td>
      <td>3</td>
      <td>1985</td>
      <td>6</td>
      <td>3</td>
      <td>69962</td>
      <td>1476.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Correlation Matrix
corr = X.corr()
def highlight_max(s): 
    if s.dtype == 'object': 
        is_corr = [False for _ in range(s.shape[0])] 
    else: 
        is_corr = s > 0.5
    return ['color: red;' if cell else 'color:black' 
            for cell in is_corr]
corr.style.apply(highlight_max)
```




<style type="text/css">
#T_e7ff3_row0_col0, #T_e7ff3_row1_col1, #T_e7ff3_row1_col2, #T_e7ff3_row1_col11, #T_e7ff3_row2_col1, #T_e7ff3_row2_col2, #T_e7ff3_row3_col3, #T_e7ff3_row3_col7, #T_e7ff3_row4_col4, #T_e7ff3_row4_col6, #T_e7ff3_row5_col5, #T_e7ff3_row6_col4, #T_e7ff3_row6_col6, #T_e7ff3_row7_col3, #T_e7ff3_row7_col7, #T_e7ff3_row8_col8, #T_e7ff3_row9_col9, #T_e7ff3_row10_col10, #T_e7ff3_row11_col1, #T_e7ff3_row11_col11, #T_e7ff3_row12_col12, #T_e7ff3_row13_col13, #T_e7ff3_row14_col14 {
  color: red;
}
#T_e7ff3_row0_col1, #T_e7ff3_row0_col2, #T_e7ff3_row0_col3, #T_e7ff3_row0_col4, #T_e7ff3_row0_col5, #T_e7ff3_row0_col6, #T_e7ff3_row0_col7, #T_e7ff3_row0_col8, #T_e7ff3_row0_col9, #T_e7ff3_row0_col10, #T_e7ff3_row0_col11, #T_e7ff3_row0_col12, #T_e7ff3_row0_col13, #T_e7ff3_row0_col14, #T_e7ff3_row1_col0, #T_e7ff3_row1_col3, #T_e7ff3_row1_col4, #T_e7ff3_row1_col5, #T_e7ff3_row1_col6, #T_e7ff3_row1_col7, #T_e7ff3_row1_col8, #T_e7ff3_row1_col9, #T_e7ff3_row1_col10, #T_e7ff3_row1_col12, #T_e7ff3_row1_col13, #T_e7ff3_row1_col14, #T_e7ff3_row2_col0, #T_e7ff3_row2_col3, #T_e7ff3_row2_col4, #T_e7ff3_row2_col5, #T_e7ff3_row2_col6, #T_e7ff3_row2_col7, #T_e7ff3_row2_col8, #T_e7ff3_row2_col9, #T_e7ff3_row2_col10, #T_e7ff3_row2_col11, #T_e7ff3_row2_col12, #T_e7ff3_row2_col13, #T_e7ff3_row2_col14, #T_e7ff3_row3_col0, #T_e7ff3_row3_col1, #T_e7ff3_row3_col2, #T_e7ff3_row3_col4, #T_e7ff3_row3_col5, #T_e7ff3_row3_col6, #T_e7ff3_row3_col8, #T_e7ff3_row3_col9, #T_e7ff3_row3_col10, #T_e7ff3_row3_col11, #T_e7ff3_row3_col12, #T_e7ff3_row3_col13, #T_e7ff3_row3_col14, #T_e7ff3_row4_col0, #T_e7ff3_row4_col1, #T_e7ff3_row4_col2, #T_e7ff3_row4_col3, #T_e7ff3_row4_col5, #T_e7ff3_row4_col7, #T_e7ff3_row4_col8, #T_e7ff3_row4_col9, #T_e7ff3_row4_col10, #T_e7ff3_row4_col11, #T_e7ff3_row4_col12, #T_e7ff3_row4_col13, #T_e7ff3_row4_col14, #T_e7ff3_row5_col0, #T_e7ff3_row5_col1, #T_e7ff3_row5_col2, #T_e7ff3_row5_col3, #T_e7ff3_row5_col4, #T_e7ff3_row5_col6, #T_e7ff3_row5_col7, #T_e7ff3_row5_col8, #T_e7ff3_row5_col9, #T_e7ff3_row5_col10, #T_e7ff3_row5_col11, #T_e7ff3_row5_col12, #T_e7ff3_row5_col13, #T_e7ff3_row5_col14, #T_e7ff3_row6_col0, #T_e7ff3_row6_col1, #T_e7ff3_row6_col2, #T_e7ff3_row6_col3, #T_e7ff3_row6_col5, #T_e7ff3_row6_col7, #T_e7ff3_row6_col8, #T_e7ff3_row6_col9, #T_e7ff3_row6_col10, #T_e7ff3_row6_col11, #T_e7ff3_row6_col12, #T_e7ff3_row6_col13, #T_e7ff3_row6_col14, #T_e7ff3_row7_col0, #T_e7ff3_row7_col1, #T_e7ff3_row7_col2, #T_e7ff3_row7_col4, #T_e7ff3_row7_col5, #T_e7ff3_row7_col6, #T_e7ff3_row7_col8, #T_e7ff3_row7_col9, #T_e7ff3_row7_col10, #T_e7ff3_row7_col11, #T_e7ff3_row7_col12, #T_e7ff3_row7_col13, #T_e7ff3_row7_col14, #T_e7ff3_row8_col0, #T_e7ff3_row8_col1, #T_e7ff3_row8_col2, #T_e7ff3_row8_col3, #T_e7ff3_row8_col4, #T_e7ff3_row8_col5, #T_e7ff3_row8_col6, #T_e7ff3_row8_col7, #T_e7ff3_row8_col9, #T_e7ff3_row8_col10, #T_e7ff3_row8_col11, #T_e7ff3_row8_col12, #T_e7ff3_row8_col13, #T_e7ff3_row8_col14, #T_e7ff3_row9_col0, #T_e7ff3_row9_col1, #T_e7ff3_row9_col2, #T_e7ff3_row9_col3, #T_e7ff3_row9_col4, #T_e7ff3_row9_col5, #T_e7ff3_row9_col6, #T_e7ff3_row9_col7, #T_e7ff3_row9_col8, #T_e7ff3_row9_col10, #T_e7ff3_row9_col11, #T_e7ff3_row9_col12, #T_e7ff3_row9_col13, #T_e7ff3_row9_col14, #T_e7ff3_row10_col0, #T_e7ff3_row10_col1, #T_e7ff3_row10_col2, #T_e7ff3_row10_col3, #T_e7ff3_row10_col4, #T_e7ff3_row10_col5, #T_e7ff3_row10_col6, #T_e7ff3_row10_col7, #T_e7ff3_row10_col8, #T_e7ff3_row10_col9, #T_e7ff3_row10_col11, #T_e7ff3_row10_col12, #T_e7ff3_row10_col13, #T_e7ff3_row10_col14, #T_e7ff3_row11_col0, #T_e7ff3_row11_col2, #T_e7ff3_row11_col3, #T_e7ff3_row11_col4, #T_e7ff3_row11_col5, #T_e7ff3_row11_col6, #T_e7ff3_row11_col7, #T_e7ff3_row11_col8, #T_e7ff3_row11_col9, #T_e7ff3_row11_col10, #T_e7ff3_row11_col12, #T_e7ff3_row11_col13, #T_e7ff3_row11_col14, #T_e7ff3_row12_col0, #T_e7ff3_row12_col1, #T_e7ff3_row12_col2, #T_e7ff3_row12_col3, #T_e7ff3_row12_col4, #T_e7ff3_row12_col5, #T_e7ff3_row12_col6, #T_e7ff3_row12_col7, #T_e7ff3_row12_col8, #T_e7ff3_row12_col9, #T_e7ff3_row12_col10, #T_e7ff3_row12_col11, #T_e7ff3_row12_col13, #T_e7ff3_row12_col14, #T_e7ff3_row13_col0, #T_e7ff3_row13_col1, #T_e7ff3_row13_col2, #T_e7ff3_row13_col3, #T_e7ff3_row13_col4, #T_e7ff3_row13_col5, #T_e7ff3_row13_col6, #T_e7ff3_row13_col7, #T_e7ff3_row13_col8, #T_e7ff3_row13_col9, #T_e7ff3_row13_col10, #T_e7ff3_row13_col11, #T_e7ff3_row13_col12, #T_e7ff3_row13_col14, #T_e7ff3_row14_col0, #T_e7ff3_row14_col1, #T_e7ff3_row14_col2, #T_e7ff3_row14_col3, #T_e7ff3_row14_col4, #T_e7ff3_row14_col5, #T_e7ff3_row14_col6, #T_e7ff3_row14_col7, #T_e7ff3_row14_col8, #T_e7ff3_row14_col9, #T_e7ff3_row14_col10, #T_e7ff3_row14_col11, #T_e7ff3_row14_col12, #T_e7ff3_row14_col13 {
  color: black;
}
</style>
<table id="T_e7ff3">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e7ff3_level0_col0" class="col_heading level0 col0" >AGE1</th>
      <th id="T_e7ff3_level0_col1" class="col_heading level0 col1" >LMED</th>
      <th id="T_e7ff3_level0_col2" class="col_heading level0 col2" >FMR</th>
      <th id="T_e7ff3_level0_col3" class="col_heading level0 col3" >IPOV</th>
      <th id="T_e7ff3_level0_col4" class="col_heading level0 col4" >BEDRMS</th>
      <th id="T_e7ff3_level0_col5" class="col_heading level0 col5" >BUILT</th>
      <th id="T_e7ff3_level0_col6" class="col_heading level0 col6" >ROOMS</th>
      <th id="T_e7ff3_level0_col7" class="col_heading level0 col7" >PER</th>
      <th id="T_e7ff3_level0_col8" class="col_heading level0 col8" >ZINC2</th>
      <th id="T_e7ff3_level0_col9" class="col_heading level0 col9" >total_cost</th>
      <th id="T_e7ff3_level0_col10" class="col_heading level0 col10" >Midwest</th>
      <th id="T_e7ff3_level0_col11" class="col_heading level0 col11" >Northeast</th>
      <th id="T_e7ff3_level0_col12" class="col_heading level0 col12" >South</th>
      <th id="T_e7ff3_level0_col13" class="col_heading level0 col13" >central_city</th>
      <th id="T_e7ff3_level0_col14" class="col_heading level0 col14" >Adequate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e7ff3_level0_row0" class="row_heading level0 row0" >AGE1</th>
      <td id="T_e7ff3_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_e7ff3_row0_col1" class="data row0 col1" >-0.010146</td>
      <td id="T_e7ff3_row0_col2" class="data row0 col2" >-0.040166</td>
      <td id="T_e7ff3_row0_col3" class="data row0 col3" >-0.440358</td>
      <td id="T_e7ff3_row0_col4" class="data row0 col4" >-0.108804</td>
      <td id="T_e7ff3_row0_col5" class="data row0 col5" >-0.138537</td>
      <td id="T_e7ff3_row0_col6" class="data row0 col6" >-0.046702</td>
      <td id="T_e7ff3_row0_col7" class="data row0 col7" >-0.401749</td>
      <td id="T_e7ff3_row0_col8" class="data row0 col8" >-0.206399</td>
      <td id="T_e7ff3_row0_col9" class="data row0 col9" >-0.237089</td>
      <td id="T_e7ff3_row0_col10" class="data row0 col10" >-0.013970</td>
      <td id="T_e7ff3_row0_col11" class="data row0 col11" >0.019804</td>
      <td id="T_e7ff3_row0_col12" class="data row0 col12" >0.000219</td>
      <td id="T_e7ff3_row0_col13" class="data row0 col13" >-0.018975</td>
      <td id="T_e7ff3_row0_col14" class="data row0 col14" >0.002276</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row1" class="row_heading level0 row1" >LMED</th>
      <td id="T_e7ff3_row1_col0" class="data row1 col0" >-0.010146</td>
      <td id="T_e7ff3_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_e7ff3_row1_col2" class="data row1 col2" >0.683971</td>
      <td id="T_e7ff3_row1_col3" class="data row1 col3" >0.078243</td>
      <td id="T_e7ff3_row1_col4" class="data row1 col4" >0.100760</td>
      <td id="T_e7ff3_row1_col5" class="data row1 col5" >-0.158059</td>
      <td id="T_e7ff3_row1_col6" class="data row1 col6" >0.123096</td>
      <td id="T_e7ff3_row1_col7" class="data row1 col7" >0.077243</td>
      <td id="T_e7ff3_row1_col8" class="data row1 col8" >0.172790</td>
      <td id="T_e7ff3_row1_col9" class="data row1 col9" >0.329919</td>
      <td id="T_e7ff3_row1_col10" class="data row1 col10" >-0.152054</td>
      <td id="T_e7ff3_row1_col11" class="data row1 col11" >0.541857</td>
      <td id="T_e7ff3_row1_col12" class="data row1 col12" >-0.372614</td>
      <td id="T_e7ff3_row1_col13" class="data row1 col13" >-0.007961</td>
      <td id="T_e7ff3_row1_col14" class="data row1 col14" >0.007868</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row2" class="row_heading level0 row2" >FMR</th>
      <td id="T_e7ff3_row2_col0" class="data row2 col0" >-0.040166</td>
      <td id="T_e7ff3_row2_col1" class="data row2 col1" >0.683971</td>
      <td id="T_e7ff3_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_e7ff3_row2_col3" class="data row2 col3" >0.216047</td>
      <td id="T_e7ff3_row2_col4" class="data row2 col4" >0.478068</td>
      <td id="T_e7ff3_row2_col5" class="data row2 col5" >-0.009525</td>
      <td id="T_e7ff3_row2_col6" class="data row2 col6" >0.349391</td>
      <td id="T_e7ff3_row2_col7" class="data row2 col7" >0.218579</td>
      <td id="T_e7ff3_row2_col8" class="data row2 col8" >0.248834</td>
      <td id="T_e7ff3_row2_col9" class="data row2 col9" >0.456375</td>
      <td id="T_e7ff3_row2_col10" class="data row2 col10" >-0.380552</td>
      <td id="T_e7ff3_row2_col11" class="data row2 col11" >0.330926</td>
      <td id="T_e7ff3_row2_col12" class="data row2 col12" >-0.197852</td>
      <td id="T_e7ff3_row2_col13" class="data row2 col13" >0.052698</td>
      <td id="T_e7ff3_row2_col14" class="data row2 col14" >0.017112</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row3" class="row_heading level0 row3" >IPOV</th>
      <td id="T_e7ff3_row3_col0" class="data row3 col0" >-0.440358</td>
      <td id="T_e7ff3_row3_col1" class="data row3 col1" >0.078243</td>
      <td id="T_e7ff3_row3_col2" class="data row3 col2" >0.216047</td>
      <td id="T_e7ff3_row3_col3" class="data row3 col3" >1.000000</td>
      <td id="T_e7ff3_row3_col4" class="data row3 col4" >0.329640</td>
      <td id="T_e7ff3_row3_col5" class="data row3 col5" >0.088701</td>
      <td id="T_e7ff3_row3_col6" class="data row3 col6" >0.259868</td>
      <td id="T_e7ff3_row3_col7" class="data row3 col7" >0.989618</td>
      <td id="T_e7ff3_row3_col8" class="data row3 col8" >0.242891</td>
      <td id="T_e7ff3_row3_col9" class="data row3 col9" >0.299183</td>
      <td id="T_e7ff3_row3_col10" class="data row3 col10" >-0.014878</td>
      <td id="T_e7ff3_row3_col11" class="data row3 col11" >0.034332</td>
      <td id="T_e7ff3_row3_col12" class="data row3 col12" >-0.043398</td>
      <td id="T_e7ff3_row3_col13" class="data row3 col13" >0.004929</td>
      <td id="T_e7ff3_row3_col14" class="data row3 col14" >-0.005402</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row4" class="row_heading level0 row4" >BEDRMS</th>
      <td id="T_e7ff3_row4_col0" class="data row4 col0" >-0.108804</td>
      <td id="T_e7ff3_row4_col1" class="data row4 col1" >0.100760</td>
      <td id="T_e7ff3_row4_col2" class="data row4 col2" >0.478068</td>
      <td id="T_e7ff3_row4_col3" class="data row4 col3" >0.329640</td>
      <td id="T_e7ff3_row4_col4" class="data row4 col4" >1.000000</td>
      <td id="T_e7ff3_row4_col5" class="data row4 col5" >0.141762</td>
      <td id="T_e7ff3_row4_col6" class="data row4 col6" >0.744225</td>
      <td id="T_e7ff3_row4_col7" class="data row4 col7" >0.333674</td>
      <td id="T_e7ff3_row4_col8" class="data row4 col8" >0.276075</td>
      <td id="T_e7ff3_row4_col9" class="data row4 col9" >0.350501</td>
      <td id="T_e7ff3_row4_col10" class="data row4 col10" >-0.034834</td>
      <td id="T_e7ff3_row4_col11" class="data row4 col11" >0.013665</td>
      <td id="T_e7ff3_row4_col12" class="data row4 col12" >-0.007476</td>
      <td id="T_e7ff3_row4_col13" class="data row4 col13" >-0.031947</td>
      <td id="T_e7ff3_row4_col14" class="data row4 col14" >0.031496</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row5" class="row_heading level0 row5" >BUILT</th>
      <td id="T_e7ff3_row5_col0" class="data row5 col0" >-0.138537</td>
      <td id="T_e7ff3_row5_col1" class="data row5 col1" >-0.158059</td>
      <td id="T_e7ff3_row5_col2" class="data row5 col2" >-0.009525</td>
      <td id="T_e7ff3_row5_col3" class="data row5 col3" >0.088701</td>
      <td id="T_e7ff3_row5_col4" class="data row5 col4" >0.141762</td>
      <td id="T_e7ff3_row5_col5" class="data row5 col5" >1.000000</td>
      <td id="T_e7ff3_row5_col6" class="data row5 col6" >0.142026</td>
      <td id="T_e7ff3_row5_col7" class="data row5 col7" >0.088280</td>
      <td id="T_e7ff3_row5_col8" class="data row5 col8" >0.126572</td>
      <td id="T_e7ff3_row5_col9" class="data row5 col9" >0.141853</td>
      <td id="T_e7ff3_row5_col10" class="data row5 col10" >-0.089905</td>
      <td id="T_e7ff3_row5_col11" class="data row5 col11" >-0.207186</td>
      <td id="T_e7ff3_row5_col12" class="data row5 col12" >0.212904</td>
      <td id="T_e7ff3_row5_col13" class="data row5 col13" >-0.151060</td>
      <td id="T_e7ff3_row5_col14" class="data row5 col14" >0.089468</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row6" class="row_heading level0 row6" >ROOMS</th>
      <td id="T_e7ff3_row6_col0" class="data row6 col0" >-0.046702</td>
      <td id="T_e7ff3_row6_col1" class="data row6 col1" >0.123096</td>
      <td id="T_e7ff3_row6_col2" class="data row6 col2" >0.349391</td>
      <td id="T_e7ff3_row6_col3" class="data row6 col3" >0.259868</td>
      <td id="T_e7ff3_row6_col4" class="data row6 col4" >0.744225</td>
      <td id="T_e7ff3_row6_col5" class="data row6 col5" >0.142026</td>
      <td id="T_e7ff3_row6_col6" class="data row6 col6" >1.000000</td>
      <td id="T_e7ff3_row6_col7" class="data row6 col7" >0.268147</td>
      <td id="T_e7ff3_row6_col8" class="data row6 col8" >0.353855</td>
      <td id="T_e7ff3_row6_col9" class="data row6 col9" >0.401176</td>
      <td id="T_e7ff3_row6_col10" class="data row6 col10" >-0.021500</td>
      <td id="T_e7ff3_row6_col11" class="data row6 col11" >0.037715</td>
      <td id="T_e7ff3_row6_col12" class="data row6 col12" >-0.016843</td>
      <td id="T_e7ff3_row6_col13" class="data row6 col13" >-0.048646</td>
      <td id="T_e7ff3_row6_col14" class="data row6 col14" >0.043317</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row7" class="row_heading level0 row7" >PER</th>
      <td id="T_e7ff3_row7_col0" class="data row7 col0" >-0.401749</td>
      <td id="T_e7ff3_row7_col1" class="data row7 col1" >0.077243</td>
      <td id="T_e7ff3_row7_col2" class="data row7 col2" >0.218579</td>
      <td id="T_e7ff3_row7_col3" class="data row7 col3" >0.989618</td>
      <td id="T_e7ff3_row7_col4" class="data row7 col4" >0.333674</td>
      <td id="T_e7ff3_row7_col5" class="data row7 col5" >0.088280</td>
      <td id="T_e7ff3_row7_col6" class="data row7 col6" >0.268147</td>
      <td id="T_e7ff3_row7_col7" class="data row7 col7" >1.000000</td>
      <td id="T_e7ff3_row7_col8" class="data row7 col8" >0.242559</td>
      <td id="T_e7ff3_row7_col9" class="data row7 col9" >0.292750</td>
      <td id="T_e7ff3_row7_col10" class="data row7 col10" >-0.015877</td>
      <td id="T_e7ff3_row7_col11" class="data row7 col11" >0.035158</td>
      <td id="T_e7ff3_row7_col12" class="data row7 col12" >-0.043685</td>
      <td id="T_e7ff3_row7_col13" class="data row7 col13" >0.000784</td>
      <td id="T_e7ff3_row7_col14" class="data row7 col14" >-0.003385</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row8" class="row_heading level0 row8" >ZINC2</th>
      <td id="T_e7ff3_row8_col0" class="data row8 col0" >-0.206399</td>
      <td id="T_e7ff3_row8_col1" class="data row8 col1" >0.172790</td>
      <td id="T_e7ff3_row8_col2" class="data row8 col2" >0.248834</td>
      <td id="T_e7ff3_row8_col3" class="data row8 col3" >0.242891</td>
      <td id="T_e7ff3_row8_col4" class="data row8 col4" >0.276075</td>
      <td id="T_e7ff3_row8_col5" class="data row8 col5" >0.126572</td>
      <td id="T_e7ff3_row8_col6" class="data row8 col6" >0.353855</td>
      <td id="T_e7ff3_row8_col7" class="data row8 col7" >0.242559</td>
      <td id="T_e7ff3_row8_col8" class="data row8 col8" >1.000000</td>
      <td id="T_e7ff3_row8_col9" class="data row8 col9" >0.462820</td>
      <td id="T_e7ff3_row8_col10" class="data row8 col10" >-0.049765</td>
      <td id="T_e7ff3_row8_col11" class="data row8 col11" >0.074081</td>
      <td id="T_e7ff3_row8_col12" class="data row8 col12" >-0.052341</td>
      <td id="T_e7ff3_row8_col13" class="data row8 col13" >-0.034424</td>
      <td id="T_e7ff3_row8_col14" class="data row8 col14" >0.045125</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row9" class="row_heading level0 row9" >total_cost</th>
      <td id="T_e7ff3_row9_col0" class="data row9 col0" >-0.237089</td>
      <td id="T_e7ff3_row9_col1" class="data row9 col1" >0.329919</td>
      <td id="T_e7ff3_row9_col2" class="data row9 col2" >0.456375</td>
      <td id="T_e7ff3_row9_col3" class="data row9 col3" >0.299183</td>
      <td id="T_e7ff3_row9_col4" class="data row9 col4" >0.350501</td>
      <td id="T_e7ff3_row9_col5" class="data row9 col5" >0.141853</td>
      <td id="T_e7ff3_row9_col6" class="data row9 col6" >0.401176</td>
      <td id="T_e7ff3_row9_col7" class="data row9 col7" >0.292750</td>
      <td id="T_e7ff3_row9_col8" class="data row9 col8" >0.462820</td>
      <td id="T_e7ff3_row9_col9" class="data row9 col9" >1.000000</td>
      <td id="T_e7ff3_row9_col10" class="data row9 col10" >-0.131444</td>
      <td id="T_e7ff3_row9_col11" class="data row9 col11" >0.180337</td>
      <td id="T_e7ff3_row9_col12" class="data row9 col12" >-0.111354</td>
      <td id="T_e7ff3_row9_col13" class="data row9 col13" >-0.013059</td>
      <td id="T_e7ff3_row9_col14" class="data row9 col14" >0.026506</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row10" class="row_heading level0 row10" >Midwest</th>
      <td id="T_e7ff3_row10_col0" class="data row10 col0" >-0.013970</td>
      <td id="T_e7ff3_row10_col1" class="data row10 col1" >-0.152054</td>
      <td id="T_e7ff3_row10_col2" class="data row10 col2" >-0.380552</td>
      <td id="T_e7ff3_row10_col3" class="data row10 col3" >-0.014878</td>
      <td id="T_e7ff3_row10_col4" class="data row10 col4" >-0.034834</td>
      <td id="T_e7ff3_row10_col5" class="data row10 col5" >-0.089905</td>
      <td id="T_e7ff3_row10_col6" class="data row10 col6" >-0.021500</td>
      <td id="T_e7ff3_row10_col7" class="data row10 col7" >-0.015877</td>
      <td id="T_e7ff3_row10_col8" class="data row10 col8" >-0.049765</td>
      <td id="T_e7ff3_row10_col9" class="data row10 col9" >-0.131444</td>
      <td id="T_e7ff3_row10_col10" class="data row10 col10" >1.000000</td>
      <td id="T_e7ff3_row10_col11" class="data row10 col11" >-0.371976</td>
      <td id="T_e7ff3_row10_col12" class="data row10 col12" >-0.422289</td>
      <td id="T_e7ff3_row10_col13" class="data row10 col13" >-0.016145</td>
      <td id="T_e7ff3_row10_col14" class="data row10 col14" >0.009619</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row11" class="row_heading level0 row11" >Northeast</th>
      <td id="T_e7ff3_row11_col0" class="data row11 col0" >0.019804</td>
      <td id="T_e7ff3_row11_col1" class="data row11 col1" >0.541857</td>
      <td id="T_e7ff3_row11_col2" class="data row11 col2" >0.330926</td>
      <td id="T_e7ff3_row11_col3" class="data row11 col3" >0.034332</td>
      <td id="T_e7ff3_row11_col4" class="data row11 col4" >0.013665</td>
      <td id="T_e7ff3_row11_col5" class="data row11 col5" >-0.207186</td>
      <td id="T_e7ff3_row11_col6" class="data row11 col6" >0.037715</td>
      <td id="T_e7ff3_row11_col7" class="data row11 col7" >0.035158</td>
      <td id="T_e7ff3_row11_col8" class="data row11 col8" >0.074081</td>
      <td id="T_e7ff3_row11_col9" class="data row11 col9" >0.180337</td>
      <td id="T_e7ff3_row11_col10" class="data row11 col10" >-0.371976</td>
      <td id="T_e7ff3_row11_col11" class="data row11 col11" >1.000000</td>
      <td id="T_e7ff3_row11_col12" class="data row11 col12" >-0.368684</td>
      <td id="T_e7ff3_row11_col13" class="data row11 col13" >-0.059059</td>
      <td id="T_e7ff3_row11_col14" class="data row11 col14" >-0.022263</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row12" class="row_heading level0 row12" >South</th>
      <td id="T_e7ff3_row12_col0" class="data row12 col0" >0.000219</td>
      <td id="T_e7ff3_row12_col1" class="data row12 col1" >-0.372614</td>
      <td id="T_e7ff3_row12_col2" class="data row12 col2" >-0.197852</td>
      <td id="T_e7ff3_row12_col3" class="data row12 col3" >-0.043398</td>
      <td id="T_e7ff3_row12_col4" class="data row12 col4" >-0.007476</td>
      <td id="T_e7ff3_row12_col5" class="data row12 col5" >0.212904</td>
      <td id="T_e7ff3_row12_col6" class="data row12 col6" >-0.016843</td>
      <td id="T_e7ff3_row12_col7" class="data row12 col7" >-0.043685</td>
      <td id="T_e7ff3_row12_col8" class="data row12 col8" >-0.052341</td>
      <td id="T_e7ff3_row12_col9" class="data row12 col9" >-0.111354</td>
      <td id="T_e7ff3_row12_col10" class="data row12 col10" >-0.422289</td>
      <td id="T_e7ff3_row12_col11" class="data row12 col11" >-0.368684</td>
      <td id="T_e7ff3_row12_col12" class="data row12 col12" >1.000000</td>
      <td id="T_e7ff3_row12_col13" class="data row12 col13" >0.000810</td>
      <td id="T_e7ff3_row12_col14" class="data row12 col14" >-0.010960</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row13" class="row_heading level0 row13" >central_city</th>
      <td id="T_e7ff3_row13_col0" class="data row13 col0" >-0.018975</td>
      <td id="T_e7ff3_row13_col1" class="data row13 col1" >-0.007961</td>
      <td id="T_e7ff3_row13_col2" class="data row13 col2" >0.052698</td>
      <td id="T_e7ff3_row13_col3" class="data row13 col3" >0.004929</td>
      <td id="T_e7ff3_row13_col4" class="data row13 col4" >-0.031947</td>
      <td id="T_e7ff3_row13_col5" class="data row13 col5" >-0.151060</td>
      <td id="T_e7ff3_row13_col6" class="data row13 col6" >-0.048646</td>
      <td id="T_e7ff3_row13_col7" class="data row13 col7" >0.000784</td>
      <td id="T_e7ff3_row13_col8" class="data row13 col8" >-0.034424</td>
      <td id="T_e7ff3_row13_col9" class="data row13 col9" >-0.013059</td>
      <td id="T_e7ff3_row13_col10" class="data row13 col10" >-0.016145</td>
      <td id="T_e7ff3_row13_col11" class="data row13 col11" >-0.059059</td>
      <td id="T_e7ff3_row13_col12" class="data row13 col12" >0.000810</td>
      <td id="T_e7ff3_row13_col13" class="data row13 col13" >1.000000</td>
      <td id="T_e7ff3_row13_col14" class="data row13 col14" >-0.029027</td>
    </tr>
    <tr>
      <th id="T_e7ff3_level0_row14" class="row_heading level0 row14" >Adequate</th>
      <td id="T_e7ff3_row14_col0" class="data row14 col0" >0.002276</td>
      <td id="T_e7ff3_row14_col1" class="data row14 col1" >0.007868</td>
      <td id="T_e7ff3_row14_col2" class="data row14 col2" >0.017112</td>
      <td id="T_e7ff3_row14_col3" class="data row14 col3" >-0.005402</td>
      <td id="T_e7ff3_row14_col4" class="data row14 col4" >0.031496</td>
      <td id="T_e7ff3_row14_col5" class="data row14 col5" >0.089468</td>
      <td id="T_e7ff3_row14_col6" class="data row14 col6" >0.043317</td>
      <td id="T_e7ff3_row14_col7" class="data row14 col7" >-0.003385</td>
      <td id="T_e7ff3_row14_col8" class="data row14 col8" >0.045125</td>
      <td id="T_e7ff3_row14_col9" class="data row14 col9" >0.026506</td>
      <td id="T_e7ff3_row14_col10" class="data row14 col10" >0.009619</td>
      <td id="T_e7ff3_row14_col11" class="data row14 col11" >-0.022263</td>
      <td id="T_e7ff3_row14_col12" class="data row14 col12" >-0.010960</td>
      <td id="T_e7ff3_row14_col13" class="data row14 col13" >-0.029027</td>
      <td id="T_e7ff3_row14_col14" class="data row14 col14" >1.000000</td>
    </tr>
  </tbody>
</table>





```python
# From the correlation matrix above let's drop variables with correlation values greater than 0.5
X_cleaned = X.drop(columns=["BEDRMS"])
```


```python
corr.to_csv("correlation.csv")
```


```python
# Empirical distribution of the Variable "VALUE"
warnings.filterwarnings("ignore")
val = sns.displot(data=owned_single_family_housing_df, x='VALUE', height=3, aspect=0.8, 
            kde=True, col="REGION")
val.set_axis_labels("Market Value ($)", "Number of Units")
val.set_titles("{col_name} Region");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_88_0.png)
    


The distribution of `VALUE` is right skewed. A log transformation could help to make the distribution more bell sahped!


```python
LNVALUE = np.log(owned_single_family_housing_df.VALUE)
owned_single_family_housing_df['ln_value'] = LNVALUE
```


```python
warnings.filterwarnings("ignore")
ln_val = sns.displot(data = owned_single_family_housing_df, x='ln_value', height=3, aspect=0.8, 
            kde=True, col="REGION")
ln_val.set_axis_labels("Ln(VALUE)", "Number of Units")
ln_val.set_titles("{col_name} Region");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_91_0.png)
    


After transformation, the data became normally distributed with equal mean and median of 12, and standard variation of 1. Also the distribution curve above shaow that the `ln_value` variable is normally distributed. Thus, we will use the transformed variable `ln_value` to build the linear regression model.


```python
# lower triangular matrix
mask = np.triu(np.ones_like(X.corr()))
warnings.filterwarnings("ignore")
# plotting a triangle correlation heatmap
sns.heatmap(X.corr(), cmap="YlGnBu", annot=True, fmt='0.1f', mask=mask);
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_93_0.png)
    


### Linear Regression Model


```python
y = owned_single_family_housing_df["VALUE"]
X_new = sm.add_constant(X_cleaned)
# Create the linear model
model = sm.OLS(y, X_new)
# Fit the model
results = model.fit()
# Print out the summary of the model results
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>VALUE</td>      <th>  R-squared:         </th>  <td>   0.470</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.469</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1908.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 14 Jan 2024</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>21:46:13</td>     <th>  Log-Likelihood:    </th> <td>-4.1189e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 30174</td>      <th>  AIC:               </th>  <td>8.238e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 30159</td>      <th>  BIC:               </th>  <td>8.239e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    14</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>  1.79e+05</td> <td> 9.79e+04</td> <td>    1.828</td> <td> 0.068</td> <td>-1.29e+04</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>AGE1</th>         <td> 2136.8019</td> <td>   90.752</td> <td>   23.546</td> <td> 0.000</td> <td> 1958.925</td> <td> 2314.679</td>
</tr>
<tr>
  <th>LMED</th>         <td>   -0.0122</td> <td>    0.160</td> <td>   -0.076</td> <td> 0.939</td> <td>   -0.325</td> <td>    0.301</td>
</tr>
<tr>
  <th>FMR</th>          <td>  126.7216</td> <td>    5.442</td> <td>   23.285</td> <td> 0.000</td> <td>  116.055</td> <td>  137.388</td>
</tr>
<tr>
  <th>IPOV</th>         <td>   -9.6768</td> <td>    1.463</td> <td>   -6.615</td> <td> 0.000</td> <td>  -12.544</td> <td>   -6.809</td>
</tr>
<tr>
  <th>BUILT</th>        <td> -186.8934</td> <td>   48.683</td> <td>   -3.839</td> <td> 0.000</td> <td> -282.314</td> <td>  -91.473</td>
</tr>
<tr>
  <th>ROOMS</th>        <td> 1.391e+04</td> <td>  869.253</td> <td>   15.999</td> <td> 0.000</td> <td> 1.22e+04</td> <td> 1.56e+04</td>
</tr>
<tr>
  <th>PER</th>          <td> 2.043e+04</td> <td> 5991.085</td> <td>    3.411</td> <td> 0.001</td> <td> 8692.119</td> <td> 3.22e+04</td>
</tr>
<tr>
  <th>ZINC2</th>        <td>    0.4793</td> <td>    0.016</td> <td>   29.941</td> <td> 0.000</td> <td>    0.448</td> <td>    0.511</td>
</tr>
<tr>
  <th>total_cost</th>   <td>  114.3134</td> <td>    1.295</td> <td>   88.272</td> <td> 0.000</td> <td>  111.775</td> <td>  116.852</td>
</tr>
<tr>
  <th>Midwest</th>      <td>-7.526e+04</td> <td> 4435.001</td> <td>  -16.969</td> <td> 0.000</td> <td> -8.4e+04</td> <td>-6.66e+04</td>
</tr>
<tr>
  <th>Northeast</th>    <td>-7.382e+04</td> <td> 4405.075</td> <td>  -16.759</td> <td> 0.000</td> <td>-8.25e+04</td> <td>-6.52e+04</td>
</tr>
<tr>
  <th>South</th>        <td>-6.273e+04</td> <td> 4019.413</td> <td>  -15.607</td> <td> 0.000</td> <td>-7.06e+04</td> <td>-5.49e+04</td>
</tr>
<tr>
  <th>central_city</th> <td>-1.036e+04</td> <td> 2959.067</td> <td>   -3.502</td> <td> 0.000</td> <td>-1.62e+04</td> <td>-4562.801</td>
</tr>
<tr>
  <th>Adequate</th>     <td> 1.349e+04</td> <td> 7918.202</td> <td>    1.703</td> <td> 0.089</td> <td>-2034.515</td> <td>  2.9e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>30606.729</td> <th>  Durbin-Watson:     </th>  <td>   1.914</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>2336359.516</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 4.978</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>44.943</td>   <th>  Cond. No.          </th>  <td>1.14e+07</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.14e+07. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# Remove variables that may cause a multicollinearity issue
X_reduced = X.drop(columns=["BEDRMS"])
# VIF dataframe 
vif_data_2 = pd.DataFrame() 
vif_data_2["feature"] = X_reduced.columns 
  
# calculating VIF for each feature 
vif_data_2["VIF"] = [variance_inflation_factor(X_reduced.values, i) 
                          for i in range(len(X_reduced.columns))] 
  
print(vif_data_2)
```

             feature         VIF
    0           AGE1   18.989908
    1           LMED   87.123319
    2            FMR   38.378476
    3           IPOV  528.409464
    4          BUILT  220.863768
    5          ROOMS   26.359082
    6            PER  228.799280
    7          ZINC2    2.869590
    8     total_cost    5.239554
    9        Midwest    4.144784
    10     Northeast    3.330547
    11         South    3.414944
    12  central_city    1.293290
    13      Adequate   43.886523
    


```python
X_reduced["ln_total_cost"] = np.log(X_reduced.total_cost)
X_reduced["sqrt_ZINC2"] = np.sqrt(X_reduced.ZINC2)
X_reduced["ln_lmed"] = np.log(X_reduced.LMED)
X_reduced["ln_fmr"] = np.log(X_reduced.FMR)
X_reduced["ln_ipov"] = np.log(X_reduced.IPOV)
X_reduced_cleaned = X_reduced.drop(columns=["total_cost", "ZINC2", "LMED", "FMR", "IPOV"])
X_reduced_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE1</th>
      <th>BUILT</th>
      <th>ROOMS</th>
      <th>PER</th>
      <th>Midwest</th>
      <th>Northeast</th>
      <th>South</th>
      <th>central_city</th>
      <th>Adequate</th>
      <th>ln_total_cost</th>
      <th>sqrt_ZINC2</th>
      <th>ln_lmed</th>
      <th>ln_fmr</th>
      <th>ln_ipov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>82</td>
      <td>2006</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6.819743</td>
      <td>134.242318</td>
      <td>11.208274</td>
      <td>6.862758</td>
      <td>9.311723</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>1980</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6.672876</td>
      <td>350.657953</td>
      <td>10.930353</td>
      <td>7.003065</td>
      <td>10.094851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>1985</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>7.378696</td>
      <td>167.254297</td>
      <td>10.930353</td>
      <td>7.003065</td>
      <td>9.646658</td>
    </tr>
    <tr>
      <th>3</th>
      <td>67</td>
      <td>1985</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6.270358</td>
      <td>179.499304</td>
      <td>10.930353</td>
      <td>6.855409</td>
      <td>9.544238</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>1985</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7.297091</td>
      <td>264.503308</td>
      <td>11.018482</td>
      <td>6.895683</td>
      <td>9.800901</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = owned_single_family_housing_df["ln_value"]
X_reduced_new = sm.add_constant(X_reduced_cleaned)
# Create the linear model
model_reduced = sm.OLS(y, X_reduced_new)
# Fit the model
results_reduced = model_reduced.fit()
# Print out the summary of the model results
results_reduced.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>ln_value</td>     <th>  R-squared:         </th> <td>   0.544</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.544</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2573.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 14 Jan 2024</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>21:46:14</td>     <th>  Log-Likelihood:    </th> <td> -23919.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 30174</td>      <th>  AIC:               </th> <td>4.787e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 30159</td>      <th>  BIC:               </th> <td>4.799e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    14</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>         <td>    0.1251</td> <td>    0.682</td> <td>    0.183</td> <td> 0.854</td> <td>   -1.211</td> <td>    1.462</td>
</tr>
<tr>
  <th>AGE1</th>          <td>    0.0068</td> <td>    0.000</td> <td>   26.339</td> <td> 0.000</td> <td>    0.006</td> <td>    0.007</td>
</tr>
<tr>
  <th>BUILT</th>         <td>    0.0026</td> <td>    0.000</td> <td>   20.218</td> <td> 0.000</td> <td>    0.002</td> <td>    0.003</td>
</tr>
<tr>
  <th>ROOMS</th>         <td>    0.0667</td> <td>    0.002</td> <td>   28.952</td> <td> 0.000</td> <td>    0.062</td> <td>    0.071</td>
</tr>
<tr>
  <th>PER</th>           <td>    0.0469</td> <td>    0.012</td> <td>    3.792</td> <td> 0.000</td> <td>    0.023</td> <td>    0.071</td>
</tr>
<tr>
  <th>Midwest</th>       <td>   -0.3301</td> <td>    0.012</td> <td>  -28.660</td> <td> 0.000</td> <td>   -0.353</td> <td>   -0.307</td>
</tr>
<tr>
  <th>Northeast</th>     <td>   -0.1729</td> <td>    0.011</td> <td>  -15.331</td> <td> 0.000</td> <td>   -0.195</td> <td>   -0.151</td>
</tr>
<tr>
  <th>South</th>         <td>   -0.2409</td> <td>    0.010</td> <td>  -23.158</td> <td> 0.000</td> <td>   -0.261</td> <td>   -0.221</td>
</tr>
<tr>
  <th>central_city</th>  <td>   -0.0811</td> <td>    0.008</td> <td>  -10.501</td> <td> 0.000</td> <td>   -0.096</td> <td>   -0.066</td>
</tr>
<tr>
  <th>Adequate</th>      <td>    0.1372</td> <td>    0.021</td> <td>    6.648</td> <td> 0.000</td> <td>    0.097</td> <td>    0.178</td>
</tr>
<tr>
  <th>ln_total_cost</th> <td>    0.5033</td> <td>    0.007</td> <td>   75.331</td> <td> 0.000</td> <td>    0.490</td> <td>    0.516</td>
</tr>
<tr>
  <th>sqrt_ZINC2</th>    <td>    0.0012</td> <td> 3.06e-05</td> <td>   39.636</td> <td> 0.000</td> <td>    0.001</td> <td>    0.001</td>
</tr>
<tr>
  <th>ln_lmed</th>       <td>    0.3095</td> <td>    0.029</td> <td>   10.507</td> <td> 0.000</td> <td>    0.252</td> <td>    0.367</td>
</tr>
<tr>
  <th>ln_fmr</th>        <td>    0.4461</td> <td>    0.019</td> <td>   23.185</td> <td> 0.000</td> <td>    0.408</td> <td>    0.484</td>
</tr>
<tr>
  <th>ln_ipov</th>       <td>   -0.4627</td> <td>    0.061</td> <td>   -7.583</td> <td> 0.000</td> <td>   -0.582</td> <td>   -0.343</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2465.572</td> <th>  Durbin-Watson:     </th> <td>   1.851</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>9572.312</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-0.345</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.672</td>  <th>  Cond. No.          </th> <td>4.42e+05</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.42e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Analysis of Variance


```python
# Ordinary Least Squares (OLS) model
model_aov = ols('ln_value ~ C(REGION)', data = owned_single_family_housing_df).fit()
anova_table = sm.stats.anova_lm(model_aov, typ=2)
anova_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(REGION)</th>
      <td>2305.719476</td>
      <td>3.0</td>
      <td>1395.326198</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>16618.230365</td>
      <td>30170.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Independent ttest by Region
region_northwest = owned_single_family_housing_df[owned_single_family_housing_df.REGION=="Northeast"]["VALUE"]
region_west = owned_single_family_housing_df[owned_single_family_housing_df.REGION=="West"]["VALUE"]
region_south = owned_single_family_housing_df[owned_single_family_housing_df.REGION=="South"]["VALUE"]
region_midwest = owned_single_family_housing_df[owned_single_family_housing_df.REGION=="Midwest"]["VALUE"]
```


```python
res_region = stats.tukey_hsd(region_northwest, region_west, region_south, region_midwest)
type(res_region)
```




    scipy.stats._hypotests.TukeyHSDResult



Using Tukey HSD test, all the means of current market values are different throughout all 4 regions as shown in the table above. P-values of all groups are close to zero, and thus the null hypothesis is rejected. The null hypothesis claims that the means between two groups are equal.  

### Summary Tables


```python
pd.pivot_table(owned_single_family_housing_df, values=["IPOV", "VALUE", "LMED", "FMR", 
            "total_cost", "ZINC2", "ROOMS", "PER", "ln_value"], 
               index=['REGION']).round(2).sort_values(by="VALUE").reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REGION</th>
      <th>FMR</th>
      <th>IPOV</th>
      <th>LMED</th>
      <th>PER</th>
      <th>ROOMS</th>
      <th>VALUE</th>
      <th>ZINC2</th>
      <th>ln_value</th>
      <th>total_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Midwest</td>
      <td>1053.29</td>
      <td>17654.28</td>
      <td>65497.60</td>
      <td>2.62</td>
      <td>6.73</td>
      <td>186870.77</td>
      <td>84145.04</td>
      <td>11.86</td>
      <td>1502.39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>South</td>
      <td>1163.39</td>
      <td>17390.63</td>
      <td>61205.95</td>
      <td>2.55</td>
      <td>6.74</td>
      <td>216831.41</td>
      <td>83743.34</td>
      <td>12.00</td>
      <td>1536.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Northeast</td>
      <td>1515.51</td>
      <td>18148.62</td>
      <td>80309.21</td>
      <td>2.74</td>
      <td>6.90</td>
      <td>328229.01</td>
      <td>101913.84</td>
      <td>12.44</td>
      <td>2113.13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>1585.94</td>
      <td>18227.47</td>
      <td>68912.90</td>
      <td>2.76</td>
      <td>6.80</td>
      <td>387585.92</td>
      <td>98622.69</td>
      <td>12.54</td>
      <td>1984.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compare the mean and median of the market value of unit befor and after the log transformation
pd.pivot_table(owned_single_family_housing_df, values=["VALUE", "ln_value"], index=['REGION'], 
              aggfunc = {"ln_value": ["mean", "median"],
                         "VALUE": ["mean", "median"]}).round(2).reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>REGION</th>
      <th colspan="2" halign="left">VALUE</th>
      <th colspan="2" halign="left">ln_value</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Midwest</td>
      <td>186870.77</td>
      <td>150000.0</td>
      <td>11.86</td>
      <td>11.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Northeast</td>
      <td>328229.01</td>
      <td>270000.0</td>
      <td>12.44</td>
      <td>12.51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South</td>
      <td>216831.41</td>
      <td>160000.0</td>
      <td>12.00</td>
      <td>11.98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>387585.92</td>
      <td>280000.0</td>
      <td>12.54</td>
      <td>12.54</td>
    </tr>
  </tbody>
</table>
</div>



Mean and median are approximately equal throughout the 4 regions after the Variable `VALUE` was log transformed. Howver, before it was transformed, the mean and median seemed to be significantly different, which implies that current market values of housing units are right skewed in this case since means are higher than medians.

### Use Scikitlearn to run the above Regression Analysis

#### Importing Required Packages


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
```


```python
y = owned_single_family_housing_df["ln_value"]
```


```python
# Sperate train and test data
X_train, X_test, y_train, y_test = train_test_split(X_reduced_cleaned, y, test_size = 0.2, random_state=42)
print("The number of samples in train set is {}".format(X_train.shape[0]))
print("The number of samples in test set is {}".format(X_test.shape[0]))
```

    The number of samples in train set is 24139
    The number of samples in test set is 6035
    


```python
# (1) Initiate the model
lr = LinearRegression()
# (2) Fit the model
lr.fit(X_train, y_train)
# (3) Score the model
lr.score(X_test, y_test)
```




    0.5287333822667113




```python
# Coefficients
lr.coef_
```




    array([ 0.00684111,  0.00239845,  0.06792728,  0.04411304, -0.33905494,
           -0.17960855, -0.24822985, -0.08106516,  0.14785642,  0.50224208,
            0.00123396,  0.29875615,  0.44772303, -0.44009229])




```python
# Intercept
lr.intercept_
```




    0.3579597451082446




```python
# Predict
y_pred = lr.predict(X_test)
# Mean Square error
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error {}".format(mse))
```

    Mean Squared Error 0.2888147061037005
    


```python
plt.plot(y_test, y_pred, '.')
# plot a line
x = np.linspace(9, 15, 50)
y = x
plt.plot(x, y);
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_117_0.png)
    



```python
params = np.append(lr.intercept_, lr.coef_)
params
```




    array([ 0.35795975,  0.00684111,  0.00239845,  0.06792728,  0.04411304,
           -0.33905494, -0.17960855, -0.24822985, -0.08106516,  0.14785642,
            0.50224208,  0.00123396,  0.29875615,  0.44772303, -0.44009229])




```python
new_X = np.append(np.ones((len(X_test), 1)), X_test, axis=1)
new_X[0]
```




    array([1.00000000e+00, 5.50000000e+01, 1.95000000e+03, 9.00000000e+00,
           2.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 1.00000000e+00, 7.31366473e+00, 2.46752913e+02,
           1.09685258e+01, 7.01121399e+00, 9.64549372e+00])




```python
MSE = (sum((y_test - y_pred)**2)) / (len(new_X) - len(new_X[0]))
MSE
```




    0.28953434407571965




```python
v_b = MSE*(np.linalg.inv(np.dot(new_X.T, new_X)).diagonal())
v_b
```




    array([2.05849036e+00, 3.25093365e-07, 8.29152794e-08, 2.68055427e-05,
           6.20237000e-04, 6.82132546e-04, 6.50615557e-04, 5.45439704e-04,
           3.05842417e-04, 2.27099509e-03, 2.26989324e-04, 4.71487291e-09,
           4.45108265e-03, 1.90402779e-03, 1.53447322e-02])




```python
se = np.sqrt(v_b)
se
```




    array([1.43474400e+00, 5.70169593e-04, 2.87950134e-04, 5.17740695e-03,
           2.49045578e-02, 2.61176673e-02, 2.55071668e-02, 2.33546506e-02,
           1.74883509e-02, 4.76549587e-02, 1.50661649e-02, 6.86649322e-05,
           6.67164347e-02, 4.36351668e-02, 1.23873856e-01])




```python
t_b = params/se
t_b
```




    array([  0.24949381,  11.99837366,   8.32938418,  13.1199431 ,
             1.77128381, -12.98182317,  -7.04149338, -10.6287118 ,
            -4.63538035,   3.10264492,  33.33576166,  17.97080166,
             4.47799939,  10.26060092,  -3.55274553])




```python
p_val = [2*(1 - stats.t.cdf(np.abs(i), (len(new_X) - len(new_X[0])))) for i in t_b]
p_val = np.round(p_val, 3)
p_val
```




    array([0.803, 0.   , 0.   , 0.   , 0.077, 0.   , 0.   , 0.   , 0.   ,
           0.002, 0.   , 0.   , 0.   , 0.   , 0.   ])



### 2011 Explanatory Variables and 2013 Market Value Variable

- Data from 2011 and 2013 will be merged;
- The Variable `VALUE` (market value of housing units) from 2013 is the response variable;
- Explanatory variables and the market value variable (`VALUE`) from 2011 will be used to build the regression model. This way, the market value from 2011 will contribute to predict the market value in 2013;
- Data will be split - 1,000 observations will be held for testing and evaluating the model;


```python
# Reread the datasets
data_2011 = pd.read_csv("thads2011.txt", sep=",", usecols=usecols)
data_2013 = pd.read_csv("thads2013.txt", sep=",", usecols=usecols)
```


```python
df_merged_2011_2013 = data_2011.merge(data_2013[["CONTROL", "VALUE"]], on = "CONTROL")
df_merged_2011_2013.shape
```




    (46381, 28)




```python
# Rename VALUE_x and VALUE_y according to the year
df_merged_2011_2013.rename(columns={"VALUE_x": "VALUE_2011", "VALUE_y": "VALUE_2013"}, inplace=True)
```


```python
# Keep VALUE_2011 and VALUE_2013 that are $1000 or more
df_merged_2011_2013_sub = df_merged_2011_2013[(df_merged_2011_2013["VALUE_2011"]>= 1000) & (df_merged_2011_2013["VALUE_2013"]>= 1000)]
df_merged_2011_2013_sub.shape
```




    (24657, 28)




```python
# Owned single family housing units 
owned_single_family_units = df_merged_2011_2013_sub[(df_merged_2011_2013_sub.TYPE==1) & (df_merged_2011_2013_sub.STRUCTURETYPE==1) & (df_merged_2011_2013_sub.OWNRENT=="'1'")]  
owned_single_family_units.shape
```




    (22232, 28)




```python
owned_single_family_units_df1 = owned_single_family_units.drop(columns=["CONTROL", "TYPE", "STRUCTURETYPE", "COST06", "COST12", 
                                            "COST08", "COSTMED", "OWNRENT", "NUNITS", "ASSISTED", "STATUS", "ZSMHC"])
owned_single_family_units_df1.columns
```




    Index(['AGE1', 'METRO3', 'REGION', 'LMED', 'FMR', 'IPOV', 'PER', 'ZINC2',
           'ZADEQ', 'BEDRMS', 'BUILT', 'VALUE_2011', 'ROOMS', 'UTILITY',
           'OTHERCOST', 'VALUE_2013'],
          dtype='object')




```python
# Keep only positive values fore the variables "AGE1", "IPOV", etc.
owned_single_family_units_df2 = owned_single_family_units_df1[(
    owned_single_family_units_df1.AGE1>0) & (
    owned_single_family_units_df1.IPOV>0) & (
    owned_single_family_units_df1.PER>0) & (
    owned_single_family_units_df1.ZINC2>0) & (
    owned_single_family_units_df1.UTILITY>0) & (
    owned_single_family_units_df1.OTHERCOST>0)
].reset_index(drop=True)
owned_single_family_units_df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE1</th>
      <th>METRO3</th>
      <th>REGION</th>
      <th>LMED</th>
      <th>FMR</th>
      <th>IPOV</th>
      <th>PER</th>
      <th>ZINC2</th>
      <th>ZADEQ</th>
      <th>BEDRMS</th>
      <th>BUILT</th>
      <th>VALUE_2011</th>
      <th>ROOMS</th>
      <th>UTILITY</th>
      <th>OTHERCOST</th>
      <th>VALUE_2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>'5'</td>
      <td>'3'</td>
      <td>55770</td>
      <td>1003</td>
      <td>11572</td>
      <td>1</td>
      <td>44982</td>
      <td>'1'</td>
      <td>4</td>
      <td>1980</td>
      <td>125000</td>
      <td>8</td>
      <td>220.500000</td>
      <td>41.666667</td>
      <td>130000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65</td>
      <td>'5'</td>
      <td>'3'</td>
      <td>55770</td>
      <td>895</td>
      <td>13403</td>
      <td>2</td>
      <td>36781</td>
      <td>'1'</td>
      <td>3</td>
      <td>1985</td>
      <td>250000</td>
      <td>5</td>
      <td>230.000000</td>
      <td>40.333333</td>
      <td>200000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48</td>
      <td>'1'</td>
      <td>'3'</td>
      <td>62084</td>
      <td>935</td>
      <td>17849</td>
      <td>3</td>
      <td>57446</td>
      <td>'1'</td>
      <td>3</td>
      <td>1985</td>
      <td>169000</td>
      <td>6</td>
      <td>236.333333</td>
      <td>108.333333</td>
      <td>260000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>'5'</td>
      <td>'4'</td>
      <td>53995</td>
      <td>1224</td>
      <td>14895</td>
      <td>2</td>
      <td>129964</td>
      <td>'1'</td>
      <td>3</td>
      <td>1985</td>
      <td>140000</td>
      <td>7</td>
      <td>217.666667</td>
      <td>55.000000</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>'5'</td>
      <td>'3'</td>
      <td>55770</td>
      <td>895</td>
      <td>14849</td>
      <td>2</td>
      <td>258664</td>
      <td>'1'</td>
      <td>3</td>
      <td>1985</td>
      <td>225000</td>
      <td>5</td>
      <td>187.000000</td>
      <td>45.833333</td>
      <td>230000</td>
    </tr>
  </tbody>
</table>
</div>




```python
owned_single_family_units_df2.shape
```




    (20821, 16)




```python
# Log-transform a number of explanatory variables and the response variable "VALUE_2013"
variables = owned_single_family_units_df2[["LMED", "ZINC2", "FMR", "UTILITY", "OTHERCOST","VALUE_2011", "VALUE_2013"]]
features_transformed = np.log(variables)
features_transformed.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LMED</th>
      <th>ZINC2</th>
      <th>FMR</th>
      <th>UTILITY</th>
      <th>OTHERCOST</th>
      <th>VALUE_2011</th>
      <th>VALUE_2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20816</th>
      <td>11.138348</td>
      <td>11.512385</td>
      <td>7.029088</td>
      <td>5.468763</td>
      <td>3.624341</td>
      <td>11.918391</td>
      <td>12.206073</td>
    </tr>
    <tr>
      <th>20817</th>
      <td>11.018629</td>
      <td>9.862666</td>
      <td>6.698268</td>
      <td>5.335934</td>
      <td>4.335110</td>
      <td>12.388394</td>
      <td>12.429216</td>
    </tr>
    <tr>
      <th>20818</th>
      <td>11.018629</td>
      <td>10.735222</td>
      <td>7.074117</td>
      <td>5.542243</td>
      <td>3.555348</td>
      <td>11.849398</td>
      <td>11.918391</td>
    </tr>
    <tr>
      <th>20819</th>
      <td>11.018629</td>
      <td>10.125190</td>
      <td>7.255591</td>
      <td>5.519793</td>
      <td>3.314186</td>
      <td>11.608236</td>
      <td>12.301383</td>
    </tr>
    <tr>
      <th>20820</th>
      <td>11.018629</td>
      <td>9.902587</td>
      <td>7.074117</td>
      <td>6.080696</td>
      <td>3.912023</td>
      <td>11.849398</td>
      <td>12.388394</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop the above columns from the owned_single_family_units_df2 dataframe then concatenate the above transformed dataframe to it
cols = ["LMED", "ZINC2", "FMR", "IPOV", "UTILITY", "OTHERCOST", "VALUE_2011", "VALUE_2013"]
owned_single_family_units_df3 = owned_single_family_units_df2.drop(columns = cols)
owned_single_family_units_df4 = pd.concat([owned_single_family_units_df3, features_transformed], axis=1)
owned_single_family_units_df4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE1</th>
      <th>METRO3</th>
      <th>REGION</th>
      <th>PER</th>
      <th>ZADEQ</th>
      <th>BEDRMS</th>
      <th>BUILT</th>
      <th>ROOMS</th>
      <th>LMED</th>
      <th>ZINC2</th>
      <th>FMR</th>
      <th>UTILITY</th>
      <th>OTHERCOST</th>
      <th>VALUE_2011</th>
      <th>VALUE_2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>'5'</td>
      <td>'3'</td>
      <td>1</td>
      <td>'1'</td>
      <td>4</td>
      <td>1980</td>
      <td>8</td>
      <td>10.928991</td>
      <td>10.714018</td>
      <td>6.910751</td>
      <td>5.395898</td>
      <td>3.729701</td>
      <td>11.736069</td>
      <td>11.775290</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65</td>
      <td>'5'</td>
      <td>'3'</td>
      <td>2</td>
      <td>'1'</td>
      <td>3</td>
      <td>1985</td>
      <td>5</td>
      <td>10.928991</td>
      <td>10.512737</td>
      <td>6.796824</td>
      <td>5.438079</td>
      <td>3.697178</td>
      <td>12.429216</td>
      <td>12.206073</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48</td>
      <td>'1'</td>
      <td>'3'</td>
      <td>3</td>
      <td>'1'</td>
      <td>3</td>
      <td>1985</td>
      <td>6</td>
      <td>11.036244</td>
      <td>10.958601</td>
      <td>6.840547</td>
      <td>5.465243</td>
      <td>4.685213</td>
      <td>12.037654</td>
      <td>12.468437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>'5'</td>
      <td>'4'</td>
      <td>2</td>
      <td>'1'</td>
      <td>3</td>
      <td>1985</td>
      <td>7</td>
      <td>10.896647</td>
      <td>11.775013</td>
      <td>7.109879</td>
      <td>5.382965</td>
      <td>4.007333</td>
      <td>11.849398</td>
      <td>12.043554</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>'5'</td>
      <td>'3'</td>
      <td>2</td>
      <td>'1'</td>
      <td>3</td>
      <td>1985</td>
      <td>5</td>
      <td>10.928991</td>
      <td>12.463285</td>
      <td>6.796824</td>
      <td>5.231109</td>
      <td>3.825012</td>
      <td>12.323856</td>
      <td>12.345835</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replace codes with the corresponding value
owned_single_family_units_df4.replace({"REGION":{"'1'": "Northeast", "'2'": "Midwest", 
                                                 "'3'": "South", "'4'": "West"}}, inplace = True)
owned_single_family_units_df4["METRO3"] = np.where(owned_single_family_units_df4.METRO3 == "'1'",
                                                   "Metro", "Other")
owned_single_family_units_df4.replace({"ZADEQ":{"'1'": "Adequate", "'2'": "Moderately_inadequate",
                    "'3'": "Severely_inadequate", "'-6'": "Vacant_No_Info"}}, inplace = True)
owned_single_family_units_df4.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE1</th>
      <th>METRO3</th>
      <th>REGION</th>
      <th>PER</th>
      <th>ZADEQ</th>
      <th>BEDRMS</th>
      <th>BUILT</th>
      <th>ROOMS</th>
      <th>LMED</th>
      <th>ZINC2</th>
      <th>FMR</th>
      <th>UTILITY</th>
      <th>OTHERCOST</th>
      <th>VALUE_2011</th>
      <th>VALUE_2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20816</th>
      <td>42</td>
      <td>Other</td>
      <td>Northeast</td>
      <td>4</td>
      <td>Adequate</td>
      <td>3</td>
      <td>1970</td>
      <td>4</td>
      <td>11.138348</td>
      <td>11.512385</td>
      <td>7.029088</td>
      <td>5.468763</td>
      <td>3.624341</td>
      <td>11.918391</td>
      <td>12.206073</td>
    </tr>
    <tr>
      <th>20817</th>
      <td>72</td>
      <td>Metro</td>
      <td>West</td>
      <td>1</td>
      <td>Adequate</td>
      <td>2</td>
      <td>1985</td>
      <td>6</td>
      <td>11.018629</td>
      <td>9.862666</td>
      <td>6.698268</td>
      <td>5.335934</td>
      <td>4.335110</td>
      <td>12.388394</td>
      <td>12.429216</td>
    </tr>
    <tr>
      <th>20818</th>
      <td>55</td>
      <td>Metro</td>
      <td>West</td>
      <td>5</td>
      <td>Adequate</td>
      <td>3</td>
      <td>1960</td>
      <td>8</td>
      <td>11.018629</td>
      <td>10.735222</td>
      <td>7.074117</td>
      <td>5.542243</td>
      <td>3.555348</td>
      <td>11.849398</td>
      <td>11.918391</td>
    </tr>
    <tr>
      <th>20819</th>
      <td>26</td>
      <td>Metro</td>
      <td>West</td>
      <td>3</td>
      <td>Adequate</td>
      <td>4</td>
      <td>2008</td>
      <td>6</td>
      <td>11.018629</td>
      <td>10.125190</td>
      <td>7.255591</td>
      <td>5.519793</td>
      <td>3.314186</td>
      <td>11.608236</td>
      <td>12.301383</td>
    </tr>
    <tr>
      <th>20820</th>
      <td>48</td>
      <td>Metro</td>
      <td>West</td>
      <td>1</td>
      <td>Adequate</td>
      <td>3</td>
      <td>1950</td>
      <td>5</td>
      <td>11.018629</td>
      <td>9.902587</td>
      <td>7.074117</td>
      <td>6.080696</td>
      <td>3.912023</td>
      <td>11.849398</td>
      <td>12.388394</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create the dummy variables for the categorical variables
region = pd.get_dummies(owned_single_family_units_df4['REGION'], dtype=int)
metro = pd.get_dummies(owned_single_family_units_df4['METRO3'], dtype=int)
adequacy = pd.get_dummies(owned_single_family_units_df4['ZADEQ'], dtype=int)
# Concatenate all the dummy variables into one dataframe
dummies_df = pd.concat([region, metro, adequacy], axis=1)
dummies_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Midwest</th>
      <th>Northeast</th>
      <th>South</th>
      <th>West</th>
      <th>Metro</th>
      <th>Other</th>
      <th>Adequate</th>
      <th>Moderately_inadequate</th>
      <th>Severely_inadequate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_mat = owned_single_family_units_df4.drop(columns=["VALUE_2013", "METRO3", "REGION", "ZADEQ"])
X_mat = pd.concat([X_mat, dummies_df[['Midwest', 'Northeast', 'South', 'Metro', 'Adequate']]], axis=1)
```


```python
# lower triangular matrix
mask2 = np.triu(np.ones_like(X_mat.corr()))
warnings.filterwarnings("ignore")
# plotting a triangle correlation heatmap
sns.heatmap(X_mat.corr(), cmap="YlGnBu", annot=True, fmt='0.1f', mask=mask2)
plt.savefig("corrMatrix.png");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_139_0.png)
    



```python
y_resp = owned_single_family_units_df4["VALUE_2013"]
```


```python
test_size = round(1000 / owned_single_family_units_df4.shape[0],4)
test_size
```




    0.048




```python
# Separate the data
X2_train, X2_test, y2_train, y2_test = train_test_split(X_mat, y_resp, test_size = test_size, random_state=42)
print("The number of samples in train set is {}".format(X2_train.shape[0]))
print("The number of samples in test set is {}".format(X2_test.shape[0]))
```

    The number of samples in train set is 19821
    The number of samples in test set is 1000
    


```python
X2_train.to_csv("training_inputs.csv")
y2_train.to_csv("training_response.csv")
X2_test.to_csv("test_inputs.csv")
y2_test.to_csv("test_response.csv")
```


```python
descriptive_stats = pd.concat([y2_train.describe(), X2_train.describe()], axis=1)
descriptive_statistics = descriptive_stats.rename(columns={"VALUE_2013": "ln_value_2013"})
descriptive_statistics.to_csv("descriptive_stats.csv")
```


```python
# (1) Initiate the model
lr2 = LinearRegression()
# (2) Fit the model
lr2.fit(X2_train, y2_train)
# (3) Score the model
round(lr2.score(X2_test, y2_test),2)
```




    0.61




```python
params2 = np.append(lr2.intercept_, lr2.coef_)
params2
```




    array([-4.15115916e+00,  8.47991382e-04, -8.45210427e-03, -3.93258886e-02,
            1.91488892e-03,  4.77613216e-02,  1.59614781e-01,  5.29324517e-02,
            3.69681281e-01,  5.34945627e-02,  7.14752260e-03,  5.83990141e-01,
           -9.61762889e-02, -7.98542103e-02, -1.07717874e-01, -4.43225115e-02,
            2.29460839e-02])




```python
# Predict
y2_pred = lr2.predict(X2_test)
# Mean Square error
mse2 = metrics.mean_squared_error(y2_test, y2_pred)
print("Mean Squared Error {}".format(mse2))
```

    Mean Squared Error 0.22820281510477217
    


```python
pd.DataFrame(y2_pred).to_csv("predicted.csv")
```


```python
# Mean Absolute Difference
round((np.abs(np.exp(y2_test) - np.exp(y2_pred))).mean())
```




    72729




```python
new_X2 = np.append(np.ones((len(X2_test), 1)), X2_test, axis=1)
new_X2[0]
```




    array([1.00000000e+00, 3.90000000e+01, 5.00000000e+00, 4.00000000e+00,
           1.97000000e+03, 7.00000000e+00, 1.08573245e+01, 1.08546820e+01,
           6.85856503e+00, 6.12249281e+00, 4.60517019e+00, 1.14075649e+01,
           0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
           1.00000000e+00])




```python
# Varriance
v_b2 = mse2*(np.linalg.inv(np.dot(new_X2.T, new_X2)).diagonal())
# Standard error
se2 = np.sqrt(v_b2)
# t-statistics
t_b2 = params2/se2
# p-values
p_val2 = [2*(1 - stats.t.cdf(np.abs(i), (len(new_X2) - len(new_X2[0])))) for i in t_b2]
p_val2 = np.round(p_val2, 3)
p_val2
```




    array([0.02 , 0.449, 0.508, 0.22 , 0.002, 0.002, 0.275, 0.002, 0.   ,
           0.189, 0.732, 0.   , 0.117, 0.168, 0.032, 0.251, 0.812])




```python
warnings.filterwarnings("ignore")
sns.displot(data=owned_single_family_units_df2, x='VALUE_2013', height=3, aspect=1.4, kde=False)
plt.xlabel("Current Market Values ($)")
plt.ylabel("Number of Housing Units")
plt.title("Distribution of Current Market Value")
plt.savefig("value2013.png")
sns.displot(data=owned_single_family_units_df4, x='VALUE_2013', height=3, aspect=1.4, kde=False)
plt.title("Distribution of Logarithmic Market Values")
plt.xlabel("Logarithmic of Current Market Values (Log($))")
plt.ylabel("Number of Housing Units")
plt.savefig("logvalue2013.png");
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_152_0.png)
    



    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_152_1.png)
    



```python
# Let's first rename the features that were logarithmically transformed
X2_train.rename(columns={"VALUE_2011": "ln_value_2011", "UTILITY": "ln_utility", "OTHERCOST": "ln_othercost",
                         "LMED": "ln_lmed", "FMR": "ln_fmr", "ZINC2": "ln_zinc2"}, inplace = True)

coefs = pd.DataFrame(lr2.coef_, columns=["Coefficients"], index=X2_train.columns)

coefs_sorted = coefs.sort_values(by="Coefficients", key=abs, ascending=True)

coefs_sorted.plot(kind="barh", figsize=(9, 6))
plt.title("Multilinear Regression model - Feature Importance")
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
plt.savefig("featureImportance.png")
```


    
![png](marketValuesforHousingUnits_files/marketValuesforHousingUnits_153_0.png)
    



```python
# Feature importance
coefs.sort_values(by="Coefficients", key=abs, ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ln_value_2011</th>
      <td>0.583990</td>
    </tr>
    <tr>
      <th>ln_fmr</th>
      <td>0.369681</td>
    </tr>
    <tr>
      <th>ln_lmed</th>
      <td>0.159615</td>
    </tr>
    <tr>
      <th>South</th>
      <td>-0.107718</td>
    </tr>
    <tr>
      <th>Midwest</th>
      <td>-0.096176</td>
    </tr>
    <tr>
      <th>Northeast</th>
      <td>-0.079854</td>
    </tr>
    <tr>
      <th>ln_utility</th>
      <td>0.053495</td>
    </tr>
    <tr>
      <th>ln_zinc2</th>
      <td>0.052932</td>
    </tr>
    <tr>
      <th>ROOMS</th>
      <td>0.047761</td>
    </tr>
    <tr>
      <th>Metro</th>
      <td>-0.044323</td>
    </tr>
    <tr>
      <th>BEDRMS</th>
      <td>-0.039326</td>
    </tr>
    <tr>
      <th>Adequate</th>
      <td>0.022946</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>-0.008452</td>
    </tr>
    <tr>
      <th>ln_othercost</th>
      <td>0.007148</td>
    </tr>
    <tr>
      <th>BUILT</th>
      <td>0.001915</td>
    </tr>
    <tr>
      <th>AGE1</th>
      <td>0.000848</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
