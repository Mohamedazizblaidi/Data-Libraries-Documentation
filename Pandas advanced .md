
###  **Applying Functions**

- `df.apply(function)`: Applies a function along an axis of the DataFrame.
- `df.applymap(function)`: Applies a function elementwise.

```python
df['Age_squared'] = df['Age'].apply(lambda x: x**2)
df = df.applymap(lambda x: x * 2)

```

### **Date and Time**

- `pd.to_datetime(df['column_name'])`: Converts a column to datetime.
- `df['column_name'].dt.year`: Extracts the year from datetime.


```python
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
```

### **`to_datetime()`**

The `to_datetime()` method is used to convert a column or Series to a datetime object, which is useful for time-series analysis.


```python
import pandas as pd
data = {'Date': ['2023-08-21', '2024-01-15', '2024-05-05']}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
print(df)
```

**Key Parameters:**

- `format`: Specify the format of the date string if it's non-standard.
- `errors`: Handle parsing errors (`'coerce'`, `'ignore'`, `'raise'`).

### **Pivot Tables**

- `pd.pivot_table(df, values, index, columns, aggfunc)`: Creates a pivot table.

```python
pivot_table = pd.pivot_table(df, values='value', index='index_col',  columns='columns_col', aggfunc='mean')
```

### **Categorical Data**

- `df['column_name'] = df['column_name'].astype('category')`: Converts a column to a categorical type.
- `df['column_name'].cat.codes`: Gets codes for categories.

```python
df['Category'] = df['Category'].astype('category')
category_codes = df['Category'].cat.codes
```

### **Window Functions**

- `df.rolling(window).mean()`: Computes a rolling mean.
- `df.expanding().sum()`: Computes an expanding sum.
- `df.ewm(span).mean()`: Exponentially weighted moving average.

```python
rolling_mean = df['column'].rolling(window=3).mean()
expanding_sum = df['column'].expanding().sum()
ewm_mean = df['column'].ewm(span=5).mean()
```

### **Rankings**

- `df['rank'] = df['column'].rank()`: Assigns ranks to entries.

```python
df['rank'] = df['column'].rank()
```

### **`corr()`**

The `corr()` method computes pairwise correlation of columns in a DataFrame, excluding NA/null values. It’s commonly used in statistical analysis to understand the relationship between variables.

```python
import pandas as pd
data = {'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [10, 20, 30, 40]}
df = pd.DataFrame(data)
correlation_matrix = df.corr()
print(correlation_matrix)
```

**Key Parameters:**

- `method`: Method of correlation (`'pearson'`, `'kendall'`, `'spearman'`).


### **Plotting**

- `df.plot()`: Plots the data using Matplotlib.
- `df.plot(kind='bar')`: Creates a bar plot.

```python
df.plot(kind='bar')
```

### **`plot()`**

The `plot()` method in Pandas is a convenient way to visualize data. It provides various types of plots, including line plots, bar plots, histograms, and more.

**Basic Line Plot Example:**

```python
import pandas as pd
data = {'Year': [2020, 2021, 2022], 'Sales': [100, 150, 200]}
df = pd.DataFrame(data)
# Creating a line plot
df.plot(x='Year', y='Sales', kind='line')
```

### **`Scatter Plot`**

A scatter plot is used to visualize the relationship between two variables.

**Example:**

```python
import pandas as pd
data = {'Height': [5.1, 5.5, 6.0, 5.8], 'Weight': [65, 70, 75, 68]}
df = pd.DataFrame(data)
# Creating a scatter plot
df.plot.scatter(x='Height', y='Weight')
```

### **`Histogram`**

A histogram is used to represent the distribution of a dataset

**Example:**

```python
import pandas as pd
data = {'Age': [25, 30, 35, 40, 30, 35, 40, 45, 50]}
df = pd.DataFrame(data)
# Creating a histogram
df['Age'].plot(kind='hist', bins=5)
```

## 📊 Index and Column Operations

### `set_index()`

Manipulates the DataFrame index by setting one or more columns as the index.

```python
# Set a single column as index
df.set_index('column_name')

# Set multiple columns as index (MultiIndex)
df.set_index(['col1', 'col2'])

# Keep the original column after setting as index
df.set_index('column_name', drop=False)
```

### `join()`

Joins DataFrames based on their index (left join by default).

```python
# Join two DataFrames on index
df1.join(df2)

# Join with different join types
df1.join(df2, how='inner')  # inner, outer, left, right
```

### `.columns` and `.index`

Access DataFrame structure information.

```python
# Get column names
df.columns

# Get index values
df.index

# Get underlying data as numpy array
df.values
```

## 🔍 Data Selection and Filtering

### `.loc` vs `.iloc` - **Key Difference**

- **`.loc`**: Label-based indexing (uses column names and index labels)
- **`.iloc`**: Integer position-based indexing (uses numeric positions)

```python
# .loc - uses labels
df.loc[0, 'column_name']        # Single value
df.loc[[0, 1], ['col1', 'col2']] # Multiple rows/columns

# .iloc - uses positions
df.iloc[0, 1]                   # Single value by position
df.iloc[[0, 1], [0, 2]]         # Multiple rows/columns by position
```

### When to use `.` vs `[]`

- **Use `.`** for single column access: `df.column_name`
- **Use `[]`** for multiple columns, complex selection, or when column names have spaces: `df['column name']` or `df[['col1', 'col2']]`

## 📍 Data Access Examples

### Selecting Data

```python
# Single column
desc = reviews.description
desc = reviews['description']

# First value of a column
first_description = reviews.description[0]

# First row of DataFrame
first_row = reviews.iloc[0]

# First 10 values of a column
first_descriptions = reviews['description'].head(10)

# Specific rows by index
sample_reviews = reviews.loc[[1, 2, 3, 5, 8]]

# Multiple rows and columns
df = reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]

# Using variables for selection
cols = ['country', 'variety']
indices = range(0, 100)
df = reviews.loc[indices, cols]
```

### `range()` Function

Creates a sequence of numbers.

```python
range(0, 100)     # 0 to 99
range(10)         # 0 to 9
range(5, 20, 2)   # 5, 7, 9, 11, 13, 15, 17, 19
```

## 🎯 Filtering and Conditions

### Boolean Indexing

```python
# Simple filter
italian_wines = reviews[reviews.country == 'Italy']

# Complex filter with multiple conditions
top_oceania_wines = reviews.loc[
    ((reviews.country == 'Australia') | (reviews.country == 'New Zealand')) & 
    (reviews.points >= 95)
]
```

### `isin()` Method

Check if values are in a list.

```python
# Filter rows where country is in the list
countries_of_interest = ['Italy', 'France', 'Spain']
european_wines = reviews[reviews.country.isin(countries_of_interest)]
```

### Logical Operators in Pandas

- **`&`**: AND operator (use instead of `and`)
- **`|`**: OR operator (use instead of `or`)
- **`~`**: NOT operator

```python
# Correct pandas syntax
reviews[(reviews.country == 'Italy') & (reviews.points >= 90)]

# Wrong - don't use 'and'/'or' in pandas boolean indexing
# reviews[(reviews.country == 'Italy') and (reviews.points >= 90)]  # ❌
```

## 🔍 Missing Data Detection

### `isna()` / `isnull()` - **Same Function**

Detect missing values (NaN).

```python
df.isna()          # Returns boolean DataFrame
df.isnull()        # Identical to isna()
df.column.isna()   # Check specific column
```

### `notnull()` / `notna()`

Detect non-missing values.

```python
df.notnull()       # Returns boolean DataFrame
df.notna()         # Identical to notnull()
```

## 📊 Data Analysis Functions

### `describe()`

Generate descriptive statistics.

```python
df.describe()              # Numerical columns only
df.describe(include='all') # All columns
```

### `mean()`

Calculate average values.

```python
df.mean()           # Mean of all numerical columns
df.column.mean()    # Mean of specific column
```

### `median()`

Calculate median values.

```python
df.median()         # Median of all numerical columns
df.column.median()  # Median of specific column
```

### `unique()`

Get unique values in a column.

```python
df.column.unique()  # Returns array of unique values
```

### `value_counts()`

Count occurrences of each unique value.

```python
df.column.value_counts()  # Returns Series with counts
```

### `idxmax()`

Find index of maximum value.

```python
df.column.idxmax()  # Index of maximum value
df.idxmax()         # Index of max value for each column
```

## 🔄 Data Transformation

### `map()` Function

Transform values using a mapping or function.

```python
# Using dictionary mapping
mapping = {'A': 1, 'B': 2, 'C': 3}
df.column.map(mapping)

# Using function
df.column.map(lambda x: x.upper())
```

### `lambda` Functions

Anonymous functions for quick transformations.

```python
# Lambda examples
df.column.map(lambda x: x * 2)
df.apply(lambda row: row['col1'] + row['col2'], axis=1)
```

### `apply()` Function

Apply function along axis of DataFrame.

```python
# Apply to columns (axis=0, default)
df.apply(lambda x: x.max() - x.min())

# Apply to rows (axis=1)
df.apply(lambda row: row['col1'] + row['col2'], axis=1)
```

## 📋 Data Sorting

### `sort_values()`

Sort DataFrame by values.

```python
# Sort by single column
df.sort_values('column_name')

# Sort by multiple columns
df.sort_values(['col1', 'col2'])

# Sort in descending order
df.sort_values('column_name', ascending=False)
```

## 🔧 Syntax Rules: Dot vs Parentheses

### When to use `.` (dot notation)

- **Accessing attributes**: `df.columns`, `df.index`, `df.values`
- **Accessing columns**: `df.column_name` (only if no spaces in name)

### When to use `()` (parentheses)

- **Calling methods/functions**: `df.head()`, `df.describe()`, `df.mean()`
- **All operations that perform actions or calculations**

```python
# Attributes (no parentheses)
df.columns
df.index
df.values

# Methods (with parentheses)
df.head()
df.describe()
df.mean()
df.sort_values('column')
```

## 💡 Important Notes

### DataFrame vs Series Return Types

```python
# Returns Series (1D)
reviews.loc[100, ['country', 'points']]

# Returns DataFrame (2D) - note the double brackets
reviews.loc[[100], ['country', 'points']]
```

### Keyboard Symbols Reference

- `|` - Pipe symbol (OR operator)
- `&` - Ampersand (AND operator)
- `~` - Tilde (NOT operator)
- `[]` - Square brackets (indexing)
- `()` - Parentheses (method calls)
- `.` - Dot (attribute access/method chaining)

# Pandas Advanced Operations Reference

## 📄 Reading Files

### HTML Reading

```python
# Read HTML tables from webpage
df = pd.read_html('https://example.com/table.html')  # Returns list of DataFrames
df = pd.read_html('https://example.com/table.html')[0]  # Get first table

# Read HTML from local file
df = pd.read_html('file.html')

# Read HTML with specific table attributes
df = pd.read_html('url', attrs={'class': 'data-table'})
df = pd.read_html('url', attrs={'id': 'main-table'})

# HTML reading parameters
df = pd.read_html('url', 
                  header=0,           # Row to use as column names
                  skiprows=1,         # Skip first row
                  index_col=0)        # Use first column as index
```

### Common File Reading Functions

```python
# CSV files
df = pd.read_csv('file.csv')

# Excel files
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# JSON files
df = pd.read_json('file.json')

# Text files with custom delimiter
df = pd.read_csv('file.txt', delimiter='\t')  # Tab-separated

# Reading with specific encoding
df = pd.read_csv('file.csv', encoding='utf-8')
```

## 🗑️ Deleting Columns with `del`

### Using `del` Statement

```python
# Delete single column
del df['column_name']

# Delete multiple columns (use drop() instead)
# del df[['col1', 'col2']]  # This won't work!

# Alternative methods for deleting columns
df.drop('column_name', axis=1, inplace=True)  # Preferred method
df.drop(['col1', 'col2'], axis=1, inplace=True)  # Multiple columns
```

### `del` vs `drop()` Comparison

```python
# Using del (permanent, in-place)
del df['column_name']

# Using drop() (more flexible)
df = df.drop('column_name', axis=1)           # Returns new DataFrame
df.drop('column_name', axis=1, inplace=True)  # Modifies original
```

## 🔄 Apply Function

### Basic Apply Usage

```python
# Apply function to columns (axis=0, default)
df.apply(lambda x: x.max() - x.min())

# Apply function to rows (axis=1)
df.apply(lambda row: row['col1'] + row['col2'], axis=1)

# Apply to specific column
df['column'].apply(lambda x: x.upper())
```

### Apply with Custom Functions

```python
# Define custom function
def calculate_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    else:
        return 'F'

# Apply custom function
df['grade'] = df['score'].apply(calculate_grade)

# Apply function with multiple columns
def full_name(row):
    return f"{row['first_name']} {row['last_name']}"

df['full_name'] = df.apply(full_name, axis=1)
```

### Apply vs Map vs Applymap

```python
# apply() - Works on Series or DataFrame
df['column'].apply(function)  # Series
df.apply(function)           # DataFrame

# map() - Only works on Series
df['column'].map(function)   # Series only

# applymap() - Element-wise on entire DataFrame
df.applymap(function)        # Every element
```

## 📊 Sort Values

### Basic Sorting

```python
# Sort by single column
df.sort_values('column_name')
df.sort_values('column_name', ascending=False)  # Descending

# Sort by multiple columns
df.sort_values(['col1', 'col2'])
df.sort_values(['col1', 'col2'], ascending=[True, False])
```

### Advanced Sorting Options

```python
# Sort with custom key
df.sort_values('column', key=lambda x: x.str.lower())

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# Sort with missing values
df.sort_values('column', na_position='first')   # NaN first
df.sort_values('column', na_position='last')    # NaN last (default)
```

## 💾 Inplace Parameter

### `inplace=True` - Memory Modification

```python
# Without inplace (creates new object)
df_new = df.sort_values('column')
df_new = df.drop('column', axis=1)
df_new = df.fillna(0)

# With inplace=True (modifies original in memory)
df.sort_values('column', inplace=True)
df.drop('column', axis=1, inplace=True)
df.fillna(0, inplace=True)
```

### When to Use `inplace=True`

```python
# Memory efficient for large datasets
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df.set_index('column', inplace=True)

# Be careful - inplace=True cannot be undone!
# Good practice: make a copy first
df_backup = df.copy()
df.some_operation(inplace=True)
```

## 🔗 Concatenation with `pd.concat()`

### Basic Concatenation

```python
# Vertical concatenation (stack rows)
result = pd.concat([df1, df2])                    # Default axis=0
result = pd.concat([df1, df2], axis=0)            # Explicit vertical

# Horizontal concatenation (side by side)
result = pd.concat([df1, df2], axis=1)            # Horizontal
```

### Advanced Concatenation Options

```python
# Concatenation with keys
result = pd.concat([df1, df2], keys=['Dataset1', 'Dataset2'])

# Ignore index (reset numbering)
result = pd.concat([df1, df2], ignore_index=True)

# Inner join (only matching columns)
result = pd.concat([df1, df2], join='inner')

# Outer join (all columns, default)
result = pd.concat([df1, df2], join='outer')
```

### Concatenation Examples

```python
# Multiple DataFrames
dfs = [df1, df2, df3, df4]
result = pd.concat(dfs, ignore_index=True)

# With custom column names
result = pd.concat([df1, df2], axis=1, keys=['Left', 'Right'])

# Concatenate specific columns
result = pd.concat([df1[['col1', 'col2']], df2[['col3', 'col4']]], axis=1)
```

## 🔄 Merge Function

### Basic Merge

```python
# Merge on common column (automatic detection)
result = pd.merge(df1, df2)

# Merge on specific column
result = pd.merge(df1, df2, on='common_column')

# Merge on multiple columns
result = pd.merge(df1, df2, on=['col1', 'col2'])
```

### Merge Types

```python
# Inner join (default) - only matching rows
result = pd.merge(df1, df2, how='inner')

# Left join - all rows from left DataFrame
result = pd.merge(df1, df2, how='left')

# Right join - all rows from right DataFrame
result = pd.merge(df1, df2, how='right')

# Outer join - all rows from both DataFrames
result = pd.merge(df1, df2, how='outer')
```

### Merge with Different Column Names

```python
# When key columns have different names
result = pd.merge(df1, df2, left_on='id', right_on='user_id')

# Multiple columns with different names
result = pd.merge(df1, df2, 
                 left_on=['id', 'date'], 
                 right_on=['user_id', 'timestamp'])
```

### Merge on Index

```python
# Merge using index
result = pd.merge(df1, df2, left_index=True, right_index=True)

# Merge column with index
result = pd.merge(df1, df2, left_on='id', right_index=True)
```

## 🔄 Merge vs Concat - Key Differences

### **Merge**

- **Purpose**: Combines DataFrames based on **common columns/keys**
- **Like**: SQL JOIN operations
- **Best for**: Relational data with common identifiers
- **Alignment**: By values in specified columns

```python
# Merge example - combining by common ID
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'id': [1, 2, 4], 'age': [25, 30, 35]})
result = pd.merge(df1, df2, on='id')  # Joins where id matches
```

### **Concat**

- **Purpose**: **Stacks** DataFrames vertically or horizontally
- **Like**: Appending or side-by-side placement
- **Best for**: Combining similar structured data
- **Alignment**: By position (index/columns)

```python
# Concat example - stacking similar data
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
result = pd.concat([df1, df2])  # Stacks vertically
```

### When to Use Each

```python
# Use MERGE when:
# - You have a common identifier (ID, key)
# - You want to combine related information
# - You need SQL-like JOIN behavior
customer_info = pd.merge(customers, orders, on='customer_id')

# Use CONCAT when:
# - You have similar structured data to stack
# - You want to append new rows or columns
# - You're combining data from same source
all_data = pd.concat([january_sales, february_sales])
```

## 📋 DataFrame Columns

### Accessing Column Information

```python
# Get column names
df.columns                    # Index object with column names

# Convert to list
column_list = df.columns.tolist()

# Get column data types
df.dtypes
df.info()                     # Comprehensive info including types
```

### Column Manipulation

```python
# Rename columns
df.columns = ['new_name1', 'new_name2', 'new_name3']

# Rename specific columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# Add prefix/suffix to all columns
df.add_prefix('prefix_')
df.add_suffix('_suffix')

# Column operations
df.columns.str.upper()        # Uppercase column names
df.columns.str.replace(' ', '_')  # Replace spaces with underscores
```

### Column Selection and Reordering

```python
# Select specific columns
df[['col1', 'col2', 'col3']]

# Reorder columns
df = df[['col2', 'col1', 'col3']]

# Select columns by pattern
df.filter(regex='^sales_')    # Columns starting with 'sales_'
df.filter(like='_2023')       # Columns containing '_2023'
```

## 💡 Quick Reference Summary

### File Operations

- `pd.read_html()` - Read HTML tables
- `pd.read_csv()` - Read CSV files
- `del df['column']` - Delete column permanently

### Data Manipulation

- `df.apply()` - Apply functions to rows/columns
- `df.sort_values()` - Sort by column values
- `inplace=True` - Modify original DataFrame in memory

### Combining DataFrames

- `pd.concat()` - Stack DataFrames (vertical/horizontal)
- `pd.merge()` - Join DataFrames on common keys
- **Merge**: SQL-like joins on keys
- **Concat**: Stacking similar data

### Column Management

- `df.columns` - Access column names
- `df.rename()` - Rename columns
- `df.filter()` - Select columns by pattern

This reference covers all the essential advanced Pandas operations for data manipulation and combination!









This reference covers the essential pandas operations for data manipulation, selection, filtering, and analysis. Keep it handy for quick lookups!
