
A **Pandas DataFrame** is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns). It's one of the most widely used data structures in data analysis and manipulation with Python.

#### **Creating a DataFrame** : 

You can create a DataFrame in several ways, including from a dictionary, list of dictionaries, list of lists, or even from another DataFrame or CSV file.

**From a Dictionary of Lists**

```python
import pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print(df)
Output:
    Name  Age         City
0  Alice   25     New York
1    Bob   30  Los Angeles
2 Charlie   35     Chicago
```

**From a List of Dictionaries**

```python
data = [
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'Los Angeles'},
    {'Name': 'Charlie', 'Age': 35, 'City': 'Chicago'}
] 
df = pd.DataFrame(data)
print(df)
```

**From a List of Lists (with Column Names)**

```python
data = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Los Angeles'],
    ['Charlie', 35, 'Chicago']
]
df = pd.DataFrame(data, columns=['Name', 'Age', 'City'])
print(df)
```

**From a CSV File**

```python
df = pd.read_csv('file.csv')  # Replace 'file.csv' with your file name
print(df)
```

#### **Accessing Data in a DataFrame**

**Accessing Columns**

```python
print(df['Name'])  # Access the 'Name' column
```

**Accessing Rows by Index**

```python
print(df.iloc[0])  # Access the first row using integer-location based indexing
```

**Accessing Rows by Label**

```python
df = df.set_index('Name')  # Set 'Name' as the index
print(df.loc['Alice'])     # Access the row for 'Alice' using label-based indexing
```

**Accessing a Subset of Rows and Columns**

```python
print(df.loc[0:1, ['Name', 'City']])  # Access the first two rows and specific columns
```

#### **Common Operations on DataFrames**

 **Filtering Data**
    
- Filtering rows based on conditions

```python
print(df[df['Age'] > 30])  # Filter rows where 'Age' is greater than 30
```

**Adding New Columns**

- Creating a new column based on existing data

```python
df['Age in 5 Years'] = df['Age'] + 5
print(df)
```

**Dropping Columns or Rows**

- Dropping a column:
```python
df = df.drop('City', axis=1)  # Axis 1 means column-wise
print(df)
```

- Dropping a row:
```python
df = df.drop(0, axis=0)  # Axis 0 means row-wise
print(df)
```

**Renaming Columns**

```python
df = df.rename(columns={'Name': 'Full Name'})
print(df)
```

**Handling Missing Data**

- Checking for missing data

```python
print(df.isnull())
```

- Filling missing values:

```python
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill missing 'Age' values with the mean
```

- Dropping missing values:

```python
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill missing 'Age' values with the mean
```

**Merging and Joining DataFrames**

- **Merging** based on a key:

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
merged_df = pd.merge(df1, df2, on='key', how='inner')
print(merged_df)
```

- **Joining** DataFrames using indices:
 
```python
df1 = df1.set_index('key')
df2 = df2.set_index('key')
joined_df = df1.join(df2, how='outer')
print(joined_df)
```

#### **Summary Statistics**
 **Basic Statistics**
    
- **`df.describe()`**: Provides a summary of statistics for numerical columns.
- **`df.mean()`**: Calculate the mean for each column.
- **`df.median()`**: Calculate the median for each column.
- **`df.std()`**: Calculate the standard deviation for each column

```python
print(df.describe())
```

**Counting Unique Values**

```python
print(df['City'].value_counts())  # Count occurrences of each unique value in 'City'
```

**Grouping Data**

```python
grouped_df = df.groupby('City').mean()  # Group by 'City' and calculate the mean for each group
print(grouped_df)
```

### **Key Points:**

- A DataFrame is a two-dimensional data structure with labeled axes (rows and columns).
- You can create DataFrames from various data structures like dictionaries, lists, and files.
- DataFrames offer a wide range of functionalities for data manipulation, analysis, and cleaning.


A **Pandas Series** is a one-dimensional labeled array capable of holding any data type (integers, strings, floating-point numbers, Python objects, etc.). It's similar to a column in a Data Frame or a single-dimensional array in NumPy but with additional capabilities like labeled indexing.

#### **Creating a Pandas Series**

You can create a Series in several ways, including from a list, dictionary, scalar value, or NumPy array.

**From a List**

import pandas as pd :

```python
data = [10, 20, 30, 40, 50]
series = pd.Series(data)
print(series)
Output:
0    10
1    20
2    30
3    40
4    50
dtype: int64
```

**From a Dictionary**

```python
data = {'a': 10, 'b': 20, 'c': 30}
series = pd.Series(data)
print(series)
Output:
a    10
b    20
c    30
dtype: int64
```

**From a Scalar Value**

```python
series = pd.Series(5, index=[0, 1, 2, 3])
print(series)
Output:
0    5
1    5
2    5
3    5
dtype: int64
```

**From a NumPy Array**

```python
import numpy as np
data = np.array([1, 2, 3, 4])
series = pd.Series(data)
print(series)
Output:
0    1
1    2
2    3
3    4
dtype: int64
```

#### **Accessing Data in a Series**

 **Using the Index**
 
 ```python
print(series[0])  # Accessing the first element
```

**Using the Label (if present)**

 ```python
print(series['a'])  # Accessing the element with label 'a'
 ```
 
**Slicing**

```python
print(series[1:3])  # Accessing a range of elements
```

#### **Common Operations on Series**

**Vectorized Operations**

You can perform operations on the entire Series, similar to how you would with NumPy arrays.

```python
series = pd.Series([1, 2, 3, 4])
print(series + 10)  # Adds 10 to each element
```


**Filtering**

Use conditional statements to filter the Series
```python
print(series[series > 2])  # Returns elements greater than 2
```

**Using Methods**


- **`series.mean()`**: Compute the mean of the Series.
- **`series.max()`**: Find the maximum value in the Series.
- **`series.min()`**: Find the minimum value in the Series.
```python
print(series.mean())  # Output: 2.5
```

#### **Handling Missing Data**

 **Checking for Missing Data**
 
- **`series.isnull()`**: Returns a Series of boolean values indicating if each element is missing.
- **`series.notnull()`**: Returns the opposite of `isnull()`.

```python
series = pd.Series([1, 2, None, 4])
print(series.isnull())
```

**Filling Missing Data**

**`series.fillna(value)`**: Fills missing values with the specified `value`.
Dropping Missing Data
**`series.dropna()`**: Returns a new Series with missing values removed.

```python
print(series.fillna(0))
```

**Customizing Index**

You can assign custom labels to the indices of a Series.

```python
data = [1, 2, 3, 4]
index = ['a', 'b', 'c', 'd']
series = pd.Series(data, index=index)
print(series)
Output:
a    1
b    2
c    3
d    4
dtype: int64
```

**🏷️ What is `name`? 

- `name` is a **label** for a Pandas Series.
- It helps identify the Series, like a column name.

**✅ Example:

```python
`import pandas as pd  s = pd.Series([1, 2, 3]) s.name = "Age" print(s)`
```

**Output:**

```python
0    1
1    2
2    3
Name: Age, dtype: int64
```

**Set it during creation: 

```python
s = pd.Series([1, 2, 3], name="Age")
```

Both methods will give the same result.

** 🔍 Get the Name:

```python
print(s.name)  # Output: Age
```

**📝 Extra Tip:

If you convert the Series to a DataFrame:

```python
df = s.to_frame()
```

The `name` becomes the column name.

**General Syntax – Creating a Named Pandas Series from a Dictionary : 

🧱 General Syntax:

```python
import pandas as pd
variable_name = pd.Series({
    "index1": "value1",
    "index2": "value2",
    ...
}, name="SeriesName")
print(variable_name)
```

**📝 Description:

- `pd.Series({...})` → Create a Series from a **dictionary** (keys = index, values = data).
- `name="..."` → Optional, sets the **name** of the Series.
- `variable_name` → Stores the Series.
- `print(...)` → Displays the Series.

**✅ Example:

```python
ingredients = pd.Series({
    "Flour": "4 cups",
    "Milk": "1 cup",
    "Eggs": "2 large",
    "Spam": "1 can"
}, name="Dinner")
```

### **Key Points:**

- A Series is a one-dimensional labeled array.
- It can hold various data types (int, float, string, etc.).
- You can create it from lists, dictionaries, scalar values, or NumPy arrays.
- Series support vectorized operations and provide various methods for data manipulation.
