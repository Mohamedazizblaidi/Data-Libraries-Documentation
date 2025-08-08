
Pandas is a powerful library in Python used for data manipulation and analysis. Below is a list of some of the most commonly used Pandas functions, along with examples of how to use them.

###  **Import Pandas***
Once Pandas is installed, import it in your applications by adding the `import` keyword:

```python
import pandas
```

### **Pandas as pd***
Pandas is usually imported under the `pd` alias.
Create an alias with the `as` keyword while importing:

```python
import pandas as pd
```

###  **Creating DataFrames and Series**

- `pd.DataFrame(data, index, columns)`: Creates a DataFrame.
- `pd.Series(data, index)`: Creates a Series.

```python
import pandas as pd
# Creating a DataFrame

data = {'Name': ['John', 'Anna', 'Peter'], 'Age': [28, 24, 35]}
df = pd.DataFrame(data)
print(df)
# Creating a Series

s = pd.Series([1, 2, 3, 4])
print(s)

```

###  **Reading and Writing Data**

- `pd.read_csv('file.csv')`: Reads a CSV file into a DataFrame.
- `df.to_csv('file.csv')`: Writes a DataFrame to a CSV file.

```python
df = pd.read_csv('data.csv')
df.to_csv('output.csv')
```
### **Inspecting Data**

- `df.head(n)`: Returns the first `n` rows of the DataFrame.
- `df.tail(n)`: Returns the last `n` rows of the DataFrame.
- `df.info()`: Provides a concise summary of the DataFrame.
- `df.describe()`: Generates descriptive statistics.

```python
print(df.head(5))
print(df.tail(5))
print(df.info())
print(df.describe())
```

### **Selecting Data**

- `df['column_name']`: Selects a single column.
- `df[['col1', 'col2']]`: Selects multiple columns.
- `df.loc[row_label]`: Selects data by label.
- `df.iloc[row_position]`: Selects data by position.

```python
print(df['Name'])
print(df[['Name', 'Age']])
print(df.loc[0])
print(df.iloc[0])
```

### **Shape of a DataFrame**

- **Attribute:** `df.shape`
- **Description:** Returns a tuple representing the dimensions of the DataFrame. The first value is the number of rows, and the second value is the number of columns.

```python
import pandas as pd

# Example DataFrame
data = {'Name': ['John', 'Anna', 'Peter'], 'Age': [28, 24, 35]}
df = pd.DataFrame(data)

# Getting the shape
shape = df.shape
print(shape)  # Output: (3, 2) - 3 rows, 2 columns
```

### **Size of a DataFrame**

- **Attribute:** `df.size`
- **Description:** Returns the total number of elements in the DataFrame, which is equivalent to the number of rows multiplied by the number of columns.

```python
# Getting the size
size = df.size
print(size)  # Output: 6 - Total number of elements
```

### **Example Summary**

Given the DataFrame `df`:

```python
   Name  Age
0  John   28
1  Anna   24
2  Peter  35
```

- **`df.shape`** would return `(3, 2)`, indicating the DataFrame has 3 rows and 2 columns.
- **`df.size`** would return `6`, indicating the DataFrame contains 6 elements in total.

### **Filtering Data**

- `df[df['column_name'] > value]`: Filters data based on condition.

```python
filtered_df = df[df['Age'] > 25]
print(filtered_df)
```

### **Sorting Data**

- `df.sort_values('column_name')`: Sorts the DataFrame by a column.
- `df.sort_index()`: Sorts the DataFrame by its index.

```python
sorted_df = df.sort_values('Age')
print(sorted_df)
```

##  Pandas – `sort_values()` et le paramètre ascending : 

- `ascending` est un **paramètre booléen** dans la méthode `DataFrame.sort_values()`.
- Il permet de **définir l’ordre du tri** :
  - `ascending=True` (par défaut) → ordre **croissant**
  - `ascending=False` → ordre **décroissant**

### ✍️ Exemple :

```python
# Trier par âge en ordre croissant
df.sort_values(by='Age', ascending=True)

# Trier par âge en ordre décroissant
df.sort_values(by='Age', ascending=False)
````

###  **Handling Missing Data**

- `df.isnull()`: Detects missing values.
- `df.dropna()`: Drops rows with missing values.
- `df.fillna(value)`: Fills missing values with a specified value.

```python
missing_data = df.isnull()
df_cleaned = df.dropna()
df_filled = df.fillna(0)
```

### **Removing Rows**

You can remove rows from a DataFrame using the `drop()` method.

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
# Removing the row at index 1 (Bob's row)
df = df.drop(1)
print(df)

# Removing rows where Age is greater than 30
df = df[df['Age'] <= 30]
print(df)
```

### **`dropna()`**

The `dropna()` method is used to remove missing data (NaN) from a DataFrame or Series. It can drop rows or columns where NaN values are present.

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', None], 'Age': [25, None, 35]}
df = pd.DataFrame(data)
df_clean = df.dropna()
print(df_clean)
```

**Key Parameters:**

- `axis`: Specifies whether to drop rows (`axis=0`) or columns (`axis=1`).
- `how`: If `how='any'`, it drops rows/columns with any NaNs; if `how='all'`, it drops rows/columns only if all values are NaN.
- `subset`: Specify which columns to consider when dropping.

### **`fillna()`**

The `fillna()` method is used to fill missing data (NaN) with a specified value or method.

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, None, 35]}
df = pd.DataFrame(data)
df_filled = df.fillna(0)  # Replace NaN with 0
print(df_filled)
```

**Key Parameters:**

- `value`: The value to replace NaNs with.
- `method`: The method to use for filling (`'ffill'` for forward fill, `'bfill'` for backward fill).
- `axis`: Fill along rows (`axis=0`) or columns (`axis=1`).

###  **Grouping and Aggregating**

- `df.groupby('column_name')`: Groups data by a column.
		- `df.agg({'col1': 'sum', 'col2': 'mean'})`: Aggregates data.

```python
grouped_df = df.groupby('Age').sum()
aggregated_df = df.agg({'Age': 'sum', 'Name': 'count'})
```

###  **Merging and Joining**

- `pd.merge(df1, df2, on='column_name')`: Merges two DataFrames.
- `df1.join(df2)`: Joins two DataFrames on their indexes.

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
merged_df = pd.merge(df1, df2, on='key')
```

###   **Reshaping Data**

- `df.pivot(index, columns, values)`: Pivots the DataFrame.
- `df.melt(id_vars, value_vars)`: Unpivots the DataFrame.

```python
pivot_df = df.pivot(index='Name', columns='Age', values='Score')
melted_df = df.melt(id_vars=['Name'], value_vars=['Score'])
```

### **Date and Time**

- `pd.to_datetime(df['column_name'])`: Converts a column to datetime.
- `df['column_name'].dt.year`: Extracts the year from datetime.


```python
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
```

###  **String Operations**

- `df['column_name'].str.lower()`: Converts strings to lowercase.
- `df['column_name'].str.contains('pattern')`: Checks for a pattern


```python
df['Name'] = df['Name'].str.lower()
contains_pattern = df['Name'].str.contains('john')
```

### **Exporting Data**

- `df.to_excel('filename.xlsx')`: Exports DataFrame to an Excel file.
- `df.to_json('filename.json')`: Exports DataFrame to a JSON file.

```python
df.to_excel('output.xlsx')
df.to_json('output.json')
```

### **Sorting Data**

- **`df.sort_values(by='column', ascending=True)`**: Sort DataFrame by column values.
- **`df.sort_index()`**: Sort DataFrame by its index

```python
sorted_df = df.sort_values(by='Age', ascending=False)
```

### **Discovering Duplicates**

To find duplicate rows in a DataFrame, use the `duplicated()` method, which returns a Boolean Series indicating whether a row is a duplicate or not.

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Alice'], 'Age': [25, 30, 25]}
df = pd.DataFrame(data)

# Finding duplicates
duplicates = df.duplicated()
print(duplicates)

# Removing duplicates
df_no_duplicates = df.drop_duplicates()
print(df_no_duplicates)
```

### **`duplicated()`**

The `duplicated()` method identifies duplicate rows in a DataFrame, returning a Series of booleans where `True` indicates a duplicate.

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie', 'Alice'], 'Age': [25, 30, 35, 25]}
df = pd.DataFrame(data)
print(df.duplicated())
```

**key Parameters:**

- `subset`: Specify columns to check for duplicates.
- `keep`: Determine which duplicates to mark as `True` (`'first'`, `'last'`, `False`).

### **`drop_duplicates()`**

The `drop_duplicates()` method removes duplicate rows from a DataFrame.

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie', 'Alice'], 'Age': [25, 30, 35, 25]}
df = pd.DataFrame(data)
df_no_duplicates = df.drop_duplicates()
print(df_no_duplicates)
```

**Key Parameters:**

- `subset`: Specify columns to consider when dropping duplicates.
- `keep`: Which duplicate to keep (`'first'`, `'last'`, `False`).

### **`pd.options.display.max_rows`**

This option in Pandas controls the maximum number of rows that will be displayed when printing a DataFrame. By default, Pandas might truncate the output if the DataFrame is too large, but you can adjust this setting.

```python
import pandas as pd
# Set the maximum number of rows displayed to 100
pd.options.display.max_rows = 100
Example:
import pandas as pd
# Creating a large DataFrame
df = pd.DataFrame({'A': range(1, 101), 'B': range(101, 201)})
print(df)
```

With `max_rows` set to 100, the DataFrame will display up to 100 rows.

### **Reading Data**

**Read CSV File** :

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df)
```

**Read Excel File**:

```python
import pandas as pd
pd.read_excel('file.xlsx')
df = pd.read_excel('data.xlsx')
print(df)
```

**Read SQL Database**:

```python
pd.read_sql('query', connection)
import pandas as pd
from sqlalchemy import create_engine
# Create a database connection
engine = create_engine('sqlite:///my_database.db')
query = 'SELECT * FROM my_table'
df = pd.read_sql(query, engine)
print(df)
```

### **Reading a JSON File in Pandas and What is a JSON File**

**What is a JSON File?**

- JSON (JavaScript Object Notation) is a lightweight data-interchange format that's easy for humans to read and write, and easy for machines to parse and generate. It is commonly used for transmitting data in web applications.
- A JSON file typically consists of key-value pairs, arrays, and objects.

**Reading a JSON File in Pandas:** You can read JSON files directly into a Pandas DataFrame using the `pd.read_json()` function.

```python
import pandas as pd
# Reading a JSON file
df = pd.read_json('data.json')  # Replace 'data.json' with your file name
print(df)

Example JSON Content:
[
    {"Name": "Alice", "Age": 25, "City": "New York"},
    {"Name": "Bob", "Age": 30, "City": "Los Angeles"}
]
```

### **Renaming Columns**

```python
df.rename(columns={'old_name': 'new_name'})
import pandas as pd
# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
# Renaming column 'A' to 'alpha'
df = df.rename(columns={'A': 'alpha'})
print(df)

```css
   alpha  B
0      1  4
1      2  5
2      3  6
```

### **Setting the Index**
```python
df.set_index('column')
import pandas as pd
# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Setting column 'A' as the index
df = df.set_index('A')
print(df)

```css
   B
A   
1  4
2  5
3  6

df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.set_index('column', inplace=True)

```

### **Writing Data**

**Write CSV File**:

```python
df.to_csv('file.csv')
import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
})
df.to_csv('output.csv', index=False)  # index=False prevents writing row numbers
```

**Write Excel File** :

```python
df.to_excel('file.xlsx')
import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
})
df.to_excel('output.xlsx', index=False)  # index=False prevents writing row numbers
```

**Write SQL Table**:

```python
df.to_sql('table_name', connection)
import pandas as pd
from sqlalchemy import create_engine
df = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
})
# Create a database connection
engine = create_engine('sqlite:///my_database.db')
df.to_sql('my_table', engine, if_exists='replace', index=False)  # if_exists='replace' overwrites the table if it exists
```

### **`to_string()`**

The `to_string()` method is used to render a DataFrame or Series as a string in a human-readable format. It's especially useful for displaying data in environments where the output might get truncated.

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print(df.to_string())
```

**Key Parameters:**

- `index`: Whether to print the index (default is `True`).
- `columns`: Specify which columns to include in the output.
- `max_rows` and `min_rows`: Control the number of rows to show.

### **Replacing Values**

The `replace()` method is used to replace specific values in a DataFrame.

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
# Replacing a specific value in the DataFrame
df['Age'] = df['Age'].replace(25, 26)
print(df)
```

You can replace multiple values or use a dictionary for more complex replacements:

```python
df['Name'] = df['Name'].replace({'Alice': 'Alicia', 'Bob': 'Bobby'})
print(df)
```

### `index_col` in Pandas : 

- **Purpose**: Used when reading data into a DataFrame with functions like `pd.read_csv()`, `pd.read_excel()`, etc.
- **Function**: Specifies which column(s) to use as the **row index** of the Dat Frame.

#### ✅ Example:

``` python
`df = pd.read_csv('data.csv', index_col=0)`
```

- This sets the **first column (index 0)** as the DataFrame index.

#### 🔁 You can use:

- An **integer** (column position)
- A **string** (column name)
- A **list** (multiple columns for a MultiIndex)
