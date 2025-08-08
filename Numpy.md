## Overview

NumPy is a fundamental library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. NumPy is the backbone of many scientific and data analysis libraries in Python, such as SciPy, Pandas, and Matplotlib.

**Key Features**:

- **N-dimensional Arrays**: Efficient storage and manipulation of arrays.
- **Mathematical Operations**: Element-wise operations, linear algebra, Fourier transforms, and more.
- **Broadcasting**: Perform operations on arrays of different shapes.
- **Integration**: Works seamlessly with other Python libraries for scientific computing.

---

## Installation

To install NumPy, use pip or conda in your terminal or command prompt:

```bash
# Using pip
pip install numpy

# Using conda
conda install numpy
```

Verify the installation by checking the version in Python:

```python
import numpy as np
print(np.__version__)
```

---
## Getting Started

### Importing NumPy

NumPy is typically imported with the alias `np`:

```python
import numpy as np
```

### Creating Arrays

NumPy arrays (`ndarray`) are the core data structure. Here are common ways to create them:

```python
# 1D array from a list
arr1 = np.array([1, 2, 3, 4])
print(arr1)  # Output: [1 2 3 4]

# 2D array (matrix)
arr2 = np.array([[1, 2], [3, 4]])
print(arr2)
# Output: 
# [[1 2]
#  [3 4]]

# Array with zeros
zeros = np.zeros((2, 3))  # 2x3 array of zeros
print(zeros)
# Output:
# [[0. 0. 0.]
#  [0. 0. 0.]]

# Array with ones
ones = np.ones((3, 2))  # 3x2 array of ones
print(ones)
# Output:
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]

# Array with a range of values
range_arr = np.arange(0, 10, 2)  # Start, stop, step
print(range_arr)  # Output: [0 2 4 6 8]
```

### Array Attributes

Inspect array properties:

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape:", arr.shape)  # Output: (2, 3)
print("Dimensions:", arr.ndim)  # Output: 2
print("Data type:", arr.dtype)  # Output: int64
print("Size:", arr.size)  # Output: 6
```

---

## Basic Operations

### Element-wise Operations

NumPy supports element-wise arithmetic:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)  # Output: [5 7 9]
print(a * b)  # Output: [4 10 18]
print(a ** 2)  # Output: [1 4 9]
```

### Broadcasting

Broadcasting allows operations on arrays of different shapes:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print(a + b)
# Output:
# [[11 22 33]
#  [14 25 36]]
```

### Mathematical Functions

NumPy provides many mathematical functions:

```python
arr = np.array([0, np.pi/2, np.pi])
print(np.sin(arr))  # Output: [0. 1. 0.]
print(np.exp(arr))  # Output: [ 1.          4.81047738 23.14069263]
```

---

## Indexing and Slicing

Access and modify array elements:

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 1])  # Output: 2 (first row, second column)
print(arr[:, 1])  # Output: [2 5] (all rows, second column)
print(arr[1, :])  # Output: [4 5 6] (second row, all columns)

# Modify elements
arr[0, 0] = 10
print(arr)
# Output:
# [[10  2  3]
#  [ 4  5  6]]
```

---

## Advanced Features

### Advanced Indexing

NumPy supports advanced indexing techniques, including integer and boolean indexing:

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Integer indexing
indices = np.array([0, 2])
print(arr[indices])  # Select rows 0 and 2
# Output:
# [[1 2 3]
#  [7 8 9]]

# Boolean indexing
mask = arr > 5
print(arr[mask])  # Select elements greater than 5
# Output: [6 7 8 9]
```

### Structured Arrays

Structured arrays allow for heterogeneous data types, similar to a database table:

```python
# Define a structured dtype
dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
data = np.array([('Alice', 25, 55.5), ('Bob', 30, 80.0)], dtype=dtype)
print(data['name'])  # Output: ['Alice' 'Bob']
print(data[0])  # Output: ('Alice', 25, 55.5)
```

### Universal Functions (ufuncs)

ufuncs are vectorized functions that operate element-wise on arrays. Custom ufuncs can be created:

```python
# Built-in ufunc
arr = np.array([1, 4, 9])
print(np.sqrt(arr))  # Output: [1. 2. 3.]

# Custom ufunc
def my_func(x):
    return x ** 2 + 2 * x + 1
ufunc = np.frompyfunc(my_func, 1, 1)
print(ufunc(arr))  # Output: [ 4  9 16]
```

### Memory Management

NumPy arrays are memory-efficient due to their contiguous memory layout. Use `copy` and `view` to control memory usage:

```python
arr = np.array([1, 2, 3])
view = arr.view()  # Shares memory
copy = arr.copy()  # Independent copy
view[0] = 10
print(arr)  # Output: [10  2  3] (view modifies original)
copy[0] = 20
print(arr)  # Output: [10  2  3] (copy does not)
```

### Broadcasting Rules

Advanced broadcasting involves aligning arrays with compatible shapes:

```python
a = np.ones((3, 1))  # Shape (3, 1)
b = np.array([1, 2, 3])  # Shape (3,)
print(a + b)  # Broadcasting stretches b to (3, 3)
# Output:
# [[2. 3. 4.]
#  [2. 3. 4.]
#  [2. 3. 4.]]
```

### Array Manipulation

Advanced array manipulation techniques include stacking, splitting, and tiling:

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Stacking
vstack = np.vstack((a, b))  # Vertical stack
print(vstack)
# Output:
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Splitting
split = np.split(vstack, 2)  # Split into 2 arrays
print(split)
# Output: [array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]])]

# Tiling
tiled = np.tile(a, (2, 3))  # Repeat a 2 times vertically, 3 times horizontally
print(tiled)
# Output:
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]
```

### Linear Algebra (Expanded)

Beyond basic operations, NumPy supports advanced linear algebra:

```python
a = np.array([[1, 2], [3, 4]])

# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(a)
print("Eigenvalues:", eigvals)
# Output: Eigenvalues: [-0.37228132  5.37228132]

# Singular Value Decomposition (SVD)
u, s, vh = np.linalg.svd(a)
print("Singular values:", s)
# Output: Singular values: [5.4649857 0.36596619]
```

### Random Number Generation

NumPy's `random` module provides tools for generating random numbers:

```python
# Random integers
rand_int = np.random.randint(1, 10, size=(2, 3))
print(rand_int)  # Output: e.g., [[4 7 2]
                 #              [8 1 5]]

# Random normal distribution
rand_norm = np.random.normal(loc=0, scale=1, size=(2, 2))
print(rand_norm)  # Output: e.g., [[-0.123  0.456]
                  #              [ 1.789 -0.321]]
```

### Performance Optimization

Use in-place operations and vectorization to optimize performance:

```python
arr = np.arange(1000000)
# Vectorized operation (fast)
arr *= 2

# Avoid loops (slow)
# for i in range(len(arr)):
#     arr[i] *= 2
```

## 📊 Array Creation

### Basic Array Creation

```python
# From list
arr = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# From tuple
arr = np.array((1, 2, 3, 4))

# Specify data type
arr = np.array([1, 2, 3], dtype=np.float32)
arr = np.array([1, 2, 3], dtype='int64')
```

### Array Generation Functions

#### `np.arange()` - Range of Values

```python
# Basic usage
np.arange(10)           # [0, 1, 2, ..., 9]
np.arange(1, 11)        # [1, 2, 3, ..., 10]
np.arange(0, 10, 2)     # [0, 2, 4, 6, 8] (step=2)
np.arange(0, 1, 0.1)    # [0.0, 0.1, 0.2, ..., 0.9]
```

#### `np.ones()` - Array of Ones

```python
np.ones(5)              # [1. 1. 1. 1. 1.]
np.ones((3, 4))         # 3x4 matrix of ones
np.ones((2, 3, 4))      # 3D array of ones
np.ones(5, dtype=int)   # Integer ones
```

#### `np.zeros()` - Array of Zeros

```python
np.zeros(5)             # [0. 0. 0. 0. 0.]
np.zeros((3, 4))        # 3x4 matrix of zeros
np.zeros_like(arr)      # Zeros with same shape as arr
```

#### `np.eye()` - Identity Matrix

```python
np.eye(3)               # 3x3 identity matrix
np.eye(4, 5)            # 4x5 identity matrix
np.eye(3, k=1)          # Diagonal offset by 1
```

#### `np.full()` - Array with Specific Value

```python
np.full(5, 7)           # [7 7 7 7 7]
np.full((2, 3), 3.14)   # 2x3 matrix filled with 3.14
```

## 🎲 Random Array Generation

### `np.random.rand()` - Uniform Random [0,1)

```python
np.random.rand()        # Single random number
np.random.rand(5)       # 1D array of 5 random numbers
np.random.rand(3, 4)    # 3x4 matrix of random numbers
```

### `np.random.randint()` - Random Integers

```python
np.random.randint(10)           # Random int from 0 to 9
np.random.randint(1, 11)        # Random int from 1 to 10
np.random.randint(1, 11, 5)     # Array of 5 random ints
np.random.randint(1, 11, (3, 4)) # 3x4 matrix of random ints
```

### Other Random Functions

```python
# Normal distribution
np.random.randn(5)              # Standard normal distribution
np.random.normal(0, 1, 1000)    # Normal(mean=0, std=1)

# Random choice
np.random.choice([1, 2, 3, 4, 5], size=10)  # Random sampling
np.random.shuffle(arr)          # Shuffle array in-place

# Set random seed for reproducibility
np.random.seed(42)
```

## 🔢 Mathematical Operations

### Basic Arithmetic

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# Element-wise operations
arr1 + arr2             # Addition
arr1 - arr2             # Subtraction
arr1 * arr2             # Multiplication
arr1 / arr2             # Division
arr1 ** 2               # Power
arr1 % 2                # Modulo
```

### Mathematical Functions

```python
# Trigonometric functions
np.sin(arr)             # Sine
np.cos(arr)             # Cosine
np.tan(arr)             # Tangent
np.arcsin(arr)          # Inverse sine
np.degrees(arr)         # Radians to degrees
np.radians(arr)         # Degrees to radians

# Exponential and logarithmic
np.exp(arr)             # e^x
np.log(arr)             # Natural logarithm
np.log10(arr)           # Base-10 logarithm
np.log2(arr)            # Base-2 logarithm

# Rounding
np.round(arr, 2)        # Round to 2 decimal places
np.ceil(arr)            # Ceiling
np.floor(arr)           # Floor
np.trunc(arr)           # Truncate

# Other functions
np.sqrt(arr)            # Square root
np.abs(arr)             # Absolute value
np.sign(arr)            # Sign function
```

## 📊 Statistical Functions

### Aggregation Functions

```python
arr = np.array([1, 2, 3, 4, 5])

# Basic statistics
np.sum(arr)             # Sum of all elements
np.mean(arr)            # Mean
np.median(arr)          # Median
np.std(arr)             # Standard deviation
np.var(arr)             # Variance
np.min(arr)             # Minimum
np.max(arr)             # Maximum

# Along axes (for 2D arrays)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
np.sum(arr_2d, axis=0)  # Sum along columns
np.sum(arr_2d, axis=1)  # Sum along rows
```

### Other Statistical Functions

```python
# Percentiles
np.percentile(arr, 25)  # 25th percentile
np.percentile(arr, [25, 50, 75])  # Multiple percentiles

# Correlation and covariance
np.corrcoef(arr1, arr2) # Correlation coefficient
np.cov(arr1, arr2)      # Covariance

# Histograms
np.histogram(arr, bins=10)  # Histogram data
```

## 📏 Distance Calculations

### Euclidean Distance

```python
# Between two points
point1 = np.array([1, 2, 3])
point2 = np.array([4, 6, 8])
distance = np.linalg.norm(point2 - point1)

# Manual calculation
distance = np.sqrt(np.sum((point2 - point1)**2))

# Between arrays of points
points1 = np.array([[1, 2], [3, 4]])
points2 = np.array([[5, 6], [7, 8]])
distances = np.linalg.norm(points2 - points1, axis=1)
```

### Other Distance Metrics

```python
# Manhattan distance
manhattan = np.sum(np.abs(point2 - point1))

# Cosine similarity
cosine_sim = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
```

## 🔪 Array Slicing and Indexing

### 1D Array Slicing

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Basic indexing
arr[0]                  # First element
arr[-1]                 # Last element
arr[2:5]                # Elements 2, 3, 4
arr[:5]                 # First 5 elements
arr[5:]                 # Elements from index 5 to end
arr[::2]                # Every 2nd element
arr[::-1]               # Reverse array
```

### 2D Array Slicing

```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Basic indexing
arr_2d[0, 0]            # Element at row 0, column 0
arr_2d[1, 2]            # Element at row 1, column 2
arr_2d[0]               # First row
arr_2d[:, 0]            # First column
arr_2d[0:2, 1:3]        # Submatrix

# Advanced indexing
arr_2d[[0, 2], [1, 3]]  # Elements at (0,1) and (2,3)
```

### Multidimensional Array Access

```python
# 3D array
arr_3d = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])

# Access elements
arr_3d[0]               # First 2D slice
arr_3d[0, 1]            # First slice, second row
arr_3d[0, 1, 0]         # Specific element
arr_3d[:, :, 0]         # All first elements of innermost arrays
```

## 🔄 Broadcasting and Changing Values

### Broadcasting

```python
# Broadcasting allows operations between arrays of different shapes
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Add scalar to all elements
arr + 10

# Add 1D array to each row
arr + np.array([10, 20, 30])

# Add column vector to each column
arr + np.array([[10], [20]])
```

### Changing Values

```python
arr = np.array([1, 2, 3, 4, 5])

# Change single element
arr[0] = 10

# Change multiple elements
arr[1:4] = [20, 30, 40]

# Change with broadcasting
arr[:] = 0              # Set all to 0
arr[arr > 3] = 999      # Conditional assignment
```

### Matrix Operations

```python
# Matrix creation
matrix = np.array([[1, 2], [3, 4]])

# Matrix multiplication
result = np.dot(matrix, matrix)     # or matrix @ matrix
result = np.matmul(matrix, matrix)  # Same as above

# Matrix properties
np.transpose(matrix)    # or matrix.T
np.linalg.inv(matrix)   # Inverse
np.linalg.det(matrix)   # Determinant
np.trace(matrix)        # Trace (sum of diagonal)
```

## 🎯 Conditional Selection

### Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create boolean mask
mask = arr > 5          # [False, False, False, False, False, True, True, True, True, True]

# Select elements
arr[mask]               # [6, 7, 8, 9, 10]
arr[arr > 5]            # Same as above

# Multiple conditions
arr[(arr > 3) & (arr < 8)]  # [4, 5, 6, 7]
arr[(arr < 3) | (arr > 8)]  # [1, 2, 9, 10]
```

### `np.where()` Function

```python
# np.where(condition, value_if_true, value_if_false)
arr = np.array([1, 2, 3, 4, 5])
result = np.where(arr > 3, arr, 0)  # [0, 0, 0, 4, 5]

# Get indices where condition is True
indices = np.where(arr > 3)         # (array([3, 4]),)
```

## 🛠️ Built-in Functions and Methods

### Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Shape and dimensions
arr.shape               # (2, 3)
arr.ndim                # 2
arr.size                # 6
arr.dtype               # Data type
arr.itemsize            # Size of each element in bytes
```

### Array Manipulation

```python
# Reshaping
arr.reshape(3, 2)       # Change shape
arr.flatten()           # Flatten to 1D
arr.ravel()             # Flatten (view if possible)

# Transposing
arr.T                   # Transpose
arr.transpose()         # Same as above

# Joining arrays
np.concatenate([arr1, arr2])        # Join along existing axis
np.vstack([arr1, arr2])             # Vertical stack
np.hstack([arr1, arr2])             # Horizontal stack
np.column_stack([arr1, arr2])       # Stack as columns

# Splitting arrays
np.split(arr, 2)        # Split into 2 parts
np.hsplit(arr, 2)       # Horizontal split
np.vsplit(arr, 2)       # Vertical split
```

### Array Copying

```python
# View (shares memory)
arr_view = arr.view()

# Copy (independent)
arr_copy = arr.copy()

# Check if arrays share memory
np.shares_memory(arr, arr_view)     # True
np.shares_memory(arr, arr_copy)     # False
```

## 📋 Task Examples

### Task #1: Define Single and Multi-dimensional Arrays

```python
# Single-dimensional array
arr_1d = np.array([1, 2, 3, 4, 5])
arr_1d_range = np.arange(10)
arr_1d_ones = np.ones(5)

# Multi-dimensional arrays
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_2d_zeros = np.zeros((3, 4))
arr_2d_random = np.random.rand(3, 4)

# 3D array
arr_3d = np.ones((2, 3, 4))
arr_3d_identity = np.eye(4)
```

### Task #2: Leverage NumPy Built-in Methods

```python
arr = np.array([1, 2, 3, 4, 5])

# Statistical methods
mean_val = np.mean(arr)
std_val = np.std(arr)
max_val = np.max(arr)
min_val = np.min(arr)

# Array manipulation methods
reshaped = arr.reshape(5, 1)
transposed = arr.T
sorted_arr = np.sort(arr)
unique_vals = np.unique(arr)
```

### Task #3: Mathematical Operations

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# Basic operations
addition = arr1 + arr2
multiplication = arr1 * arr2
power = arr1 ** 2

# Mathematical functions
sqrt_vals = np.sqrt(arr1)
log_vals = np.log(arr1)
sin_vals = np.sin(arr1)

# Linear algebra
matrix = np.array([[1, 2], [3, 4]])
inverse = np.linalg.inv(matrix)
determinant = np.linalg.det(matrix)
```

### Task #4: Array Slicing and Indexing

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 1D slicing
first_three = arr[:3]
last_three = arr[-3:]
every_second = arr[::2]

# 2D slicing
first_row = matrix[0, :]
first_col = matrix[:, 0]
submatrix = matrix[0:2, 1:3]

# Advanced indexing
diagonal = matrix[[0, 1, 2], [0, 1, 2]]
```

### Task #5: Conditional Selection

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Boolean indexing
greater_than_5 = arr[arr > 5]
between_3_and_7 = arr[(arr >= 3) & (arr <= 7)]

# Using np.where()
conditional_replace = np.where(arr > 5, arr, 0)
indices = np.where(arr > 5)[0]

# Multiple conditions
complex_condition = arr[(arr > 2) & (arr < 8) & (arr % 2 == 0)]
```

## 🔗 NumPy Documentation

**Official NumPy Documentation:** https://numpy.org/doc/stable/

### Key Documentation Sections:

- **User Guide**: https://numpy.org/doc/stable/user/index.html
- **API Reference**: https://numpy.org/doc/stable/reference/index.html
- **Array Creation**: https://numpy.org/doc/stable/reference/routines.array-creation.html
- **Mathematical Functions**: https://numpy.org/doc/stable/reference/routines.math.html
- **Linear Algebra**: https://numpy.org/doc/stable/reference/routines.linalg.html
- **Random Sampling**: https://numpy.org/doc/stable/reference/random/index.html

## 💡 Quick Reference Summary

### Array Creation

- `np.array()` - From lists/tuples
- `np.arange()` - Range of values
- `np.ones()`, `np.zeros()` - Filled arrays
- `np.eye()` - Identity matrix
- `np.random.rand()`, `np.random.randint()` - Random arrays

### Mathematical Operations

- Element-wise: `+`, `-`, `*`, `/`, `**`
- Functions: `np.sin()`, `np.cos()`, `np.sqrt()`, `np.log()`
- Statistics: `np.mean()`, `np.std()`, `np.sum()`
- Linear algebra: `np.dot()`, `np.linalg.inv()`

### Indexing & Slicing

- Basic: `arr[0]`, `arr[1:5]`, `arr[::-1]`
- 2D: `arr[row, col]`, `arr[row, :]`, `arr[:, col]`
- Boolean: `arr[arr > 5]`, `arr[(arr > 3) & (arr < 8)]`

### Key Functions

- `np.where()` - Conditional selection
- `np.reshape()` - Change array shape
- `np.concatenate()` - Join arrays
- `np.split()` - Split arrays

# 🔁 NumPy: `tolist()` Function

## 📘 Description

The `tolist()` method in NumPy converts a **NumPy array** into a **native Python list**.  
This is useful when you want to:

- Export data.
- Use regular Python operations.
- Serialize data (e.g. to JSON).

---

## ✅ Syntax

```python
array.tolist()
```


### Example 1: 1D Array : 

```python
import numpy as np

arr = np.array([1, 2, 3])
py_list = arr.tolist()

print(py_list)
```

### ▶ Output:

``` python
[1, 2, 3] 
```

### Example 2: 2D Array : 

``` python
arr2d = np.array([[1, 2], [3, 4]])
py_list_2d = arr2d.tolist()

print(py_list_2d)
```
### ▶ Output:

````python
[[1, 2], [3, 4]]
````

### Example 3: 3D Array : 

```` python
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
py_list_3d = arr3d.tolist()
print(py_list_3d)
````
### ▶ Output:

````python
[
  [[1, 2], [3, 4]],
  [[5, 6], [7, 8]]
]
````

### 🔎 Use Cases : 

Convert NumPy data to JSON:

````python
import json
json_data = json.dumps(arr.tolist())
````

- Return results from a NumPy-based function in native Python format.

## 📎 Notes

- Works for arrays of any shape or dimension.
- Handles nested arrays properly.
- Maintains element data types (e.g., int, float) when possible.
- Output is **not** a NumPy object anymore.

This comprehensive guide covers all essential NumPy operations for scientific computing and data analysis!

---
## Further Resources

- **Official NumPy Documentation**: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)
- **NumPy User Guide**: [https://numpy.org/doc/stable/user/](https://numpy.org/doc/stable/user/)
- **Tutorials**:
    - [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
    - [SciPy Lecture Notes](https://scipy-lectures.org/intro/numpy/)
- **Community**:
    - Stack Overflow: `#numpy`
    - NumPy GitHub: [https://github.com/numpy/numpy](https://github.com/numpy/numpy)

---
## Tags

#Python #NumPy #DataScience #ScientificComputing
