# Complete Seaborn Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Setup](#basic-setup)
4. [Core Concepts](#core-concepts)
5. [Plot Types](#plot-types)
6. [Styling and Themes](#styling-and-themes)
7. [Working with DataFrames](#working-with-dataframes)
8. [Statistical Plots](#statistical-plots)
9. [Multi-plot Grids](#multi-plot-grids)
10. [Advanced Features](#advanced-features)
11. [Tips and Best Practices](#tips-and-best-practices)
12. [Common Issues and Solutions](#common-issues-and-solutions)

## Introduction

Seaborn is a Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. It comes with several built-in themes for styling matplotlib graphics and integrates closely with pandas data structures.

### Key Features
- Beautiful default styles and color palettes
- Built-in statistical plotting functions
- Flexible multi-plot grid system
- Integration with pandas DataFrames
- Statistical model fitting and visualization

## Installation

```bash
# Using pip
pip install seaborn

# Using conda
conda install seaborn

# For the latest development version
pip install git+https://github.com/mwaskom/seaborn.git
```

### Dependencies
- NumPy (>= 1.15)
- SciPy (>= 1.0)
- Pandas (>= 0.25)
- Matplotlib (>= 3.1)

## Basic Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Set the color palette
sns.set_palette("husl")

# Load example datasets
tips = sns.load_dataset("tips")
flights = sns.load_dataset("flights")
iris = sns.load_dataset("iris")
```

## Core Concepts

### Figure vs Axes-level Functions

Seaborn functions are divided into two categories:

**Figure-level functions**: Create entire figures and can show multiple subplots
- `relplot()`, `displot()`, `catplot()`, `lmplot()`, `jointplot()`, `pairplot()`

**Axes-level functions**: Draw onto specific matplotlib axes
- `scatterplot()`, `lineplot()`, `histplot()`, `boxplot()`, `barplot()`, etc.

```python
# Figure-level function
sns.relplot(data=tips, x="total_bill", y="tip", col="time")

# Axes-level function
fig, ax = plt.subplots()
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax)
```

### Long vs Wide Data Format

Seaborn works best with data in "long" (tidy) format:

```python
# Wide format (avoid)
wide_data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Long format (preferred)
long_data = wide_data.melt(var_name='category', value_name='value')
```

## Plot Types

### Distribution Plots

#### Histogram
```python
# Basic histogram
sns.histplot(data=tips, x="total_bill")

# With KDE overlay
sns.histplot(data=tips, x="total_bill", kde=True)

# Bivariate histogram
sns.histplot(data=tips, x="total_bill", y="tip")

# Multiple distributions
sns.histplot(data=tips, x="total_bill", hue="time", multiple="stack")
```

#### Kernel Density Estimation (KDE)
```python
# Univariate KDE
sns.kdeplot(data=tips, x="total_bill")

# Bivariate KDE
sns.kdeplot(data=tips, x="total_bill", y="tip")

# Multiple KDEs
sns.kdeplot(data=tips, x="total_bill", hue="time")
```

#### Box Plot
```python
# Basic box plot
sns.boxplot(data=tips, x="day", y="total_bill")

# With hue
sns.boxplot(data=tips, x="day", y="total_bill", hue="time")

# Horizontal box plot
sns.boxplot(data=tips, y="day", x="total_bill")
```

#### Violin Plot
```python
# Basic violin plot
sns.violinplot(data=tips, x="day", y="total_bill")

# Split violin plot
sns.violinplot(data=tips, x="day", y="total_bill", hue="time", split=True)

# Inner points
sns.violinplot(data=tips, x="day", y="total_bill", inner="points")
```

### Relational Plots

#### Scatter Plot
```python
# Basic scatter plot
sns.scatterplot(data=tips, x="total_bill", y="tip")

# With categorical variable
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")

# Size mapping
sns.scatterplot(data=tips, x="total_bill", y="tip", size="size")

# Style mapping
sns.scatterplot(data=tips, x="total_bill", y="tip", style="time")
```

#### Line Plot
```python
# Basic line plot
sns.lineplot(data=flights, x="year", y="passengers")

# Multiple lines
sns.lineplot(data=flights, x="year", y="passengers", hue="month")

# Confidence intervals
sns.lineplot(data=tips, x="size", y="total_bill", ci=95)

# Error bars instead of CI
sns.lineplot(data=tips, x="size", y="total_bill", err_style="bars")
```

### Categorical Plots

#### Bar Plot
```python
# Basic bar plot
sns.barplot(data=tips, x="day", y="total_bill")

# With hue
sns.barplot(data=tips, x="day", y="total_bill", hue="time")

# Estimator function
sns.barplot(data=tips, x="day", y="total_bill", estimator=np.median)

# Error bars
sns.barplot(data=tips, x="day", y="total_bill", ci=95)
```

#### Count Plot
```python
# Basic count plot
sns.countplot(data=tips, x="day")

# With hue
sns.countplot(data=tips, x="day", hue="time")

# Horizontal count plot
sns.countplot(data=tips, y="day")
```

#### Point Plot
```python
# Basic point plot
sns.pointplot(data=tips, x="day", y="total_bill")

# With hue
sns.pointplot(data=tips, x="day", y="total_bill", hue="time")

# Custom markers
sns.pointplot(data=tips, x="day", y="total_bill", markers=["o", "s"])
```

#### Strip Plot
```python
# Basic strip plot
sns.stripplot(data=tips, x="day", y="total_bill")

# Jittered
sns.stripplot(data=tips, x="day", y="total_bill", jitter=True)

# Dodged
sns.stripplot(data=tips, x="day", y="total_bill", hue="time", dodge=True)
```

#### Swarm Plot
```python
# Basic swarm plot
sns.swarmplot(data=tips, x="day", y="total_bill")

# With hue
sns.swarmplot(data=tips, x="day", y="total_bill", hue="time")

# Limited size for performance
sns.swarmplot(data=tips.sample(50), x="day", y="total_bill")
```

### Regression Plots

#### Linear Model Plot
```python
# Basic regression plot
sns.regplot(data=tips, x="total_bill", y="tip")

# Without regression line
sns.regplot(data=tips, x="total_bill", y="tip", fit_reg=False)

# Polynomial regression
sns.regplot(data=tips, x="total_bill", y="tip", order=2)

# Logistic regression
sns.regplot(data=tips, x="total_bill", y="tip", logistic=True)

# Robust regression
sns.regplot(data=tips, x="total_bill", y="tip", robust=True)
```

#### LM Plot (Figure-level)
```python
# Basic lmplot
sns.lmplot(data=tips, x="total_bill", y="tip")

# Separate plots by category
sns.lmplot(data=tips, x="total_bill", y="tip", col="time")

# Multiple regression lines
sns.lmplot(data=tips, x="total_bill", y="tip", hue="time")

# Grid of plots
sns.lmplot(data=tips, x="total_bill", y="tip", col="time", row="smoker")
```

#### Residual Plot
```python
# Residual plot
sns.residplot(data=tips, x="total_bill", y="tip")

# Against fitted values
sns.residplot(data=tips, x="total_bill", y="tip", lowess=True)
```

## Styling and Themes

### Built-in Themes
```python
# Available styles
styles = ["darkgrid", "whitegrid", "dark", "white", "ticks"]

for style in styles:
    sns.set_style(style)
    # Your plotting code here
```

### Color Palettes
```python
# Qualitative palettes
sns.set_palette("deep")  # default
sns.set_palette("muted")
sns.set_palette("bright")
sns.set_palette("pastel")
sns.set_palette("dark")
sns.set_palette("colorblind")

# Sequential palettes
sns.set_palette("Blues")
sns.set_palette("viridis")

# Diverging palettes
sns.set_palette("RdBu")
sns.set_palette("coolwarm")

# Custom palette
custom_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
sns.set_palette(custom_palette)

# View palette
sns.palplot(sns.color_palette("husl", 8))
```

### Context and Scaling
```python
# Context for different uses
sns.set_context("paper")    # smallest
sns.set_context("notebook")  # default
sns.set_context("talk")     # medium
sns.set_context("poster")   # largest

# Custom scaling
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
```

### Customizing with RC Parameters
```python
# Temporary style changes
with sns.axes_style("darkgrid"):
    # Your plots here
    pass

# Custom RC parameters
sns.set_theme(
    style="whitegrid",
    palette="muted",
    rc={
        "figure.figsize": (10, 6),
        "axes.spines.right": False,
        "axes.spines.top": False
    }
)
```

## Working with DataFrames

### Data Preparation
```python
# Sample data
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'value': np.random.randint(1, 10, 100)
})

# Melting wide to long format
df_melted = pd.melt(df, id_vars=['category'], value_vars=['x', 'y'])

# Pivoting for heatmaps
pivot_df = df.pivot_table(values='value', index='category', columns='variable')
```

### Handling Missing Data
```python
# Seaborn automatically handles NaN values in most cases
df_with_nan = df.copy()
df_with_nan.loc[10:20, 'y'] = np.nan

# Plot will automatically exclude NaN values
sns.scatterplot(data=df_with_nan, x='x', y='y')
```

## Statistical Plots

### Heatmap
```python
# Correlation matrix
corr_matrix = tips.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)

# Custom annotations
sns.heatmap(corr_matrix, 
           annot=True, 
           fmt='.2f',
           cmap='RdBu',
           square=True,
           cbar_kws={"shrink": .8})

# Pivot table heatmap
flights_pivot = flights.pivot("month", "year", "passengers")
sns.heatmap(flights_pivot, cmap='YlOrRd')
```

### Cluster Map
```python
# Hierarchical clustering
sns.clustermap(flights_pivot, cmap='viridis', standard_scale=1)

# Custom clustering
from scipy.cluster.hierarchy import linkage
linkage_matrix = linkage(flights_pivot, method='ward')
sns.clustermap(flights_pivot, row_linkage=linkage_matrix, cmap='viridis')
```

### Pair Plot
```python
# Basic pair plot
sns.pairplot(iris)

# With hue
sns.pairplot(iris, hue='species')

# Subset of variables
sns.pairplot(iris, vars=['sepal_length', 'sepal_width'])

# Different plot types
sns.pairplot(iris, diag_kind='kde', plot_kws={'alpha': 0.7})
```

### Joint Plot
```python
# Basic joint plot
sns.jointplot(data=tips, x='total_bill', y='tip')

# Different plot types
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde')

# Marginal plots
sns.jointplot(data=tips, x='total_bill', y='tip', 
              marginal_kws={'bins': 25, 'fill': False})
```

## Multi-plot Grids

### FacetGrid
```python
# Basic FacetGrid
g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(sns.scatterplot, 'total_bill', 'tip')
g.add_legend()

# Custom function
def custom_plot(x, y, **kwargs):
    plt.scatter(x, y, **kwargs)
    plt.axline((0, 0), slope=0.15, color='red', linestyle='--')

g = sns.FacetGrid(tips, col='time')
g.map(custom_plot, 'total_bill', 'tip', alpha=0.7)

# Sharing axes
g = sns.FacetGrid(tips, col='time', sharey=False, aspect=1.2)
g.map(sns.histplot, 'total_bill')
```

### PairGrid
```python
# Basic PairGrid
g = sns.PairGrid(iris)
g.map(sns.scatterplot)

# Different plots for different positions
g = sns.PairGrid(iris, hue='species')
g.map_diag(sns.histplot)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, fill=True)
g.add_legend()
```

### Using Figure-level Functions
```python
# relplot for relationships
sns.relplot(data=tips, x='total_bill', y='tip', 
           col='time', hue='day', style='smoker')

# displot for distributions
sns.displot(data=tips, x='total_bill', col='time', 
           hue='day', kind='kde', fill=True)

# catplot for categorical data
sns.catplot(data=tips, x='day', y='total_bill', 
           col='time', kind='violin', hue='smoker')
```

## Advanced Features

### Custom Functions with FacetGrid
```python
def annotate_plot(x, y, **kwargs):
    data = kwargs.pop('data')
    sns.scatterplot(data=data, x=x, y=y, **kwargs)
    
    # Add correlation annotation
    r = np.corrcoef(data[x], data[y])[0, 1]
    plt.gca().text(0.1, 0.9, f'r = {r:.2f}', 
                   transform=plt.gca().transAxes)

g = sns.FacetGrid(tips, col='time')
g.map_dataframe(annotate_plot, 'total_bill', 'tip')
```

### Statistical Annotations
```python
from scipy import stats

# Add statistical test results
def add_pvalue(x, y, **kwargs):
    data = kwargs.pop('data')
    sns.boxplot(data=data, x=x, y=y, **kwargs)
    
    # Perform t-test between groups
    groups = data.groupby(x)[y].apply(list)
    if len(groups) == 2:
        stat, p = stats.ttest_ind(*groups)
        plt.gca().text(0.5, 0.95, f'p = {p:.3f}', 
                       transform=plt.gca().transAxes, ha='center')

g = sns.FacetGrid(tips, col='time')
g.map_dataframe(add_pvalue, 'smoker', 'total_bill')
```

### Combining with Matplotlib
```python
# Using subplots with seaborn
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(data=tips, x='total_bill', y='tip', ax=axes[0, 0])
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0, 1])
sns.histplot(data=tips, x='total_bill', ax=axes[1, 0])
sns.heatmap(tips.corr(), ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

### Working with Dates
```python
# Time series data
dates = pd.date_range('2020-01-01', periods=100)
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.cumsum(np.random.randn(100)),
    'category': np.random.choice(['A', 'B'], 100)
})

# Line plot with dates
sns.lineplot(data=ts_data, x='date', y='value', hue='category')
plt.xticks(rotation=45)

# Using matplotlib date formatting
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
```

## Tips and Best Practices

### Performance Tips

1. **Use appropriate plot types for data size**:
```python
# For large datasets, use hexbin or kde instead of scatter
# Bad for large data
sns.scatterplot(data=large_df, x='x', y='y')  

# Better for large data
sns.jointplot(data=large_df, x='x', y='y', kind='hex')
```

2. **Sample large datasets**:
```python
# Sample large datasets
if len(df) > 10000:
    df_sample = df.sample(10000)
    sns.scatterplot(data=df_sample, x='x', y='y')
```

3. **Use rasterization for publication**:
```python
sns.scatterplot(data=df, x='x', y='y', rasterized=True)
```

### Data Preparation Tips

1. **Check data types**:
```python
# Ensure categorical variables are properly typed
df['category'] = df['category'].astype('category')
```

2. **Handle outliers**:
```python
# Remove outliers using IQR
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df['value'] < (Q1 - 1.5 * IQR)) | 
                (df['value'] > (Q3 + 1.5 * IQR)))]
```

3. **Consistent ordering**:
```python
# Define custom order for categorical variables
day_order = ['Thur', 'Fri', 'Sat', 'Sun']
sns.boxplot(data=tips, x='day', y='total_bill', order=day_order)
```

### Visualization Best Practices

1. **Choose appropriate color palettes**:
```python
# Qualitative data: use qualitative palettes
sns.scatterplot(data=tips, x='total_bill', y='tip', 
               hue='day', palette='Set2')

# Sequential data: use sequential palettes
sns.scatterplot(data=tips, x='total_bill', y='tip', 
               hue='size', palette='viridis')

# Diverging data: use diverging palettes
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0)
```

2. **Proper axis labels and titles**:
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.title('Relationship between Total Bill and Tip Amount')
```

3. **Use appropriate aspect ratios**:
```python
# For time series
plt.figure(figsize=(12, 4))
sns.lineplot(data=ts_data, x='date', y='value')

# For correlation matrices
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, square=True)
```

### Statistical Considerations

1. **Understanding confidence intervals**:
```python
# Bootstrap confidence intervals (default)
sns.barplot(data=tips, x='day', y='total_bill', ci=95)

# Standard error
sns.barplot(data=tips, x='day', y='total_bill', ci='sd')

# No error bars
sns.barplot(data=tips, x='day', y='total_bill', ci=None)
```

2. **Choosing appropriate estimators**:
```python
# Mean (default)
sns.barplot(data=tips, x='day', y='total_bill')

# Median (robust to outliers)
sns.barplot(data=tips, x='day', y='total_bill', estimator=np.median)

# Custom estimator
def q75(x):
    return np.percentile(x, 75)

sns.barplot(data=tips, x='day', y='total_bill', estimator=q75)
```

## Common Issues and Solutions

### Issue 1: Overlapping Labels
```python
# Problem: overlapping x-axis labels
sns.boxplot(data=tips, x='day', y='total_bill')

# Solution: rotate labels
plt.xticks(rotation=45)

# Or use horizontal plot
sns.boxplot(data=tips, y='day', x='total_bill')
```

### Issue 2: Legend Positioning
```python
# Problem: legend overlaps with plot
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='day')

# Solution: move legend outside
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Or remove legend if not needed
plt.legend().remove()
```

### Issue 3: Color Mapping Issues
```python
# Problem: inconsistent colors across plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bad: colors may not match
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0])
sns.countplot(data=tips, x='day', ax=axes[1])

# Good: specify consistent palette
palette = sns.color_palette("Set2", 4)
sns.boxplot(data=tips, x='day', y='total_bill', palette=palette, ax=axes[0])
sns.countplot(data=tips, x='day', palette=palette, ax=axes[1])
```

### Issue 4: Memory Issues with Large Data
```python
# Problem: plotting millions of points
# Solution: use density plots or sampling

# Density plot
sns.kdeplot(data=large_df, x='x', y='y', fill=True)

# Or hexbin
sns.jointplot(data=large_df, x='x', y='y', kind='hex')

# Or sample
sample_size = min(10000, len(large_df))
sample_df = large_df.sample(sample_size)
sns.scatterplot(data=sample_df, x='x', y='y')
```

### Issue 5: Customizing Figure-level Functions
```python
# Problem: hard to customize figure-level functions
g = sns.relplot(data=tips, x='total_bill', y='tip', col='time')

# Solution: access the underlying figure and axes
g.fig.suptitle('Tips by Time of Day', y=1.02)
for ax in g.axes.flat:
    ax.set_xlabel('Total Bill ($)')
    ax.set_ylabel('Tip ($)')

# Adjust layout
g.fig.tight_layout()
```

### Final Tips

1. Always explore your data first with `df.info()`, `df.describe()`, and `df.head()`
2. Start with simple plots and add complexity gradually
3. Use figure-level functions for quick exploration, axes-level for precise control
4. Save high-resolution figures: `plt.savefig('plot.png', dpi=300, bbox_inches='tight')`
5. Consider colorblind-friendly palettes: `sns.color_palette("colorblind")`
6. Use `plt.show()` explicitly in scripts to ensure plots display
7. For interactive exploration, consider using `%matplotlib inline` in Jupyter notebooks

This documentation covers the essential aspects of Seaborn. For the most up-to-date information and advanced features, always refer to the official Seaborn documentation at https://seaborn.pydata.org/.
