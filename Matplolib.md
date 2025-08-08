## Introduction : 

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It provides an object-oriented API for embedding plots into applications and a procedural interface similar to MATLAB.

### Key Features

- Publication-quality figures in various formats
- Interactive figures that can zoom, pan, and update
- Customizable to the smallest detail
- Works well with NumPy and Pandas
- Extensive gallery of examples and documentation

---

## Installation : 

### Using pip

```bash
pip install matplotlib
```

### Using conda

```bash
conda install matplotlib
```

### With additional dependencies

```bash
pip install matplotlib[complete]
```

### Basic Import

```python
import matplotlib.pyplot as plt
import numpy as np
```

---

## Basic Concepts

### Architecture

Matplotlib has a hierarchical structure:

- **Figure**: The entire figure (window or page)
- **Axes**: A subplot within a figure (where data is plotted)
- **Axis**: The number lines (x-axis, y-axis)
- **Artist**: Everything on the figure (lines, text, etc.)

### Two Interfaces

1. **Pyplot Interface**: MATLAB-style stateful interface
2. **Object-Oriented Interface**: More explicit control

```python
# Pyplot interface
plt.plot([1, 2, 3, 4])
plt.ylabel('Values')
plt.show()

# Object-oriented interface
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4])
ax.set_ylabel('Values')
plt.show()
```

---

## Figure and Axes

### Creating Figures

```python
# Simple figure
fig = plt.figure()

# Figure with specific size
fig = plt.figure(figsize=(10, 6))

# Figure with subplots
fig, ax = plt.subplots()
fig, axes = plt.subplots(2, 2)  # 2x2 grid
```

### Subplot Configuration

```python
# Various subplot arrangements
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

# Manual subplot positioning
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(221)  # 2x2 grid, position 1
ax2 = fig.add_subplot(222)  # 2x2 grid, position 2
```

### Axes Properties

```python
ax.set_xlim(0, 10)
ax.set_ylim(-5, 5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Plot Title')
ax.grid(True)
```

---

## Basic Plotting

### Line Plots

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.plot(x, y, 'r--')  # Red dashed line
plt.plot(x, y, color='blue', linestyle='-', linewidth=2)
```

### Scatter Plots

```python
x = np.random.randn(100)
y = np.random.randn(100)

plt.scatter(x, y)
plt.scatter(x, y, c='red', s=50, alpha=0.5)
```

### Bar Charts

```python
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

plt.bar(categories, values)
plt.barh(categories, values)  # Horizontal bars
```

### Histograms

```python
data = np.random.normal(0, 1, 1000)

plt.hist(data, bins=30)
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
```

---

## Plot Customization

### Colors and Styles

```python
# Color specifications
plt.plot(x, y, color='red')
plt.plot(x, y, color='#FF5733')
plt.plot(x, y, color=(0.1, 0.2, 0.5))

# Line styles
plt.plot(x, y, linestyle='-')   # Solid
plt.plot(x, y, linestyle='--')  # Dashed
plt.plot(x, y, linestyle='-.')  # Dash-dot
plt.plot(x, y, linestyle=':')   # Dotted

# Markers
plt.plot(x, y, marker='o')      # Circle
plt.plot(x, y, marker='s')      # Square
plt.plot(x, y, marker='^')      # Triangle
```

### Labels and Legends

```python
plt.plot(x, y1, label='Sin(x)')
plt.plot(x, y2, label='Cos(x)')
plt.legend()
plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
```

### Text and Annotations

```python
plt.text(5, 0.5, 'Text at (5, 0.5)')
plt.annotate('Maximum', xy=(max_x, max_y), xytext=(max_x+1, max_y+0.1),
            arrowprops=dict(arrowstyle='->', color='red'))
```

### Styling with rcParams

```python
import matplotlib as mpl

# Global settings
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.grid'] = True
mpl.rcParams['figure.figsize'] = (10, 6)

# Context managers
with plt.style.context('seaborn-v0_8'):
    plt.plot(x, y)
```

---

## Multiple Plots

### Subplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, y1)
axes[0, 1].scatter(x, y2)
axes[1, 0].bar(categories, values)
axes[1, 1].hist(data)

plt.tight_layout()
```

### subplot2grid

```python
fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))
```

### GridSpec

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :-1])
ax3 = fig.add_subplot(gs[1:, -1])
ax4 = fig.add_subplot(gs[-1, 0])
ax5 = fig.add_subplot(gs[-1, -2])
```

---

## Advanced Plot Types

### Contour Plots

```python
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

plt.contour(X, Y, Z)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar()
```

### Heatmaps

```python
data = np.random.randn(10, 10)
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()
```

### Box Plots

```python
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.boxplot(data)
```

### Violin Plots

```python
plt.violinplot(data)
```

### Polar Plots

```python
theta = np.linspace(0, 2*np.pi, 100)
r = np.sin(4*theta)

ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
```

### Error Bars

```python
x = np.arange(0, 4, 0.2)
y = np.exp(-x)
yerr = 0.1 * np.sqrt(y)

plt.errorbar(x, y, yerr=yerr, fmt='o-')
```

---

## 3D Plotting

### Setup

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
```

### 3D Line Plot

```python
t = np.linspace(0, 20, 1000)
x = np.sin(t)
y = np.cos(t)
z = t

ax.plot(x, y, z)
```

### 3D Scatter Plot

```python
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

ax.scatter(x, y, z)
```

### 3D Surface Plot

```python
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis')
```

---

## Animations

### Basic Animation

```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def animate(frame):
    line.set_ydata(np.sin(x + frame/10))
    return line,

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()
```

### Saving Animations

```python
ani.save('animation.gif', writer='pillow', fps=20)
ani.save('animation.mp4', writer='ffmpeg', fps=20)
```

---

## Saving and Exporting

### Save Formats : 

```python
# Various formats
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plot.pdf', bbox_inches='tight')
plt.savefig('plot.svg', bbox_inches='tight')
plt.savefig('plot.eps', bbox_inches='tight')
```

### Quality Settings : 

```python
plt.savefig('plot.png', 
           dpi=300,                    # Resolution
           bbox_inches='tight',        # Tight bounding box
           facecolor='white',          # Background color
           edgecolor='none',           # Edge color
           transparent=True,           # Transparent background
           pad_inches=0.1)             # Padding
```

---

## Best Practices

### Code Organization

```python
# Good practice: Use object-oriented interface
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Title')
plt.tight_layout()
plt.show()
```

### Performance Tips

```python
# Use appropriate data types
x = np.array(x_data, dtype=np.float32)

# Limit data points for large datasets
if len(data) > 10000:
    data = data[::10]  # Every 10th point

# Use blitting for animations
ani = FuncAnimation(fig, animate, blit=True)
```

### Memory Management

```python
# Close figures when done
plt.close('all')
plt.close(fig)

# Use context managers
with plt.style.context('seaborn-v0_8'):
    fig, ax = plt.subplots()
    # ... plotting code ...
    plt.show()
```

---

## Common Issues and Solutions

### Issue: Figures not showing

```python
# Solution: Add plt.show() or use interactive mode
plt.ion()  # Interactive mode on
plt.ioff() # Interactive mode off
```

### Issue: Overlapping labels

```python
# Solution: Use tight_layout or adjust spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)
```

### Issue: Memory leaks

```python
# Solution: Close figures explicitly
plt.close('all')
```

### Issue: Slow performance

```python
# Solution: Use appropriate backends
import matplotlib
matplotlib.use('Agg')  # For non-interactive use
```

### Issue: Font problems

```python
# Solution: Check available fonts
from matplotlib import font_manager
print([f.name for f in font_manager.fontManager.ttflist])
```

---

## Function Reference

### Core Figure Functions


```python
plt.figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None)`
```

Creates a new figure window.

- `num`: Figure number or name
- `figsize`: Width, height in inches as tuple (8, 6)
- `dpi`: Resolution in dots per inch
- `facecolor`: Background color
- `edgecolor`: Edge color around figure

```python
fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
```

plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=None)
****
Creates figure and subplots in one call.

- `nrows`, `ncols`: Number of rows and columns
- `sharex`, `sharey`: Share x or y axis across subplots
- `squeeze`: Return single axes object if only one subplot
- Returns: `fig, ax` or `fig, axes`

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
```

### Plotting Functions : 

``` python
plt.plot(x, y, format_string, **kwargs)
```

Creates line plots.

- `x`, `y`: Data arrays
- `format_string`: Combined color, marker, line style (e.g., 'ro-')
- `label`: Legend label
- `linewidth`: Line thickness
- `markersize`: Marker size
- `alpha`: Transparency (0-1)

```python
plt.plot([1, 2, 3], [4, 5, 6], 'ro-', linewidth=2, markersize=8)
```

``` python
plt.scatter(x, y, s=None, c=None, marker=None, alpha=None, **kwargs
```


Creates scatter plots.

- `s`: Size of markers (scalar or array)
- `c`: Color (single color or array for color mapping)
- `marker`: Marker style
- `cmap`: Colormap for color mapping
- `edgecolors`: Edge color of markers

```python
plt.scatter(x, y, s=50, c=colors, marker='o', alpha=0.7, cmap='viridis')
```

```python 
plt.bar(x, height, width=0.8, bottom=None, **kwargs)
```


Creates bar charts.

- `x`: X coordinates of bars
- `height`: Heights of bars
- `width`: Width of bars
- `bottom`: Y coordinates of bar bottoms
- `align`: Alignment ('center' or 'edge')
- `color`: Bar colors

```python
plt.bar(categories, values, width=0.6, color='skyblue', edgecolor='black')
```

``` python
plt.hist(x, bins=None, density=False, alpha=None, **kwargs)`
```

Creates histograms.

- `x`: Data to histogram
- `bins`: Number of bins or bin edges
- `density`: If True, normalize to form probability density
- `cumulative`: If True, compute cumulative histogram
- `histtype`: Type ('bar', 'step', 'stepfilled')

```python
plt.hist(data, bins=30, density=True, alpha=0.7, histtype='stepfilled')
```

### Customization Functions

```python
plt.xlabel(label, fontsize=None, **kwargs)`
```

Sets x-axis label.

- `label`: Label text
- `fontsize`: Font size
- `fontweight`: Font weight ('normal', 'bold')
- `color`: Text color

```python
plt.xlabel('Time (seconds)', fontsize=14, fontweight='bold')
```

``` python 
plt.title(label, fontsize=None, loc='center', **kwargs)`
```

Sets plot title.

- `label`: Title text
- `fontsize`: Font size
- `loc`: Location ('left', 'center', 'right')
- `pad`: Padding from axes

```python
plt.title('My Plot Title', fontsize=16, loc='center', pad=20)
```

```python
plt.legend(labels=None, loc='best', **kwargs)`
```

Adds legend to plot.

- `labels`: Legend labels (if not set via plot labels)
- `loc`: Location ('best', 'upper right', 'lower left', etc.)
- `bbox_to_anchor`: Manual positioning
- `ncol`: Number of columns
- `fontsize`: Font size

```python
plt.legend(['Line 1', 'Line 2'], loc='upper right', fontsize=12)
```

```python
plt.xlim(left=None, right=None)
plt.ylim(bottom=None, top=None)
```

Sets axis limits.

- `left`, `right`: X-axis limits
- `bottom`, `top`: Y-axis limits

```python
plt.xlim(0, 10)
plt.ylim(-5, 5)
```

```python
plt.grid(visible=None, which='major', axis='both', **kwargs)`
```

Adds grid to plot.

- `visible`: Show grid (True/False)
- `which`: Grid lines ('major', 'minor', 'both')
- `axis`: Which axis ('both', 'x', 'y')
- `alpha`: Grid transparency
- `linestyle`: Grid line style

```python
plt.grid(True, alpha=0.3, linestyle='--')
```

### Advanced Functions

```python
plt.imshow(X, cmap=None, aspect=None, interpolation=None, **kwargs)`
```

Displays image or 2D array.

- `X`: Image data
- `cmap`: Colormap
- `aspect`: Aspect ratio ('auto', 'equal', number)
- `interpolation`: Interpolation method ('nearest', 'bilinear')
- `vmin`, `vmax`: Data range for colormap

```python
plt.imshow(image_data, cmap='gray', aspect='equal', interpolation='bilinear')
```

#### `plt.contour(X, Y, Z, levels=None, **kwargs)` / `plt.contourf(X, Y, Z, levels=None, **kwargs)`

Creates contour plots.

- `X`, `Y`: Coordinate arrays
- `Z`: Height values
- `levels`: Number of contour levels or specific levels
- `cmap`: Colormap
- `contourf` fills between contours

```python
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar()  # Add colorbar
```

```python
plt.annotate(text, xy, xytext=None, **kwargs)`
```

Adds annotations with arrows.

- `text`: Annotation text
- `xy`: Point to annotate
- `xytext`: Text position
- `arrowprops`: Arrow properties dict
- `fontsize`: Text font size

```python
plt.annotate('Peak', xy=(5, 10), xytext=(6, 12),
            arrowprops=dict(arrowstyle='->', color='red'))
```

### Statistical Plot Functions

```python
plt.boxplot(x, notch=False, patch_artist=False, **kwargs)`
```

Creates box plots.

- `x`: Data for box plots
- `notch`: Show notches
- `patch_artist`: Fill boxes with color
- `labels`: Box labels
- `showmeans`: Show mean markers

```python
plt.boxplot([data1, data2, data3], labels=['A', 'B', 'C'], patch_artist=True)
```

```python
plt.violinplot(dataset, positions=None, **kwargs)`
```

Creates violin plots.

- `dataset`: Data for violin plots
- `positions`: Positions of violins
- `showmeans`: Show mean markers
- `showmedians`: Show median markers

```python
plt.violinplot([data1, data2, data3], showmeans=True, showmedians=True)
```

### Axis Object Methods

```python
ax.set_xlim(left, right)` / `ax.set_ylim(bottom, top)`
```

Sets axis limits (OOP interface).

```python
ax.set_xlabel(label, **kwargs)` / `ax.set_ylabel(label, **kwargs)`
```

Sets axis labels (OOP interface).

```python
ax.set_title(label, **kwargs)
```

Sets subplot title (OOP interface).

```python 
ax.tick_params(axis='both', **kwargs)
```

Customizes tick parameters.

- `axis`: Which axis ('x', 'y', 'both')
- `labelsize`: Tick label size
- `colors`: Tick colors
- `direction`: Tick direction ('in', 'out', 'inout')

```python
ax.tick_params(axis='x', labelsize=12, colors='blue', direction='in')
```

### File I/O Functions

```python
plt.savefig(fname, dpi=None, bbox_inches=None, **kwargs)`
```

Saves figure to file.

- `fname`: Filename with extension
- `dpi`: Resolution
- `bbox_inches`: Bounding box ('tight' for automatic)
- `facecolor`: Background color
- `transparent`: Transparent background

```python
plt.savefig('plot.png', dpi=300, bbox_inches='tight', transparent=True)
```

```python
plt.show()
```

Displays the current figure.

``` python
plt.close(fig=None)
```

Closes figure(s).

- `fig`: Figure to close (None for current, 'all' for all)

### Interactive Functions

```python
plt.ion()
plt.ioff()
```

Turns interactive mode on/off.

``` python
plt.pause(interval)
```

Pauses execution for interval seconds.

## Additional Essential Topics

### Working with Dates and Times

```python
import matplotlib.dates as mdates
from datetime import datetime

# Date formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())

# Rotate date labels
plt.xticks(rotation=45)
```

### LaTeX Integration

```python
# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Use LaTeX in labels
plt.xlabel(r'$\alpha$ (radians)')
plt.ylabel(r'$\sin(\alpha)

---

*This documentation covers the essential aspects of matplotlib for creating effective visualizations. For more advanced features and detailed examples, refer to the official matplotlib documentation.*)
plt.title(r'$y = \sin(\alpha)

---

*This documentation covers the essential aspects of matplotlib for creating effective visualizations. For more advanced features and detailed examples, refer to the official matplotlib documentation.*)
```

### Event Handling

```python
def on_click(event):
    if event.inaxes is not None:
        print(f'Clicked at x={event.xdata:.2f}, y={event.ydata:.2f}')

fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', on_click)
```

### Quick Reference

### Common Parameters

- `figsize`: Figure size (width, height) in inches
- `dpi`: Dots per inch for resolution
- `color`/`c`: Color specification
- `linestyle`/`ls`: Line style ('-', '--', '-.', ':')
- `linewidth`/`lw`: Line width
- `marker`: Marker style
- `markersize`/`ms`: Marker size
- `alpha`: Transparency (0-1)

### Color Maps

- `viridis`, `plasma`, `inferno`, `magma`
- `hot`, `cool`, `spring`, `summer`, `autumn`, `winter`
- `gray`, `bone`, `copper`, `pink`
- `jet`, `rainbow`, `hsv`

### Marker Styles

- `'o'`: Circle
- `'s'`: Square
- `'^'`: Triangle up
- `'v'`: Triangle down
- `'*'`: Star
- `'+'`: Plus
- `'x'`: Cross
- `'D'`: Diamond

## 📝 Text and Labels

### Basic Text Elements

```python
import matplotlib.pyplot as plt

# Titles and labels
plt.title('Chart Title')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# Adding text at specific coordinates
plt.text(x, y, 'Text content', fontsize=12)

# Annotations with arrows
plt.annotate('Point of interest', xy=(x, y), xytext=(x+1, y+1),
            arrowprops=dict(arrowstyle='->', color='red'))
```

### Text Formatting

```python
# Font properties
plt.title('Title', fontsize=16, fontweight='bold', color='blue')
plt.xlabel('X Label', fontfamily='serif', style='italic')

# Text positioning
plt.text(0.5, 0.5, 'Centered text', 
         horizontalalignment='center', 
         verticalalignment='center',
         transform=plt.gca().transAxes)
```

## 🏷️ Legends

### Basic Legend

```python
# Simple legend
plt.plot(x, y, label='Line 1')
plt.plot(x, y2, label='Line 2')
plt.legend()

# Legend positioning
plt.legend(loc='upper right')    # or 'upper left', 'lower right', etc.
plt.legend(loc='best')           # Auto-positioning
```

### Advanced Legend Options

```python
# Custom legend
plt.legend(['Series 1', 'Series 2'])

# Legend formatting
plt.legend(frameon=True, fancybox=True, shadow=True)
plt.legend(fontsize=12, title='Legend Title')

# Legend outside plot area
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
```

## 🎨 Colors and Styling

### Web Colors

```python
# Named colors (Wikipedia web colors)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Hex colors
plt.plot(x, y, color='#FF5733')

# RGB tuples
plt.plot(x, y, color=(0.1, 0.2, 0.5))

# Color abbreviations
plt.plot(x, y, color='r')  # 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
```

### Line Styling

#### Line Width

```python
plt.plot(x, y, linewidth=2)     # or lw=2
plt.plot(x, y, lw=0.5)          # Thin line
plt.plot(x, y, lw=5)            # Thick line
```

#### Line Style

```python
plt.plot(x, y, linestyle='-')    # Solid line (default)
plt.plot(x, y, linestyle='--')   # Dashed line
plt.plot(x, y, linestyle='-.')   # Dash-dot line
plt.plot(x, y, linestyle=':')    # Dotted line

# Short notation
plt.plot(x, y, ls='--')
```

#### Markers

```python
# Common markers
plt.plot(x, y, marker='o')       # Circle
plt.plot(x, y, marker='s')       # Square
plt.plot(x, y, marker='^')       # Triangle up
plt.plot(x, y, marker='v')       # Triangle down
plt.plot(x, y, marker='*')       # Star
plt.plot(x, y, marker='+')       # Plus
plt.plot(x, y, marker='x')       # X mark
plt.plot(x, y, marker='D')       # Diamond

# Marker customization
plt.plot(x, y, marker='o', markersize=8, markerfacecolor='red', 
         markeredgecolor='black', markeredgewidth=2)
```

### Plot Styles

```python
# Available styles
plt.style.available  # List all available styles

# Apply style
plt.style.use('seaborn-v0_8')
plt.style.use('ggplot')
plt.style.use('dark_background')
plt.style.use('classic')

# Temporary style
with plt.style.context('seaborn-v0_8'):
    plt.plot(x, y)
```

## 📊 Scatter Plots

### Basic Scatter Plot

```python
plt.scatter(x, y)
plt.scatter(x, y, s=50)          # Size parameter
plt.scatter(x, y, c='red')       # Color parameter
```

### Advanced Scatter Options

```python
# Variable size and color
plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)

# Color mapping
plt.scatter(x, y, c=values, cmap='viridis')
plt.colorbar()  # Add colorbar

# Marker customization
plt.scatter(x, y, marker='s', s=100, 
           edgecolors='black', linewidths=2)
```

### Alpha (Transparency)

```python
# Alpha parameter (0 = transparent, 1 = opaque)
plt.scatter(x, y, alpha=0.5)     # Semi-transparent
plt.scatter(x, y, alpha=0.1)     # Very transparent
plt.plot(x, y, alpha=0.7)        # Works with lines too
```

## 📈 Bar Charts

### Vertical Bar Chart

```python
plt.bar(x,height)
plt.bar(categories, values, color='skyblue')
plt.bar(x, y, width=0.8)         # Bar width
```

### Horizontal Bar Chart

```python
plt.barh(y, width)
plt.barh(categories, values, color='lightgreen')
plt.barh(y, x, height=0.8) # Bar height
```

### Stacked Bar Charts

```python
# Stacked vertical bars
plt.bar(x, values1, label='Series 1')
plt.bar(x, values2, bottom=values1, label='Series 2')
plt.bar(x, values3, bottom=values1+values2, label='Series 3')

# Stacked horizontal bars
plt.barh(y, values1, label='Series 1')
plt.barh(y, values2, left=values1, label='Series 2')
```

### Bar Chart Styling

```python
# Custom colors and styling
plt.bar(x, y, color=['red', 'blue', 'green'], 
        edgecolor='black', linewidth=2)

# Error bars
plt.bar(x, y, yerr=error_values, capsize=5)
```

## 📊 Error Bars

### Basic Error Bars

```python
# Vertical error bars
plt.errorbar(x, y, yerr=error_values)

# Horizontal error bars
plt.errorbar(x, y, xerr=error_values)

# Both directions
plt.errorbar(x, y, xerr=x_errors, yerr=y_errors)
```

### Error Bar Styling

```python
plt.errorbar(x, y, yerr=errors, 
            fmt='o',                 # Marker style
            capsize=5,              # Cap size
            capthick=2,             # Cap thickness
            ecolor='red',           # Error bar color
            elinewidth=2)           # Error bar width
```

## 📊 Histograms

### Basic Histogram

```python
plt.hist(data)
plt.hist(data, bins=20)          # Number of bins
plt.hist(data, bins=50, alpha=0.7)
```

### Histogram with Bins and Range

```python
# Custom bins
plt.hist(data, bins=[0, 10, 20, 30, 40, 50])  # Custom bin edges

# Range parameter (sets min and max)
plt.hist(data, bins=30, range=(0, 100))       # Only data from 0 to 100
plt.hist(data, range=(10, 90))                # Focus on middle range
```

### Histogram Styling

```python
# Multiple histograms
plt.hist(data1, bins=30, alpha=0.7, label='Dataset 1')
plt.hist(data2, bins=30, alpha=0.7, label='Dataset 2')

# Histogram appearance
plt.hist(data, bins=20, color='skyblue', 
         edgecolor='black', linewidth=1.2)
```

## 📊 Normalization and Density

### Normalization in Bar Plots

```python
# Normalize bar heights (make them sum to 1)
values_normalized = values / sum(values)
plt.bar(x, values_normalized)

# Percentage normalization
values_percent = (values / sum(values)) * 100
plt.bar(x, values_percent)
```

### Density in Histograms

```python
# Density=True: Area under curve = 1 (probability density)
plt.hist(data, bins=30, density=True)

# Density=False: Shows actual counts (default)
plt.hist(data, bins=30, density=False)
```

**Density Explanation:**

- **`density=True`**: Histogram shows probability density (area = 1)
- **`density=False`**: Histogram shows frequency counts
- Density is useful for comparing distributions of different sample sizes

## 🎛️ Common Matplotlib Parameters

### Figure and Axes

```python
# Figure size
plt.figure(figsize=(10, 6))

# Subplot spacing
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Grid
plt.grid(True, alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.5)
```

### Axis Formatting

```python
# Axis limits
plt.xlim(0, 10)
plt.ylim(-5, 5)

# Axis scaling
plt.xscale('log')
plt.yscale('log')

# Tick formatting
plt.xticks(rotation=45)
plt.yticks(fontsize=12)
```

### Common Plot Parameters

```python
# Universal parameters that work with most plots
plt.plot(x, y, 
         color='blue',          # Color
         linewidth=2,           # Line width
         linestyle='--',        # Line style
         marker='o',            # Marker
         markersize=8,          # Marker size
         alpha=0.7,             # Transparency
         label='Data series')   # Legend label
```

## 💡 Quick Reference

### Color Options

- Named: `'red'`, `'blue'`, `'green'`, `'orange'`, `'purple'`
- Hex: `'#FF5733'`, `'#3498DB'`
- RGB: `(0.1, 0.2, 0.5)`
- Single letter: `'r'`, `'g'`, `'b'`, `'c'`, `'m'`, `'y'`, `'k'`

### Line Styles

- `'-'` (solid), `'--'` (dashed), `'-.'` (dash-dot), `':'` (dotted)

### Markers

- `'o'` (circle), `'s'` (square), `'^'` (triangle), `'*'` (star), `'+'` (plus), `'x'` (X)

### Common Plot Types

```python
plt.plot()      # Line plot
plt.scatter()   # Scatter plot
plt.bar()       # Bar chart
plt.barh()      # Horizontal bar
plt.hist()      # Histogram
plt.errorbar()  # Error bars
```

This reference covers all the essential Matplotlib functions and parameters for creating professional-looking plots!






---

_This documentation covers the essential aspects of matplotlib for creating effective visualizations. For more advanced features and detailed examples, refer to the official matplotlib documentation._
