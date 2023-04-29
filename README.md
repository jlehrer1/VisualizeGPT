# VisualizeGPT

A simple project that allows you to create visualizations with matplotlib, seaborn or plotly using natural language and gpt-turbo-3.5 from pandas dataframes.

To install, run
`pip install https://github.com/jlehrer1/VisualizeGPT.git`

The usage is very simple. Export your OpenAI key as an environment variable named `OPENAI_ACCESS_KEY` or pass it into the class initialization.

```
>>> import pandas as pd 
>>> from visualizegpt import VisualizeGPT
>>> df = pd.read_csv("my/data/file.csv")
>>> viz = VisualizeGPT(df)

>>> viz("Show me the distribution of Cell Types for each batch. make a grid of subplots and use the blues colorscheme")
import seaborn as sns

# group the data by batch and cell type, and count the number of occurrences
batch_counts = df.groupby(['Batch', 'CellType']).size().reset_index(name='Count')

# create a grid of subplots, one for each batch
g = sns.FacetGrid(batch_counts, col='Batch', col_wrap=3, sharey=False)

# plot a bar chart for each batch, with the cell types on the x-axis and the count on the y-axis
g.map(sns.barplot, 'CellType', 'Count', palette='Blues_d')

# rotate the x-axis labels to prevent overlap
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# set the title for each subplot
for ax, title in zip(g.axes.flat, batch_counts['Batch'].unique()):
    ax.set_title(title)

# set the overall title for the plot
g.fig.suptitle('Distribution of Cell Types for Each Batch', y=1.05)

# display the plot
plt.show()
```

You can run that code and get the visualization you want :-). Thanks LLMs!