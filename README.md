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

>>> plot = viz("Show me the distribution of Cell Types for each batch. make a grid of subplots and use the blues colorscheme")
# shows the plot 
>>> plot.refine("Now make the y axis log scale and change the colorscheme to rainbow")
# shows the plot with the requested changes
>>> plot.refine(...)
```

There are two classes. `VisualizeGPT` is the base class which allows you to create visualizations from a DataFrame. When you query
for a visualization, `CodePrompt` object is returned with a `refine` method that allows you to iterate on your plots. You can see the code at the given step with the `CodePrompt().response` attribute, or the `.responses` attribute to see the history of all refinements.

You can run that code and get the visualization you want :-). Thanks LLMs!

Example:
![Visualization example](example.png?raw=true "Example of using VisualizeGPT")