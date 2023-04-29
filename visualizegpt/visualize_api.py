from typing import Any
import pandas as pd
import os 
import openai

global matplotlib 
import matplotlib
global plt
import matplotlib.pyplot as plt
global sns
import seaborn as sns
global np
import numpy as np

PROMPT = """
You are visualizeGPT, a language model who is incredible at writing code to create visualizations from pandas dataframes. You are an expert python programmer at matplotlib and seaborn. You are a functional programmer, who highly values clean code. Your task is to create code that visualizes the data in the pandas DataFrame.

I am going to provide two things
1. an example of the first few rows of the pandas DataFrame
2. a query for what plot to generate

Based on the query, you will write amazing code to create the plot using matplotlib, seaborn or plotly. Follow these instructions when creating the plots: 

Make sure to make the plots visually beautiful. 
Make sure to write concise code. 
Make sure to sort values when appropriate. 
When creating plots, no text should overlap. 
You may rotate text if needed. Do not create any index errors. Write comments to explain what the code should be doing.
If the query asks to use a specific package, use that package. Otherwise, use matplotlib or seaborn.
Include all imports necessary to make the code run. 

The most important thing is to make sure the code runs. It is absolutely critical that the code runs. Make sure there are no errors in the code, since that would be very bad. Do not include any instructions. Do not return anything except the code. When I am ready for you response, I will end my query with "Response:"

If your code has an error, I will append the error to the end of this query after the text "Error:". You will respond with the corrected code, and nothing else.

Here is an example:

Example 1:
Data: 

Unnamed: 0 	nGene 	nUMI 	Cluster 	Organoid 	CellType 	Batch 	numeric_CellType
0 	11a.6mon_X1_AAACCTGAGAATCTCC 	1010 	1969 	14 	10 	Immature Interneurons 	11a_6mon 	5
1 	11a.6mon_X1_AAACCTGAGACAGACC 	2197 	5700 	4 	10 	CPNs 	11a_6mon 	1
2 	11a.6mon_X1_AAACCTGAGATGTAAC 	1142 	2152 	1 	10 	Immature PNs 	11a_6mon 	6
3 	11a.6mon_X1_AAACCTGAGGAGTTGC 	1148 	2411 	6 	10 	Immature Interneurons 	11a_6mon 	5
4 	11a.6mon_X1_AAACCTGCAATGGTCT 	2196 	5014 	3 	10 	Immature CPNs 	11a_6mon 	4

Query: Show the distribution of labels
Response:
label_counts = df['CellType'].value_counts()

plt.bar(label_counts.index, label_counts.values)
plt.xticks(rotation=90)
plt.title("Label Distribution")

Ready? Here is your prompt. It is super important to not use data from the previous example! Only use the following data to create your plot:
"""

class VisualizeGPT:
    def __init__(self, df: pd.DataFrame, error_tolerance: int = 3, openai_api_key: str = None, temperature: float = 0.0):
        try:
            os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise KeyError(f"Please set the OPENAI_API_KEY environment variable to your OpenAI API key or initialize {self.__class__.__name__} with the openai_api_key parameter. See https://platform.openai.com/docs/introduction for more information on how to get an API key.")

        self.df = df
        self.prompt = PROMPT + "\nData:\n" + str(df.head())
        self.error_tolerance = error_tolerance
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.responses = []

    def __call__(self, query: str, show: bool = True) -> str:
        prompt = self.prompt + "\nQuery: " + query + "\nResponse:"
        response = get_code_response_from_llm(prompt, self.temperature)

        if show:
            tries = 0
            while tries < self.error_tolerance:
                try:
                    # replace df with self.df so we access local df
                    exec(response.replace("df", "self.df"), globals(), locals())
                    break
                except Exception as e:
                    print(f'Failed to run code, refining query and trying again ({tries + 1} / {self.error_tolerance}).')
                    tries += 1
                    prompt += "\nError: " + str(e) + "\nResponse:"
                    response = get_code_response_from_llm(prompt, self.temperature)
        else:
            print(response)

        response = CodePrompt(code=response, df=self.df, openai_api_key=self.openai_api_key, temperature=self.temperature, error_tolerance=self.error_tolerance)
        self.responses.append(response)

    @property
    def response(self):
        print(self.responses[-1])


class CodePrompt:
    REFINEMENT_PROMPT = """
    You are visualizeGPT, a language model who is incredible at writing code to create visualizations from pandas dataframes. You are an expert python programmer at matplotlib and seaborn. You are a functional programmer, who highly values clean code. 

    Your task is to modify Python code with the requested changes. I will give you the following:
    1. A code snippet that produces a visualization using matplotlib, seaborn or plotly 
    2. A request for what changes to make to the code snippet

    You will modify the code snippet to make the requested changes. Follow these instructions when modifying the code:
    Make sure to make the plots visually beautiful. 
    Make sure to write concise code. 
    Make sure to sort values when appropriate. 
    When creating plots, no text should overlap. 
    You may rotate text if needed. Do not create any index errors. Write comments to explain what the code should be doing.
    If the query asks to use a specific package, use that package. Otherwise, use matplotlib or seaborn.
    Include all imports necessary to make the code run. 
    Make the changes given by the query.

    If your code has an error, I will append the error to the end of this query after the text "Error:". You will respond with the corrected code, and nothing else.

    Reply only with the code with comments explaining each line. Do not include any other text in your response.

    Example:
    Code: 
    label_counts = df['CellType'].value_counts()

    plt.bar(label_counts.index, label_counts.values)
    plt.xticks(rotation=90)
    plt.title("Label Distribution")

    Query: Change the color scheme to red 
    Response: 
    label_counts = df['CellType'].value_counts()

    plt.bar(label_counts.index, label_counts.values, color='red')
    plt.xticks(rotation=90)
    plt.title("Label Distribution")

    Ready? Here is your prompt. It is super important to not use anything from the previous example! Only use the following data to create your plot:
    """
    def __init__(self, code: str, df: pd.DataFrame, openai_api_key: str = None, temperature: float = 0.0, error_tolerance: int = 3):
        self.code = code
        self.df = df
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.error_tolerance = error_tolerance

        self.responses = [code]

    def refine(self, query: str, show=True) -> str:
        prompt = self.REFINEMENT_PROMPT + "\nCode:\n" + self.responses[-1] + "\nQuery:" + query + "\nResponse:"
        response = get_code_response_from_llm(prompt, self.temperature)
        self.responses.append(response)

        tries = 0
        if show:
            while tries < self.error_tolerance:
                try:
                    # replace df with self.df so we access local df
                    exec(response.replace("df", "self.df"), globals(), locals())
                    break
                except Exception as e:
                    print(f'Failed to run code, refining query and trying again ({tries + 1} / {self.error_tolerance}).')
                    tries += 1
                    prompt += "\nError: " + str(e) + "\nResponse:"
                    response = get_code_response_from_llm(prompt, self.temperature)
        else:
            print(response)

    @property
    def response(self):
        print(self.responses[-1])

    def __repr__(self) -> str:
        return self.responses[-1]
    
    def __str__(self) -> str:
        return self.responses[-1]

def get_code_response_from_llm(query: str, temperature: float = 0.0) -> str:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    model_engine = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{"role": "user", "content": query}],
        max_tokens=1000,
        temperature=temperature,
    )
    response = response.choices[0].message.content
    # sometimes the model returns a with this text, not sure why
    response = response.replace("Response:", "").rstrip()

    return response