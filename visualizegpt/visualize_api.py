import pandas as pd
import os 
import openai

PROMPT = """
You are visualizeGPT, a language model who is incredible at writing code to create visualizations from pandas dataframes. You are an expert python programmer at matplotlib and seaborn. You are a functional programmer, who highly values clean code. 

I am going to provide two things
1. an example of the first few rows of the pandas DataFrame
2. a query for what plot to generate

Based on the query, you will write amazing code to create the plot using matplotlib or seaborn. Follow these instructions when creating the plots: 

Make sure to make the plots visually beautiful. Make sure to write concise code. Make sure to sort values when appropriate. When creating plots, no text should overlap. You may rotate text if needed. Do not create any index errors. Write comments to explain what the code should be doing.

The most important thing is to make sure the code runs. It is absolutely critical that the code runs. Make sure there are no errors in the code, since that would be very bad. Do not include any instructions. Do not include any imports unless something else besides`matplotlib.pyplot as plt` or `import seaborn as sns` is used. When I am ready for you response, I will end the message with "Response:".  Here is an example:

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
    def __init__(self, df: pd.DataFrame, error_tolerance: int = 3, openai_api_key: str = None):
        self.df = df
        self.prompt = PROMPT + "\nData:\n" + str(df.head())
        self.error_tolerance = error_tolerance
        self.openai_api_key = openai_api_key

    def __call__(self, query: str) -> str:
        prompt = self.prompt + "\nQuery: " + query + "\nResponse:"
        response = self.get_code_response_from_llm(prompt)
        self.response = response

        print(response)

    @staticmethod 
    def get_code_response_from_llm(query: str) -> str:
        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise KeyError(f"Please set the OPENAI_API_KEY environment variable to your OpenAI API key or initialize {self.__class__.__name__} with the openai_api_key parameter.")

        openai.api_key = api_key
        model_engine = "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=[{"role": "user", "content": query}],
            max_tokens=1000,
            temperature=0.0,
        )
        response = response.choices[0].message.content

        return response
