from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

hub_llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"temperature": 0.05, "min_length": 512, "max_length": 1024})

template_string = """
You are a professional English teacher.
I'd like you to help me understand a specific word.

Take the word below delimited by triple backticks.
word: ```{word}```

then based on the word, please provide the meaning of the word, offer synonyms for it, and create 3 sentences using that word in different contexts to illustrate its usage.
The output should be a json":
"""

prompt = PromptTemplate(
  input_variables=["word"],
  template=template_string
)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(hub_chain.run("resolution"))