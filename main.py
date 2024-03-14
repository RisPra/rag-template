from llm import load_gemini
from llm import load_llama
from langchain.chains.sequential import SequentialChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = load_llama()

country_prompt = PromptTemplate(
    input_variables = ["letter"],
    template = '''Give me the name of one country that starts with the letter {letter}.'''
)

country_chain = LLMChain(
    llm = llm,
    prompt = country_prompt,
    output_key = "country"
)

capital_prompt = PromptTemplate(
    input_variables = ["country"],
    template = '''What is the capital of {country}'''
)

capital_chain = LLMChain(
    llm = llm,
    prompt = capital_prompt,
    output_key = "capital"
)

chain = SequentialChain(
    input_variables = ['letter'],
    output_variables = ['country', 'capital'],
    chains = [
        country_chain,
        capital_chain
    ]
)

result = chain.invoke("A")

print(result)