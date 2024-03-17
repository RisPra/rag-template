from langchain_community.llms.ctransformers import CTransformers
from transformers import pipeline, Conversation, AutoModelForCausalLM, AutoTokenizer
from util.toMarkdown import to_markdown
import os

# class LLMLoader:
    
#     model_names = [
#         "Gemini Pro",
#         "Llama 17B"
#         "Llama 7B",
#     ]
    
#     def __init__(self) -> None:
#         pass
    
#     @property
#     def llm_names():
#         pass
    
#     @classmethod
#     def load_llm(model_name: str) -> CTransformers:
#         if model_name == "Llama":
#             pass


# is langchain compatible
def load_gemini():
    '''
        Usage:
            
    '''
    from langchain_google_genai import ChatGoogleGenerativeAI
    from secret_key import gemini_api_key as google_api_key
    
    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.0-pro",
        google_api_key = google_api_key
    )
    
    return llm

def load_llama() -> CTransformers:
    # model_paths = {
    #     "hugging_face": "TheBloke/OrcaMaidXL-17B-32k-GPTQ",
    #     "local": "D:/Documents/Projects/Langchain/OrcaMaidXL-17B-32k-GPTQ",
    #     "cache": "C:/Users/risha/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGML/snapshots/76cd63c351ae389e1d4b91cab2cf470aab11864b/llama-2-7b-chat.ggmlv3.q2_K.bin"
    # }
    
    tokeniser = AutoTokenizer.from_pretrained("TheBloke/OrcaMaidXL-17B-32k-GPTQ", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("TheBloke/OrcaMaidXL-17B-32k-GPTQ", trust_remote_code=True)

    prompt = "Llamas are a creature native to the country of "

    tokenised_sentence = tokeniser(
        prompt,
        return_tensors = "pt",
    )

    # input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    # output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    # print(tokenizer.decode(output[0]))

@DeprecationWarning
def _load_llama():
    
    model = {
        "hugging face hub": "TheBloke/OrcaMaidXL-17B-32k-GPTQ",
        "local path": "D:/Documents/Projects/Langchain/OrcaMaidXL-17B-32k-GPTQ/model.saftensors",
        "cache path": "C:/Users/risha/.cache/huggingface/hub/models--TheBloke--OrcaMaidXL-17B-32k-GPTQ/"
    }
    model_type = "llama"

    llm = CTransformers(
        model = model["local path"],
        model_type = model_type,
        max_new_tokens = 512,
        temperature = 0.1
    )
    
    return llm

if __name__ == "__main__":
    load_llama()