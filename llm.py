def load_llama():
    from langchain_community.llms.ctransformers import CTransformers

    model = {
        "hugging face hub": "TheBloke/Llama-2-7B-Chat-GGML",
        "local path": "C:/Users/risha/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGML/snapshots/76cd63c351ae389e1d4b91cab2cf470aab11864b/llama-2-7b-chat.ggmlv3.q2_K.bin"
    }
    model_type = "llama"

    llm = CTransformers(
        model = model["hugging face hub"],
        model_type = model_type,
        max_new_tokens = 512,
        temperature = 0.1
    )
    
    return llm

def load_gemini():
    import google.generativeai as genai
    from secret_key import gemini_api_key as api_key
    import os
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.0-pro')
    return model

def to_markdown(text):
    text = text.replace("•", "*")
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

if __name__ == "__main__":
    from IPython.display import Markdown
    import textwrap
    llm = load_gemini()
    result = llm.generate_content("What is the capital of India, and what is its population?")
    print((result.text))