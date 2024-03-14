
## Using Gemini 

### Langchain Incompatible

```python
import google.generativeai as generativeai

genai.configure(api_key=api_key)
model = genai.GenerativeModel(
    model_name = "gemini-1.0-pro"
)
response = model.generate_content(prompt)
answer = response.text
```

### Langchain Compatible

```python
import langchain_google_genai import ChatGoogleGenerativeAI
    
model = ChatGoogleGenerativeAI(
    model_name = "gemini-1.0-pro",
    google_api_key = api_key
)
model.invoke(prompt)
```

## Using OrcaMaidXL (Llama 17B)

### Downloading

```bat
mkdir OrcaMaidXL-17B-32k-GPTQ
huggingface-cli download TheBloke/OrcaMaidXL-17B-32k-GPTQ --local-dir OrcaMaidXL-17B-32k-GPTQ --local-dir-use-symlinks False
```

### Loading

```python
from langchain_community.llms.ctransformer import CTransformer
model = CTransformer(
    model = "TheBloke/OrcaMaidXL-17B-32k-GPTQ",
    model_type = "llama",
)
```