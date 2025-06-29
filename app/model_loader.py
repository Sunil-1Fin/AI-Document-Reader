from langchain_community.llms import LlamaCpp  # Make sure this is imported

def load_llm():
    global llm
    if llm is None:
        llm = LlamaCpp(
            model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=20,  # set to 0 if CPU only
            use_mlock=True,
            verbose=True,
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
        )
    return llm
