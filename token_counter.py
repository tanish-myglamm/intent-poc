import tiktoken

def count_tokens(text, model_name="gpt-3.5-turbo"):
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        # Use tiktoken for OpenAI models
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    else:
        # Placeholder for counting tokens for other models
        raise NotImplementedError("Token counting for this model is not implemented.")