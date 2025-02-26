import requests
from utils.model import get_config, nickname_to_model_name

MAX_LENGTH = 2048

def completion_with_whitebox_llm(model, tokenizer, model_nickname: str, prompt: str, max_new_tokens=MAX_LENGTH) -> str:
    if model_nickname == "llama2-7b":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        return completion
    elif model_nickname == "llama3.1-8b":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        return completion
    else:
        raise NotImplementedError(f"Model {model_nickname} is not supported for whitebox inference.")
    
def completion_with_blackbox_llm(model_nickname: str, prompt: str, max_new_tokens=MAX_LENGTH, system: str=None) -> str:
    if model_nickname.startswith("gpt"):
        config = get_config("gpt")
    elif model_nickname.startswith("claude"):
        config = get_config("claude")
    elif model_nickname.startswith("deepseek"):
        config = get_config("deepseek")
    else:
        config = get_config("other")

    model_name = nickname_to_model_name[model_nickname]

    messages = [
        {"role": "system", "content": system or "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    response = requests.post(
        url=f"{config.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}"},
        json={
            "messages": messages, 
            "model": model_name, 
            "max_new_tokens": max_new_tokens,
            "do_sample": False
        }
    )

    if response.status_code != 200:
        raise Exception(f"Failed to get completion from {model_nickname}: {response.json()}")
    return response.json()["choices"][0]["message"]["content"]
    
def auto_completion(model_nickname: str, prompt: str, force_blackbox=False, model=None, tokenizer=None, max_new_tokens=MAX_LENGTH) -> str:
    model_name = nickname_to_model_name[model_nickname]

    if "llama" in model_name and not force_blackbox:
        assert model and tokenizer, "model and tokenizer must be provided for llama whitebox models"
        return completion_with_whitebox_llm(model, tokenizer, model_nickname, prompt, max_new_tokens=max_new_tokens)
    else:
        return completion_with_blackbox_llm(model_nickname, prompt, max_new_tokens=max_new_tokens)
