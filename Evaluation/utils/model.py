from dataclasses import dataclass
from utils.keys.get_keys import *

@dataclass
class Config:
    api_key: str
    base_url: str

nickname_to_model_name = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4-turbo": "gpt-4-turbo",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    # these models are for evaluation
    "deepseek-v3": "deepseek-chat",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "doubao": "Doubao-1.5-pro-32k",
}

nickname_to_is_whitebox = {
    "gpt-3.5-turbo": False,
    "gpt-4-turbo": False,
    "claude-3.5-sonnet": False,
    "llama2-7b": True,
    "llama3.1-8b": True,
    "deepseek-v3": False,
    "gpt-4o": False,
    "gpt-4o-mini": False,
    "doubao": False,
}

def get_config(model_family: str):
    if model_family == "gpt":
        return Config(
            api_key = get_aigc2_key(),
            base_url = "https://api.ifopen.ai/v1"
        )
    elif model_family == "claude":
        return Config(
            api_key = get_aigc2_key(),
            base_url = "https://api.ifopen.ai/v1"
        )
    elif model_family == "deepseek":
        return Config(
            api_key = get_deepseek_key(),
            base_url = "https://api.deepseek.com/v1"
        )
    elif model_family == "doubao":
        return Config(
            api_key = get_aigc2_key(),
            base_url = "https://api.ifopen.ai/v1"
        )
    else: # other models
        return Config(
            api_key = get_aigc2_key(),
            base_url = "https://api.ifopen.ai/v1"
        )
    
