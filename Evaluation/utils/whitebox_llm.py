from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.model import nickname_to_model_name

def load_whitebox_llm(model_nickname: str, low_cpu_mem_usage: bool = False, device="cuda"):
    model_name = nickname_to_model_name[model_nickname]
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer