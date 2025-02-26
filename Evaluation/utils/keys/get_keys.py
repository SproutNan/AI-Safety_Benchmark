def get_openai_key():
    with open("utils/keys/openai_key.txt", "r") as f:
        return f.read()

def get_aigc2_key():
    with open("utils/keys/aigc2.txt", "r") as f:
        return f.read()

def get_deepseek_key():
    with open("utils/keys/deepseek_key.txt", "r") as f:
        return f.read()
    
def get_llama33_key():
    with open("utils/keys/llama3.3_key.txt", "r") as f:
        return f.read()
