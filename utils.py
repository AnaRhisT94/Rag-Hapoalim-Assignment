def build_chat(tokenizer, prompt, model_name, history=None):
    if "chatglm" in model_name:
        if history is None:
            history = []
        formatted_prompt = ""
        for i, (old_query, response) in enumerate(history):
            formatted_prompt += f"[Round {i + 1}]\n\n问：{old_query}\n\n答：{response}\n\n"
        formatted_prompt += f"[Round {len(history) + 1}]\n\n问：{prompt}\n\n答："
        prompt = formatted_prompt
    elif "BLOOM" in model_name:
        prompt = f"[|Human|]:{prompt}[|AI|]:"
    elif "vicuna" in model_name or "sft" in model_name:
        system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        prompt = f"{system_message} USER: {prompt} ASSISTANT:"
    elif "llama2" in model_name or "Llama-2" in model_name or "LLaMA" in model_name:
        prompt = f"<s>[INST] {prompt} [/INST]"
    elif "Mistral" in model_name:
        prompt = f"<s>[INST] {prompt} [/INST]"
    elif "internlm" in model_name:
        prompt = f"[USER]:{prompt}<eoh>\n[Bot]:"
    else:
        prompt = f"{prompt}"
    return prompt