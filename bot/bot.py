import os
import sys
import torch
import gradio as gr


from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from train.helpers import gen_clarify_q_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


base_model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
tokenizer.padding_side = "left"


model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map=DEVICE)

adapter_path = os.path.join(PROJECT_ROOT, "aua/meta-llama/Llama-3.2-3B/gen_clarify_q/Llama-3.2-3B/best_checkpoint")
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

def generate_clarifying_question(question, max_new_tokens, temperature):
    prompt = gen_clarify_q_prompt(qa_input=question)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(DEVICE)
    attention_mask = encoded.attention_mask.to(DEVICE)

    generate_kwargs = dict(
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature

    output = model.generate(input_ids, **generate_kwargs)
    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


demo = gr.Interface(
    fn=generate_clarifying_question,
    inputs=[
        gr.Textbox(label="Question", placeholder="e.g. When did bear in the big blue house come out?"),
        gr.Slider(16, 256, value=100, step=1, label="Max New Tokens"),
        gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="Temperature (0 = greedy)"),
    ],
    outputs=gr.Textbox(label="Clarifying Question"),
    title="Clarifying Question Generator",
    description="Enter an ambiguous question and the model will generate a clarifying question.",
)

demo.launch()