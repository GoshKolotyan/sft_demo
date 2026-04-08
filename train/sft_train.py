import os
import json
import torch

from peft import LoraConfig, PeftModel
from accelerate import Accelerator, PartialState
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def train(args, train_data, dev_data, experiment_dir):
    device_count = max(torch.cuda.device_count(), 1)
    per_device_batch_size = args.batch_size // device_count
    print(f'Device Count={device_count}')
    print(f'Batch Size={args.batch_size}')
    print(f'Grad Accum Steps={args.grad_accum_steps}')

    quantization_config = None
    if getattr(args, 'load_in_8bit', False):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    training_args = SFTConfig(
        output_dir=experiment_dir,
        eval_strategy='epoch',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps,
        save_strategy='epoch',
        save_total_limit=None,
        logging_steps=50,
        warmup_ratio=args.warmup_ratio,
        report_to='tensorboard',
        max_length=512,
        packing=False,
        completion_only_loss=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map={'':PartialState().process_index}
    )

    if args.checkpoint:
        model = PeftModel.from_pretrained(
            model,
            os.path.join(args.checkpoint),
            is_trainable=True
        )
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.unk_token

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        peft_config=peft_config,
    )

    accelerator = Accelerator()

    #flash attention is by default enabled so we don't need that context manager
    trainer.train()

    if accelerator.is_local_main_process:
        with open(os.path.join(experiment_dir, 'log_history.json'), 'w') as f:
            f.write(json.dumps(trainer.state.log_history, indent=2))
        trainer.model.save_pretrained(os.path.join(experiment_dir, 'best_checkpoint'))