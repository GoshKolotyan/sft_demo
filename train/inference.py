import os
import json
import random
import argparse
import torch
from tqdm import tqdm

from peft import PeftModel
from accelerate import Accelerator, PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM

from train.helpers import batched, partitioned, get_device
from train.helpers import gen_direct_qa_output_prompt, gen_clarify_q_prompt, gen_clarify_a_prompt, gen_qa_output_prompt


def generate_and_score(model, tokenizer, input_ids, attention_mask=None, temperature=None, max_length=256):
    generate_kwargs = dict(
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if attention_mask is not None:
        generate_kwargs['attention_mask'] = attention_mask
    if temperature is None:
        generate_kwargs['do_sample'] = False
    else:
        generate_kwargs['do_sample'] = True
        generate_kwargs['temperature'] = temperature

    output_ids = model.generate(input_ids, **generate_kwargs)
    # only decode newly generated tokens (exclude the prompt)
    new_token_ids = output_ids[:, input_ids.shape[1]:]
    texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    return [{'text': t} for t in texts]


def get_response(model, tokenizer, data, temperature=None, n_samples=None, max_length=256):
    prompts = [gen_direct_qa_output_prompt(qa_input=ex['question'], qa_output=None) for ex in data]
    encoded = tokenizer(prompts, padding=True, return_tensors='pt')
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    # GREEDY PRED
    predictions = generate_and_score(
        model,
        tokenizer,
        input_ids,
        attention_mask=attention_mask,
        temperature=None,
        max_length=max_length,
    )
    if predictions:
        for ex, pred in zip(data, predictions):
            ex['pred']['response'] = pred['text'].strip()
    else:
        for ex in data:
            ex['pred']['response'] = None
    # SAMPLE PREDS
    for ex in data:
        ex['pred']['response_samples'] = []
    for _ in range(n_samples):
        predictions = generate_and_score(
            model,
            tokenizer,
            input_ids,
            attention_mask=attention_mask,
            temperature=temperature,
            max_length=max_length
        )
        if predictions:
            for ex, pred in zip(data, predictions):
                ex['pred']['response_samples'].append(pred['text'].strip())


def get_clarify_question(model, tokenizer, data, temperature=None, n_samples=None, max_length=256):
    prompts = [gen_clarify_q_prompt(qa_input=ex['question'], clarify_q=None) for ex in data]
    encoded = tokenizer(prompts, padding=True, return_tensors='pt')
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    # GREEDY PRED
    for ex in data:
        ex['pred']['clarification'] = {
            'question': None,
            'answers': [],
        }
    predictions = generate_and_score(
        model,
        tokenizer,
        input_ids,
        attention_mask=attention_mask,
        temperature=None,
        max_length=max_length
    )
    if predictions:
        for ex, pred in zip(data, predictions):
            ex['pred']['clarification']['question'] = pred['text'].strip()

    # SAMPLE PREDS
    for ex in data:
        ex['pred']['clarification_samples'] = []
    for _ in range(n_samples):
        predictions = generate_and_score(
            model,
            tokenizer,
            input_ids,
            attention_mask=attention_mask,
            temperature=temperature,
            max_length=max_length
        )
        if predictions:
            for ex, pred in zip(data, predictions):
                question = pred['text'].strip()
                if not question:
                    continue
                if any(question == s['question'] for s in ex['pred']['clarification_samples']):
                    continue
                if question == ex['pred']['clarification']['question']:
                    continue
                ex['pred']['clarification_samples'].append({
                    'question': question,
                    'answers': [],
                })

def get_clarify_answers(
    model,
    tokenizer,
    qa_io_clarifications,
    max_length=256,
):
    # GREEDY PRED
    prompts = [
        gen_clarify_a_prompt(
            qa_input=qa_input,
            clarify_q=clarification['question'],
            qa_output=qa_output,
            clarify_a=None,
        )
        for qa_input, qa_output, clarification in qa_io_clarifications
    ]
    encoded = tokenizer(prompts, padding=True, return_tensors='pt')
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    predictions = generate_and_score(
        model,
        tokenizer,
        input_ids,
        attention_mask=attention_mask,
        temperature=None,
        max_length=max_length
    )
    if predictions:
        for (_, qa_output, clarification), pred in zip(qa_io_clarifications, predictions):
            clarify_a = pred['text'].strip()
            if not clarify_a:
                continue
            clarification['answers'].append({
                'answer': clarify_a,
                'response': None,
                'gold_response': qa_output,
            })


def get_qa_outputs(model, tokenizer, ex_clarification_answers, max_length=256):
    prompts = [
        gen_qa_output_prompt(
            qa_input=ex['question'],
            clarify_q=clarification['question'],
            clarify_a=answer['answer'],
            qa_output=None
        )
        for ex, clarification, answer in ex_clarification_answers
    ]
    encoded = tokenizer(prompts, truncation=True, max_length=max_length-16, padding=True, return_tensors='pt')
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    predictions = generate_and_score(
        model,
        tokenizer,
        input_ids,
        attention_mask=attention_mask,
        temperature=None,
        max_length=max_length
    )
    if predictions:
        for (_, _, answer), pred in zip(ex_clarification_answers, predictions):
            answer['response'] = pred['text'].strip()


def main(args):
    accelerator = Accelerator()
    distributed_state = PartialState()

    # OUTPUT PATH SETUP
    output_name = os.path.split(args.dataset_path)[-1][:-len('.jsonl')]
    if args.output_name:
        output_name += f'.{args.output_name}'
    if args.mode == 'respond':
        output_name += '.respond_s{}'.format(args.n_samples)
    if args.mode == 'clarify_q':
        output_name += '.clarify_q_s{}'.format(args.n_samples)
    if args.mode == 'clarify_a':
        output_name += '.clarify_a'
    if args.mode == 'qa_output':
        output_name += '.qa_output'
    if args.mode == 'eval_qa_output':
        output_name += '.eval_qa_output'
    if args.shard_total:
        output_name += f'.{args.shard_idx}of{args.shard_total}'
    output_name += '.jsonl'
    print(os.path.join(args.checkpoint, output_name))
    

    # LOADING DATA
    with open(args.dataset_path, 'r') as f:
        data = [json.loads(l) for l in f][:args.test]
    if args.shard_total:
        data = partitioned(data, args.shard_total)[args.shard_idx-1]

    data = partitioned(
        data,
        distributed_state.num_processes
    )[distributed_state.process_index]

    # LOADING MODEL AND TOKENIZER
    if args.base_model is None:
        model_map = {
            'Llama-3.2-3B': 'meta-llama/Llama-3.2-3B',
            'Llama-3.2-1B': 'meta-llama/Llama-3.2-1B',
            'gemma-2-2b': 'google/gemma-2-2b',
        }
        if args.model in model_map:
            args.base_model = model_map[args.model]
        else:
            # Assume full HF model ID was passed directly
            args.base_model = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    device = get_device()
    print(f'Using device: {device}')

    load_kwargs = {}
    if device == 'cuda':
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs['quantization_config'] = bnb_config
        load_kwargs['device_map'] = {'': distributed_state.process_index}
    else:
        load_kwargs['device_map'] = device
        load_kwargs['torch_dtype'] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **load_kwargs,
    )
    if args.merge_checkpoint:
        model = PeftModel.from_pretrained(
            model,
            args.merge_checkpoint,
            is_trainable=False,
        )
        model = model.merge_and_unload()

    if args.merge_checkpoint_2:
        model = PeftModel.from_pretrained(
            model,
            args.merge_checkpoint_2,
            is_trainable=False,
        )
        model = model.merge_and_unload()

    checkpoint_dir = os.path.join(args.checkpoint)
    if args.adapter:
        checkpoint_dir = os.path.join(checkpoint_dir, args.adapter)
    model = PeftModel.from_pretrained(
        model,
        checkpoint_dir,
        is_trainable=False,
    )

    for ex in data:
        if 'pred' not in ex:
            ex['pred'] = {}

    if args.mode == 'respond':
        all_batched_data = batched(data, args.batch_size)
        for batch_data in tqdm(
            all_batched_data,
            desc=f'Respond Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_data)
        ):
            get_response(
                model,
                tokenizer,
                batch_data,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_length=args.max_length,
            )
    elif args.mode == 'clarify_q':
        all_batched_data = batched(data, args.batch_size)
        for batch_data in tqdm(
            all_batched_data,
            desc=f'Clarify Q Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_data)
        ):
            get_clarify_question(
                model,
                tokenizer,
                batch_data,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_length=args.max_length,
            )
    elif args.mode == 'clarify_a':
        all_qa_io_clarifications = []
        for ex in data:
            if 'isambig' in ex and ex['isambig']:
                answers = set(ex['answers'])
            else:
                answers = set(ex['nq_answers'])
            for qa_output in answers:
                if ex['pred']['clarification']:
                    all_qa_io_clarifications.append(
                        (ex['question'], qa_output, ex['pred']['clarification'])
                    )
                for clarification in ex['pred']['clarification_samples']:
                    all_qa_io_clarifications.append(
                        (ex['question'], qa_output, clarification)
                    )
        all_batched_qa_io_clarifications = batched(all_qa_io_clarifications, args.batch_size)
        for batch_qa_io_clarifications in tqdm(
            all_batched_qa_io_clarifications,
            desc=f'Clarify A Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_qa_io_clarifications)
        ):
            get_clarify_answers(
                model,
                tokenizer,
                batch_qa_io_clarifications,
                max_length=args.max_length,
            )
    elif args.mode == 'qa_output':
        all_ex_clarification_answers = []
        for ex in data:
            if ex['pred']['clarification']:
                for answer in ex['pred']['clarification']['answers']:
                    all_ex_clarification_answers.append((ex, ex['pred']['clarification'], answer))
            for clarification in ex['pred']['clarification_samples']:
                for answer in clarification['answers']:
                    all_ex_clarification_answers.append((ex, clarification, answer))
        all_batched_ex_clarification_answers = batched(all_ex_clarification_answers, args.batch_size)
        for batch_ex_clarification_answers in tqdm(
            all_batched_ex_clarification_answers,
            desc=f'QA Output Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_ex_clarification_answers)
        ):
            get_qa_outputs(
                model,
                tokenizer,
                batch_ex_clarification_answers,
                max_length=args.max_length,
            )
    elif args.mode == 'eval_qa_output':
        all_ex_clarification_answers = []
        for ex in data:
            if ex['pred']['clarification']:
                for answer in ex['pred']['clarification']['eval_answers']:
                    if answer['answer']:
                        all_ex_clarification_answers.append((ex, ex['pred']['clarification'], answer))
            for clarification in ex['pred']['clarification_samples']:
                for answer in clarification['eval_answers']:
                    if answer['answer']:
                        all_ex_clarification_answers.append((ex, clarification, answer))
        all_batched_ex_clarification_answers = batched(all_ex_clarification_answers, args.batch_size)
        for batch_ex_clarification_answers in tqdm(
            all_batched_ex_clarification_answers,
            desc=f'Eval QA Output Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_ex_clarification_answers)
        ):
            get_qa_outputs(
                model,
                tokenizer,
                batch_ex_clarification_answers,
                max_length=args.max_length,
            )

    # Setup temporary output dir and clear anything from past runs
    tmp_output_dir = os.path.join(args.checkpoint, 'tmp')
    os.makedirs(tmp_output_dir, exist_ok=True)
    tmp_output_path = os.path.join(
        tmp_output_dir,
        f'{distributed_state.process_index}_{output_name}'
    )

    with open(tmp_output_path, 'w') as f:
        for ex in data:
            f.write(json.dumps(ex) + '\n')

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        all_outputs = []
        for process_idx in range(accelerator.num_processes):
            tmp_output_path = os.path.join(
                tmp_output_dir,
                f'{process_idx}_{output_name}'
            )
            with open(tmp_output_path, 'r') as f:
                all_outputs.extend(json.loads(l) for l in f)

        with open(os.path.join(args.checkpoint, output_name), 'w') as f:
            for ex in all_outputs:
                f.write(json.dumps(ex) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA PATHING
    parser.add_argument('--dataset_path')

    # EVAL SETUP
    parser.add_argument('--mode')
    parser.add_argument('--n_samples', type=int, default=0)
    parser.add_argument('--output_name', default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--shard_idx', type=int, default=None)
    parser.add_argument('--shard_total', type=int, default=None)

    # MODEL SETUP
    parser.add_argument('--model', default='Llama-3.2-3B')
    parser.add_argument('--base_model', default=None, help="HF model name/path; resolved from --model if not set")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--merge_checkpoint', default=None)
    parser.add_argument('--merge_checkpoint_2', default=None)
    parser.add_argument('--adapter', default=None)

    # EVAL HYPERPARAMETERS
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--test', type=int, default=None)

    # MISC
    parser.add_argument('--random_seed', type=int, default=88888888)

    cli_args = parser.parse_args()
    random.seed(cli_args.random_seed)

    #same for inference already default is flash attention
    main(cli_args)