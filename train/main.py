import os
import json
import random

from datasets import Dataset
from argparse import ArgumentParser

from train.sft_train import train
from train.helpers import preprocess

def main(args):
    experiment_dir = os.path.join(
    args.output_dir,
    args.model,
    args.mode,
    args.experiment_name,
    )
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir,'config.json'), "w") as f:
        f.write(json.dumps(vars(args), indent=2))

    train_data = {}
    for path in args.train_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            train_data[os.path.split(path)[-1]] = [
                sft_ex for ex in data for sft_ex in preprocess(ex, mode=args.mode)
            ]
        # break

    # debug: uncomment to inspect first 2 training examples per dataset
    # for name, examples in train_data.items():
    #     print(f"\n{'='*60}\n[TRAIN] {name} — {len(examples)} examples\n{'='*60}")
    #     for i, ex in enumerate(examples[:2]):
    #         print(f"\n--- Example {i+1} ---")
    #         print(f"[PROMPT]\n{ex['prompt']}")
    #         print(f"[COMPLETION]{ex['completion']}")

    dev_data = {}
    for path in args.dev_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            dev_data[os.path.split(path)[-1]] = [
                sft_ex for ex in data for sft_ex in preprocess(ex, mode=args.mode)
            ]

    # debug: uncomment to inspect first 2 dev examples per dataset
    # for name, examples in dev_data.items():
    #     print(f"\n{'='*60}\n[DEV] {name} — {len(examples)} examples\n{'='*60}")
    #     for i, ex in enumerate(examples[:2]):
    #         print(f"\n--- Example {i+1} ---")
    #         print(f"[PROMPT]\n{ex['prompt']}")
    #         print(f"[COMPLETION]{ex['completion']}")

    for name, data in train_data.items():
        with open(os.path.join(experiment_dir, 'train.sft.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')

    for name, data in dev_data.items():
        with open(os.path.join(experiment_dir, 'dev.sft.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')

    train_data = Dataset.from_list([ex for data in train_data.values() for ex in data])
    dev_data = {
        name: Dataset.from_list(data) for name, data in dev_data.items()
    }

    train(args, train_data, dev_data, experiment_dir)


if __name__ == '__main__':
    parser =ArgumentParser()
    parser.add_argument('--model', default="llama3.1-base")
    parser.add_argument('--base_model', default=None, help="HF model name/path; resolved from --model if not set")
    parser.add_argument('--experiment_name')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--train_paths', nargs='+')
    parser.add_argument('--dev_paths', nargs='+')
    parser.add_argument('--test', type=int, default=None)
    parser.add_argument('--mode')
    parser.add_argument('--output_dir',default='/aua/demo-sft/')
    parser.add_argument('--random_seed', type=int, default=42)



    #general training hyperparams
    parser.add_argument('--epochs', type=float, default=5.0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    #LoRA hyperparams
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', default="none")



    cli_args = parser.parse_args()

    if cli_args.base_model is None:
        model_map = {
            'Llama-3.2-3B': 'meta-llama/Llama-3.2-3B',
        }
        if cli_args.model not in model_map:
            raise ValueError(f"Unknown model '{cli_args.model}'. Set --base_model explicitly or add to model_map.")
        cli_args.base_model = model_map[cli_args.model]

    random.seed(cli_args.random_seed)
    main(cli_args)