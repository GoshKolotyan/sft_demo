## DEMO Version of SFT train

## Instructions

### Create venv
```bash 
uv sync
```

### Some commands

```bash
make sft_train_start # start training
make run-tensorboard # for monitoring
make docker-build # build continer
make docker-train # run training on continer
```

## Some knowledge

###  Paper Mapping
| Mode                 | Paper Model Name       | Learns Mapping              | Paper Reference |
|----------------------|------------------------|-----------------------------|-----------------|
| `gen_clarify_q`      | Clarify SFT            | (question) -> clarifying question | Section 4.1    |
| `gen_direct_qa_output` | Direct Ans SFT       | (question) -> answer directly | Section 4.1    |
| `gen_qa_output`      | Ans-After-Clarify SFT  | (question, clarify_q, clarify_a) -> answer | Section 4.1 |
| `gen_clarify_a`      | User Simulator SFT     | (question, clarify_q, gold_answer) -> clarifying answer | Section 2.2 |****

### Modes explained

- `gen_clarify_q` — **1 example per question** (question -> clarifying question)
- `gen_direct_qa_output` — **k examples per question** (same question repeated with each possible answer)
- `gen_qa_output` — **k examples per question** (one per clarifying answer / response pair)
- `gen_clarify_a` — **k examples per question** (one per answer / clarifying answer pair)

### Results

#### Respond (Greedy)
| model | data |mode | recall | f1 | em |
|-------|------|-----------|--------|----|----|
| Llama-3.2-3B | NQ-Open | gen_clarify_q|  |  |  |
| Llama-3.2-3B | AmbigQA | gen_clarify_q|  |  |  |

#### Respond (Sample)
| model | mode|data | recall | f1 | em |
|-------|------|------|--------|----|----|
| Llama-3.2-3B |gen_clarify_q| NQ-Open |  |  |  |
| Llama-3.2-3B | gen_clarify_q |AmbigQA |  |  |  |

#### Clarify
| model | data | mode|f1 | em |
|-------|------|----| ---- |----|
| Llama-3.2-3B | gen_clarify_q   |  |    |    |
