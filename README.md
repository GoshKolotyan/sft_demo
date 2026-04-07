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

#### Clarifying Question Quality (gen_clarify_q)
| Model | Data | BLEU | ROUGE-L |
|-------|------|------|---------|
| Llama-3.2-3B | Ambiguous | 0.283 | 0.534 |
| Llama-3.2-1B | Ambiguous | 0.246 | 0.504 |
| Gemma-2-2B | Ambiguous | 0.259 | 0.513 |

#### Example Predictions (gen_clarify_q)

**Q: "Who sings with every beat of my heart?"**
| | Question |
|---|---|
| Ground Truth | Are you asking about the artist who originally recorded the song "Every Beat of My Heart" or a cover version? |
| Llama-3.2-3B | Are you asking about the original artist of the song "Every Beat of My Heart" or a cover version? |
| Llama-3.2-1B | Are you asking about the original artist of the song "Every Beat of My Heart" or a cover version? |
| Gemma-2-2B | Are you asking about the original artist or a cover version? |

**Q: "Where was tiger spy in the jungle filmed?"**
| | Question |
|---|---|
| Ground Truth | Are you asking about the specific location within India where "Tiger: Spy in the Jungle" was filmed? |
| Llama-3.2-3B | Are you asking about the country where the show was filmed or the specific location within that country? |
| Llama-3.2-1B | Are you asking about the specific location where the Tiger Spy in the Jungle movie was filmed or the general area? |
| Gemma-2-2B | Are you asking about the location where the show was filmed or the location where the show was set? |

**Q: "When did bear in the big blue house come out?"**
| | Question |
|---|---|
| Ground Truth | Are you asking about the original release date of the show or when the final episode was aired? |
| Llama-3.2-3B | Are you asking about the original release date of the show "Bear in the Big Blue House" or a specific episode? |
| Llama-3.2-1B | Are you asking about the original release date of the show or the release date of the DVD or Blu-ray version? |
| Gemma-2-2B | Are you asking about the first episode of the show or the first season? |
