## DEMO Version of SFT train

### 
```bash 
uv sync
```

### 

```bash
make sft_train_start # start training
make run-tensorboard # for monitoring
make docker-build # build continer
make docker-train # run training on continer
```


### Results

#### Respond (Greedy)

| model | data | recall | f1 | em |
|-------|------|--------|----|----|
| Llama-3.2-3B | NQ-Open | 0.179 | 0.183 | 0.193 |
| Llama-3.2-3B | AmbigQA | 0.081 | 0.115 | 0.215 |

#### Respond (Sample)

| model | data | recall | f1 | em |
|-------|------|--------|----|----|
| Llama-3.2-3B | NQ-Open | 0.207 | 0.171 | 0.195 |
| Llama-3.2-3B | AmbigQA | 0.360 | 0.131 | 0.300 |

#### Clarify

| model | data | f1 | em |
|-------|------|----|----|
| Llama-3.2-3B |      |    |    |