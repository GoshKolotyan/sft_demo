#basic setting
MODEL ?= Llama-3.2-3B
EXPERIMENT_NAME ?= Llama-3.2-3B
CHECKPOINT ?=
TRAIN_PATHS ?= data/ambigqa.train_4k.clarify.jsonl
DEV_PATHS ?= data/ambigqa.dev_4h.clarify.jsonl
TEST ?=
MODE ?= gen_clarify_q
OUTPUT_DIR ?= $(CURDIR)/aua
RANDOM_SEED ?= 42

#training hyperparams
EPOCHS ?= 5.0
LEARNING_RATE ?= 5e-5
BATCH_SIZE ?= 8
GRAD_ACCUM_STEPS ?= 1
WARMUP_RATIO ?= 0.1
WEIGHT_DECAY ?= 0.1

#LoRA hyperparams (need to be configured for each model)
LORA_R ?= 16
LORA_ALPHA ?= 32
LORA_DROPOUT ?= 0.05
LORA_BIAS ?= none

#inference settings
INFERENCE_MODE ?= respond
DATASET_PATH ?= $(DEV_PATHS)
N_SAMPLES ?= 10
MAX_LENGTH ?= 512
TEMPERATURE ?= 1.0

#start training
sft_train_start:
	python -m train.main \
	    --model $(MODEL) \
	    $(if $(EXPERIMENT_NAME),--experiment_name $(EXPERIMENT_NAME)) \
	    $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT)) \
	    $(if $(TRAIN_PATHS),--train_paths $(TRAIN_PATHS)) \
	    $(if $(DEV_PATHS),--dev_paths $(DEV_PATHS)) \
	    $(if $(TEST),--test $(TEST)) \
	    $(if $(MODE),--mode $(MODE)) \
	    --output_dir $(OUTPUT_DIR) \
	    --random_seed $(RANDOM_SEED) \
	    --epochs $(EPOCHS) \
	    --learning_rate $(LEARNING_RATE) \
	    --batch_size $(BATCH_SIZE) \
	    --grad_accum_steps $(GRAD_ACCUM_STEPS) \
	    --warmup_ratio $(WARMUP_RATIO) \
	    --weight_decay $(WEIGHT_DECAY) \
	    --lora_r $(LORA_R) \
	    --lora_alpha $(LORA_ALPHA) \
	    --lora_dropout $(LORA_DROPOUT) \
	    --lora_bias $(LORA_BIAS)
#run inference
sft_inference:
	python -m train.inference \
	    --model $(MODEL) \
	    --checkpoint $(OUTPUT_DIR)/$(MODEL)/$(MODE)/$(EXPERIMENT_NAME)/best_checkpoint \
	    --dataset_path $(DATASET_PATH) \
	    --mode $(INFERENCE_MODE) \
	    --n_samples $(N_SAMPLES) \
	    --batch_size $(BATCH_SIZE) \
	    --max_length $(MAX_LENGTH) \
	    --temperature $(TEMPERATURE)

#run metrics evaluation
sft_metrics:
	python -m train.metrics \
	    --input_path $(INPUT_PATH) \
	    --mode $(METRICS_MODE)

#for visual
run-tensorboard:
	tensorboard --logdir $(OUTPUT_DIR)/$(MODEL)/$(MODE)/$(EXPERIMENT_NAME)/
#dry-build
docker-build:
	docker build -t clarifying-questions .

#dry-run(train)
docker-train:
	docker run --gpus all clarifying-questions \
	    --model llama3.1-base \
	    --experiment_name llama3.1 \
	    --train_paths data/ambigqa.train_4k.clarify.jsonl \
	    --dev_paths data/ambigqa.dev_4h.clarify.jsonl \
	    --mode gen_direct_qa_output \
	    --output_dir $(OUTPUT_DIR)/$(MODEL)/$(MODE)/$(EXPERIMENT_NAME)/

#make run gpu drivers are installed for docker!!
docker-run-make:
	docker run --gpus all --entrypoint make clarifying-questions sft_train_start