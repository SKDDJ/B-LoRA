export TASK_NAME=cola

CUDA_VISIBLE_DEVICES=4 python run_glue_no_trainer.py \
  --model_name_or_path roberta-base \
  --wandb_name lora_BF16\
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_train_batch_size 32 \
  --learning_rate 4e-4 \
  --num_train_epochs 80 \
  --output_dir /home/wuyujia/LoRA/examples/NLU/examples/text-classification/temp/$TASK_NAME/ \
  --weight_decay 0.1\
  --seed 0 \
  --apply_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --warmup_ratio 0.06\