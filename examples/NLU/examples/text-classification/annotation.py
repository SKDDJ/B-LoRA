""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
# 使用🤗 Transformers模型对GLUE数据集进行序列分类的微调。

import argparse
# 导入argparse库，用于解析命令行参数。

import logging
# 导入logging库，用于记录日志。

import math
# 导入math库，用于数学运算。

import os
# 导入os库，用于处理文件和目录。

import random
# 导入random库，用于生成随机数。

import torch
# 导入torch库，PyTorch的主要库，用于深度学习。

import datasets
# 导入datasets库，用于加载和处理数据集。

from datasets import load_dataset, load_metric
# 从datasets库导入load_dataset和load_metric函数，用于加载数据集和评估指标。

from torch.utils.data.dataloader import DataLoader
# 从torch.utils.data.dataloader导入DataLoader，用于创建数据加载器。

from tqdm.auto import tqdm
# 从tqdm.auto导入tqdm，用于显示进度条。

import transformers
# 导入transformers库，用于访问预训练模型和工具。

from accelerate import Accelerator
# 从accelerate库导入Accelerator，用于加速模型训练。

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
# 从transformers库导入多个工具和类，包括优化器、配置、模型、分词器、数据整理器、调度器等。

logger = logging.getLogger(__name__)
# 初始化日志记录器。

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
# 定义不同任务对应的输入键值。

def parse_args():
    # 定义一个解析命令行参数的函数。
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    # 创建一个ArgumentParser对象，用于处理命令行参数。
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    # 添加一个命令行参数，用于指定GLUE任务的名称。
    # 省略剩余的parser.add_argument代码块，这些块为脚本添加不同的命令行选项。
    # ...
    
    args = parser.parse_args()
    # 解析命令行输入的参数。

    # Sanity checks
    # 进行一些基本的检查。
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    # 如果没有指定任务名称或训练/验证文件，则抛出异常。
    else:
        # 检查训练和验证文件的扩展名是否正确。
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    # 如果指定了输出目录，则创建该目录。

    return args
    # 返回解析得到的参数。
    
    
    
    
def main():
    # 定义主函数。
    args = parse_args()
    # 解析命令行参数。

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # 初始化加速器。在这个例子中，我们让加速器处理设备放置。
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    # 为调试配置在每个进程上记录一次日志。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # 设置日志，我们只希望每台机器上有一个进程在屏幕上记录日志。
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # 如果提供了，现在设置训练种子。
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    # 获取数据集：你可以提供自己的CSV/JSON训练和评估文件（见下文），或者指定一个GLUE基准任务（数据集将自动从datasets Hub下载）。

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    # 对于CSV/JSON文件，此脚本将使用名为'label'的列作为标签，如果存在这样的列，则使用名为'sentence1'和'sentence2'的列作为句子对，
    # 或者如果提供了至少两列，则使用前两列（不命名为'label'的列）。

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    # 如果CSV/JSON文件只包含一个非标签列，则脚本将对这个单独的列进行单句分类。你可以很容易地调整这个行为（见下文）。

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # 在分布式训练中，load_dataset函数保证只有一个本地进程可以同时下载数据集。
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        # 从Hub下载并加载数据集。
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        # 从本地csv或json文件加载数据集。
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # 有关加载任何类型的标准或自定义数据集的更多信息，请访问https://huggingface.co/docs/datasets/loading_datasets.html。

    # Labels
    # 标签处理部分。
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # 尝试在这里有好的默认设置，根据需要随时调整。
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # 加载预训练模型和分词器。
    # 在分布式训练中，.from_pretrained方法保证只有一个本地进程可以同时下载模型和词汇表。
    config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                        num_labels=num_labels, 
                                        finetuning_task=args.task_name,
                                        apply_lora=args.apply_lora,
                                        lora_alpha=args.lora_alpha,
                                        lora_r=args.lora_r,
                                       )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )


    # Preprocessing the datasets
    # 预处理数据集
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        # 我们尝试有一些好的默认设置，但不要犹豫根据您的用例进行调整。
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    # 一些模型已经设置了标签的使用顺序，让我们确保我们使用它。
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        # 一些配置中的标签是大写，有些不是。
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        # 分词处理文本
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                # 将标签映射到ID（GLUE任务不需要）
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                # 在所有情况下，将列重命名为labels，因为模型会期望这样。
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    # 记录训练集中的一些随机样本：
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    # 创建DataLoaders：
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        # 如果已经对最大长度进行了填充，我们使用默认的data collator，它会将所有内容转换为张量。
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        # 否则，`DataCollatorWithPadding`将为我们应用动态填充（通过填充到传递的样本的最大长度）。当使用混合精度时，我们添加`pad_to_multiple_of=8`
        # 来将所有张量填充为8的倍数，这将启用在具有计算能力>=7.5（Volta）的NVIDIA硬件上使用Tensor Cores。
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    # 创建训练数据加载器，参数包括数据集、是否随机打乱、数据整理函数和每设备批处理大小。

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    # 创建评估数据加载器，参数包括数据集、数据整理函数和每设备批处理大小。

    for name, params in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
    # 遍历模型的所有参数，对于包含'lora_A'或'lora_B'的参数启用梯度，其余参数禁用梯度。

    # Optimizer
    # 配置优化器
    # Split weights in two groups, one with weight decay and the other not.
    # 将权重分为两组，一组使用权重衰减，另一组不使用。
    no_decay = ["bias", "LayerNorm.weight"]
    lora_param_names = [n for n, p in model.named_parameters() if 'lora_A' in n or 'lora_B' in n]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in lora_param_names],
            "weight_decay": args.weight_decay,  # 或者你希望为LoRA参数设定的其他值
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in lora_param_names and n not in no_decay],
            "weight_decay": 0.0,  # 非LoRA参数的权重衰减
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # 使用AdamW优化器，配置不同的参数组。

    # Prepare everything with our `accelerator`.
    # 使用`accelerator`准备所有内容。
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)
    # 注意 -> 在下面获取训练数据加载器的长度之前需要准备好它（因为在多进程中它的长度会更短）。

    # Scheduler and math around the number of training steps.
    # 计划器和围绕训练步骤数量的数学计算。
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # 根据每个epoch的更新步骤数和总训练步骤计算总训练epoch数。

    warmup_ratio = 0.06
    n_steps = len(train_dataloader) * args.num_train_epochs
    warmup_steps = warmup_ratio * n_steps
    # 计算预热步骤数。

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # 配置学习率计划器。

    # Get the metric function
    # 获取评估指标函数
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)

    # Train!
    # 开始训练！
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # 计算总批处理大小。

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # 记录训练过程的相关信息。

    # Only show the progress bar once on each machine.
    # 每台机器上只显示一次进度条。
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        # 设置模型为训练模式。
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            # 计算损失并进行反向传播。

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            # 更新优化器、学习率计划器，并重置梯度。

            if completed_steps >= args.max_train_steps:
                break
        # 完成所有训练步骤后退出训练循环。

        model.eval()
        # 设置模型为评估模式。
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
        # 在评估数据上运行模型并收集预测结果。

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
    # 计算并记录每个epoch的评估指标。

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    # 如果指定了输出目录，保存训练好的模型。

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        # 在不匹配的验证集上进行最终评估
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        # 设置模型为评估模式。
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
        # 在不匹配的验证数据上运行模型并收集预测结果。

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")
    # 计算并记录mnli任务的不匹配验证集的评估指标。

if __name__ == "__main__":
    main()
    # 如果脚本是作为主程序运行，则执行main函数。
