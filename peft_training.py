# TODO fix autotask so that AutoTask.get() does not require config as parameter

import functools

import os
import torch

# from peft import TaskType
# from peft import get_peft_model,PromptTuningConfig

from cpeft import get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from datetime import datetime

from tasks import AutoTask, TaskDataCollatorForSeq2Seq, ExtraDefaultDataCollator
from trainer import Trainer


class PeftTraining:
    configs = None
    use_wandb = None

    def __init__(self, configs, use_wandb=True):
        self.configs = configs
        self.use_wandb = use_wandb

    def preprocess_function(
        self, examples, config, tokenizer, max_target_length, task_id=None
    ):
        padding = "max_length" if config["pad_to_max_length"] else False

        inputs = tokenizer(
            examples["source"],
            max_length=config["max_source_length"],
            padding=padding,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_target_length,
                padding=padding,
                truncation=True,
            )

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        inputs["labels"] = labels["input_ids"]
        inputs["extra_fields"] = examples["extra_fields"]

        if task_id is not None:
            inputs["task_ids"] = [task_id for _ in examples["extra_fields"]]

        return inputs

    def get_data(self, config, tokenizer):
        cols_to_remove = ["source", "target"]

        max_target_lengths = [
            AutoTask.get(dataset_name, config).get_max_target_length(
                tokenizer, default_max_length=config["max_target_length"]
            )
            for dataset_name in config["datasets"]
        ]

        config["max_target_length"] = max(max_target_lengths)
        print(
            "Max target length:",
            config["max_target_length"],
            "Chosen from:",
            max_target_lengths,
        )

        train_datasets = [
            AutoTask.get(dataset_name, config).get(
                split="train",
                split_validation_test=config["split_validation_test"],
                add_prefix=True,
                n_obs=(
                    config["max_train_samples"]
                    if "max_train_samples" in config
                    else None
                ),
            )
            for dataset_name in config["datasets"]
        ]

        # print(train_datasets)

        if config["max_train_samples"] > 0:
            for i, train_dataset in enumerate(train_datasets):
                if config["shared_attn"] is True:
                    train_datasets[i] = train_datasets[i].map(
                        functools.partial(
                            self.preprocess_function,
                            config=config,
                            tokenizer=tokenizer,
                            max_target_length=max_target_lengths[i],
                            task_id=i,
                        ),
                        batched=True,
                        load_from_cache_file=False,
                        desc="Running preprocess_function on train_dataset",
                    )
                else:
                    train_datasets[i] = train_datasets[i].map(
                        functools.partial(
                            self.preprocess_function,
                            config=config,
                            tokenizer=tokenizer,
                            max_target_length=max_target_lengths[i],
                        ),
                        batched=True,
                        load_from_cache_file=False,
                        desc="Running preprocess_function on train_dataset",
                    )

                    train_datasets[i] = train_datasets[i].remove_columns(
                        cols_to_remove + ["extra_fields"]
                    )

            
            train_dataset = concatenate_datasets(train_datasets)
        else:
            train_dataset = None

        valid_datasets = {
            dataset_name: AutoTask.get(dataset_name, config).get(
                split="validation",
                split_validation_test=config["split_validation_test"],
                add_prefix=True,
                n_obs=(
                    config["max_valid_samples"]
                    if "max_valid_samples" in config
                    else None
                ),
            )
            for dataset_name in config["datasets"]
        }

        for i, name in enumerate(valid_datasets):
            if config["shared_attn"] is True:
                valid_datasets[name] = valid_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                        task_id=i,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on valid_dataset",
                )
            else:
                valid_datasets[name] = valid_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on valid_dataset",
                )

            valid_datasets[name] = valid_datasets[name].remove_columns(cols_to_remove)

        test_datasets = {
            dataset_name: AutoTask.get(dataset_name, config).get(
                split="test",
                split_validation_test=config["split_validation_test"],
                add_prefix=True,
                n_obs=(
                    config["max_test_samples"] if "max_test_samples" in config else None
                ),
            )
            for dataset_name in config["datasets"]
        }

        for i, name in enumerate(test_datasets):
            if config["shared_attn"] is True:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                        task_id=i,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on test_dataset",
                )
            else:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on test_dataset",
                )

            test_datasets[name] = test_datasets[name].remove_columns(cols_to_remove)

        if config["pad_to_max_length"]:
            data_collator = ExtraDefaultDataCollator(return_tensors="pt")
        else:
            data_collator = TaskDataCollatorForSeq2Seq(tokenizer, return_tensors="pt")

        if config["max_train_samples"] > 0:
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=data_collator,
                batch_size=config["batch_size"],
                pin_memory=True,
            )
        else:
            train_dataloader = None

        valid_dataloaders = {
            name: DataLoader(
                valid_datasets[name],
                collate_fn=data_collator,
                batch_size=config["batch_size"],
                pin_memory=True,
            )
            for name in valid_datasets
        }

        test_dataloaders = {
            name: DataLoader(
                test_datasets[name],
                collate_fn=data_collator,
                batch_size=config["batch_size"],
                pin_memory=True,
            )
            for name in test_datasets
        }

        # print(train_dataset)

        return train_dataloader, valid_dataloaders, test_dataloaders

    # compute metrics for multi task setting
    def build_compute_metrics_fn(self, tokenizer, config):
        def reset_metrics(metrics):
            for _, m in metrics.items():
                m.reset()

        def compute_metrics_all(metrics, prefix):
            result = {}

            for n, m in metrics.items():
                # print("compute_metrics_all:", id(m))

                if "squad" in n.lower():
                    squad_m = m.compute()
                    result[f"{prefix}_em"] = squad_m["em"]
                    result[f"{prefix}_f1"] = squad_m["f1"]
                else:
                    result[f"{prefix}_{n}"] = m.compute()

            return result

        def compute_metrics(eval_preds, metrics, prefix, post_processor=None):
            preds, labels, data_info = eval_preds
            decoded_preds, decoded_labels = post_processor(
                preds,
                labels,
                tokenizer,
                ignore_pad_token_for_loss=True,
                data_info=data_info,
            )

            result = {}

            for n, m in metrics.items():
                # print("compute_metrics:", id(m))

                if "squad" in n.lower():
                    squad_m = m(decoded_preds, decoded_labels)
                    result[f"{prefix}_em"] = squad_m["em"]
                    result[f"{prefix}_f1"] = squad_m["f1"]
                else:
                    result[f"{prefix}_{n}"] = m(decoded_preds, decoded_labels)

            # print(decoded_preds, decoded_labels, result)

            return result

        def tasks_metrics(task):
            post_processor = AutoTask.get(task, config).postprocessor

            metrics = {
                n: m()
                for n, m in zip(
                    AutoTask.get(task, config).metric_names,
                    AutoTask.get(task, config).metrics,
                )
            }

            return {
                "compute_metrics": functools.partial(
                    compute_metrics,
                    metrics=metrics,
                    post_processor=post_processor,
                ),
                "compute_metrics_all": functools.partial(
                    compute_metrics_all,
                    metrics=metrics,
                ),
                "reset_metrics": functools.partial(reset_metrics, metrics=metrics),
            }

        return {task: tasks_metrics(task) for task in config["datasets"]}

    def run(self):
        for config in self.configs:
            config["output_dir"] = os.path.join(
                os.path.dirname(__file__), f"experiments/{config['output_dir']}"
            )

            from cpeft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[config["peft_type"]](
                task_type=config["task_type"],
                num_virtual_tokens=config["num_virtual_tokens"],
                prompt_init=config["prompt_init"],
            )

            if config["peft_type"] == "attempt":
                peft_config.prompt_init_embedding = config["prompt_init_embedding"]
                peft_config.prompt_embedding_paths = config["prompt_embedding_paths"]
                peft_config.attn_method = config["attn_method"]
                peft_config.prefix_num = config["prefix_num"]
                peft_config.temperature = config["temperature"]

                if config["shared_attn"]:
                    peft_config.shared_attn = config["shared_attn"]
                    peft_config.n_targets = len(config["datasets"])

            # peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=config["num_virtual_tokens"])

            for nr in range(config["n_runs"]):
                config["run"] = nr + 1

                print(f"Started {config['run']}. run...")

                model = AutoModelForSeq2SeqLM.from_pretrained(
                    config["model_name_or_path"]
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    config["tokenizer_name_or_path"],
                    model_max_length=512,
                    use_fast=True,
                )
                model.resize_token_embeddings(len(tokenizer))

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                model = get_peft_model(model, peft_config)

                # pretrained_attempt = torch.load(os.path.join(config["output_dir"], "attempt_original/MNLI/adapter_model.bin"))
                # print(pretrained_attempt, pretrained_attempt.size())
                # print(model.prompt_encoder.peft.embedding.weight)

                # model.prompt_encoder.peft.embedding.weight = pretrained_attempt
                # print(model.prompt_encoder.peft.embedding.weight)

                # model.prompt_encoder.peft.embedding.weight = torch.load("soft_prompts/sst2.bin")

                model.print_trainable_parameters()
                model.to(config["device"])
                config["timestamp"] = datetime.now().strftime("%m%d%Y%H%M%S")

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config["learning_rate"],
                    weight_decay=config.get("weight_decay", 0.01),
                )


                metrics_fn = self.build_compute_metrics_fn(tokenizer, config)
                # print(metrics_fn)

                trainer = Trainer(
                    model=model,
                    config=config,
                    dataloaders=self.get_data(config, tokenizer),
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    metrics_fn=metrics_fn,
                )

                trainer.run()

                model.print_trainable_parameters()

                print(f"Finished {config['run']}. run...")
