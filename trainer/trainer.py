import torch
import os
import wandb

from transformers import get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM

from cpeft import PeftModel

from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        config,
        dataloaders,
        optimizer,
        tokenizer,
        metric_fs,
        compute_metrics,
        wandb=True,
    ):
        self.min_eval_loss = torch.inf
        self.model = model
        self.config = config
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = dataloaders
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.metric_fs = metric_fs
        self.compute_metrics = compute_metrics
        self.metrics = {}
        self.wandb = wandb

        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=(len(self.train_dataloader) * config["num_epochs"]),
        )

        if self.wandb:
            self.wandb_run = self.init_wandb()

    def init_wandb(self):
        return wandb.init(
            project=self.config["wandb_project"],
            config=self.config,
            name=f"{self.config['model_name_or_path']}_{'_'.join(self.config['datasets'])}_{self.config['timestamp']}_{self.config['run']}",
        )

    def compute_metrics_all(self, prefix):
        result_m = {}
        for n, m in self.metric_fs.items():
            if "squad" in n.lower():
                result_m[f"{prefix}_em"] = m["em"].compute()
                result_m[f"{prefix}_f1"] = m["f1"].compute()
            else:
                result_m[f"{prefix}_{n}"] = m.compute()

        return result_m

    def reset_metrics(self):
        for n in self.metric_fs:
            self.metric_fs[n].reset()

    def train(self):
        self.model.train()
        train_loss = 0
        metrics = {}
        max_new_tokens = self.config["max_target_length"]

        for _, batch in enumerate(tqdm(self.train_dataloader)):
            # batch = {k: v.to(config["device"]) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.config["device"]),
                labels=batch["labels"].to(self.config["device"]),
                attention_mask=batch["attention_mask"].to(self.config["device"]),
            )

            preds = self.model.generate(
                input_ids=batch["input_ids"].to(self.config["device"]),
                labels=batch["labels"].to(self.config["device"]),
                attention_mask=batch["attention_mask"].to(self.config["device"]),
                max_new_tokens=max_new_tokens,
            )

            loss = outputs.loss
            train_loss += loss.detach().float()
            metrics = self.compute_metrics(
                preds.cpu(), batch["labels"].cpu(), self.tokenizer, self.config, "train"
            )

            # print(metrics)

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            self.optimizer.zero_grad()

        metrics = self.compute_metrics_all("train")
        metrics.update({"train_loss": train_loss.cpu() / len(self.train_dataloader)})
        metrics.update({"train_ppl": torch.exp(metrics["train_loss"])})

        self.reset_metrics()

        return metrics

    def valid(self):
        self.model.eval()
        valid_loss = 0
        metrics = {}
        max_new_tokens = self.config["max_target_length"]

        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.valid_dataloader)):
                # batch = {k: v.to(config["device"]) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.config["device"]),
                    labels=batch["labels"].to(self.config["device"]),
                    attention_mask=batch["attention_mask"].to(self.config["device"]),
                )

                preds = self.model.generate(
                    input_ids=batch["input_ids"].to(self.config["device"]),
                    labels=batch["labels"].to(self.config["device"]),
                    attention_mask=batch["attention_mask"].to(self.config["device"]),
                    max_new_tokens=max_new_tokens,
                )

                loss = outputs.loss
                valid_loss += loss.detach().float()
                metrics = self.compute_metrics(
                    preds.cpu(),
                    batch["labels"].cpu(),
                    self.tokenizer,
                    self.config,
                    "valid",
                )

        metrics = self.compute_metrics_all("valid")
        metrics.update({"valid_loss": valid_loss.cpu() / len(self.valid_dataloader)})
        metrics.update({"valid_ppl": torch.exp(metrics["valid_loss"])})

        self.reset_metrics()

        return metrics

    def test(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config["model_name_or_path"])
        model = PeftModel.from_pretrained(model, self.config["best_model_path"])

        model.to(self.config["device"])
        model.eval()
        valid_loss = 0
        metrics = {}

        max_new_tokens = self.config["max_target_length"]

        for _, batch in enumerate(tqdm(self.test_dataloader)):
            # batch = {k: v.to(config["device"]) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"].to(self.config["device"]),
                    labels=batch["labels"].to(self.config["device"]),
                    attention_mask=batch["attention_mask"].to(self.config["device"]),
                )

            preds = model.generate(
                input_ids=batch["input_ids"].to(self.config["device"]),
                labels=batch["labels"].to(self.config["device"]),
                attention_mask=batch["attention_mask"].to(self.config["device"]),
                max_new_tokens=max_new_tokens,
            )

            loss = outputs.loss
            valid_loss += loss.detach().float()
            metrics = self.compute_metrics(
                preds.cpu(), batch["labels"].cpu(), self.tokenizer, self.config, "test"
            )

        metrics = self.compute_metrics_all("test")
        metrics.update({"test_loss": valid_loss.cpu() / len(self.test_dataloader)})
        metrics.update({"test_ppl": torch.exp(metrics["test_loss"])})

        self.reset_metrics()

        return metrics

    def run(self):
        for epoch in range(self.config["num_epochs"]):
            self.metrics.update(self.train())
            self.metrics.update(self.valid())

            if self.metrics["valid_loss"] < self.min_eval_loss:
                self.min_eval_loss = self.metrics["valid_loss"]
                artifact_name = f"{'_'.join(self.config['datasets'])}_{self.config['timestamp']}_{self.config['run']}"
                checkpoint_name = os.path.join(
                    os.path.dirname(__file__),
                    f"{self.config['output_dir']}/{artifact_name}",
                )
                self.model.save_pretrained(checkpoint_name)
                self.config["best_model_path"] = checkpoint_name

                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="weights",
                )

                artifact.add_dir(local_path=checkpoint_name)
                self.wandb_run.log_artifact(artifact)

            wandb.log(self.metrics)
            print(f"{epoch=},", self.metrics)

        test_metrics = self.test()
        self.metrics.update(test_metrics)

        wandb.log(test_metrics)
        print("Test: ", test_metrics)
        print("Run metrics: ", self.metrics)

        if self.wandb_run:
            self.wandb_run.finish()
