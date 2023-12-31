# Some code is adapted from https://github.com/AkariAsai/ATTEMPT/blob/main/attempt/data/tasks.py

import functools
import datasets
import numpy as np
import regex as re
import torch

from .type import AutoType

from collections import OrderedDict, defaultdict
from typing import Mapping

from metrics import (
    F1ScoreWithInvalid,
    Accuracy,
    SquadMetric,
    SpearmanCorrCoef,
    PearsonCorrCoef,
    MatthewCorrCoef,
    MeanMulticlassF1,
    MultircF1,
    MeanGroupMetric,
    ExactMatch,
)

from utils import pad_punctuation, round_stsb_target


class AbstractTask:
    name = NotImplemented
    preprocessor = NotImplemented
    formater = NotImplemented
    metrics = NotImplemented
    metric_names = NotImplemented
    config = NotImplemented
    dataset_config_name = NotImplemented
    seed = NotImplemented
    labels_list = None
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    small_datasets_without_all_splits = [
        "cola",
        "wnli",
        "rte",
        "superglue-cb",
        "superglue-copa",
        "superglue-multirc",
        "superglue-wic",
        "superglue-wsc.fixed",
        "superglue-rte",
        "mrpc",
        "stsb",
        "superglue-boolq",
        "xsum",
        "scitail",
    ]
    large_data_without_all_splits = [
        "qqp",
        "qnli",
        "superglue-record",
        "sst2",
        "squad",
        "snli",
        "anli",
        "amazon_polarity",
        "yelp_polarity",
        "winogrande",
        "newsqa",
        "searchqa",
        "triviaqa",
        "nq",
        "hotpotqa",
    ]

    def __init__(self, config, seed=42):
        self.dataset_config_name = config["dataset_config_name"][0]
        self.config = config
        self.seed = seed
        self.formater = AutoType.get(self.config["task_type"]).formater

    def postprocessor(
        self, preds, labels, tokenizer, ignore_pad_token_for_loss, info=None
    ):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        return decoded_preds, decoded_labels

    # get maximum token lenght from labels
    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def map_dataset(self, dataset, add_prefix):
        return dataset.map(
            functools.partial(self.preprocessor, add_prefix=add_prefix),
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc=f"Running {self.name}_preprocessor on dataset",
        )

    def load_dataset(self, split: int):
        return datasets.load_dataset(
            self.name, self.dataset_config_name, split=split, script_version="master"
        )

    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False):
        # to better uderstand this please see comments provided by authors https://github.com/AkariAsai/ATTEMPT/blob/main/attempt/data/tasks.py#L98
        if (
            split_validation_test
            and self.name in self.small_datasets_without_all_splits
            and split != "train"
        ):
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(
                split, dataset, validation_size=len(dataset) // 2
            )
            dataset = self.subsample(dataset, n_obs, indices)

        elif (
            split_validation_test
            and self.name in self.large_data_without_all_splits
            and split != "test"
        ):
            dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)

        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split)

            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)

        return self.map_dataset(dataset, add_prefix)


class Squad(AbstractTask):
    name = "squad"
    metrics = [SquadMetric]
    metric_names = ["SquadMetric"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example["answers"]).split("\t")
        question = pad_punctuation(example["question"])
        context = pad_punctuation(example["context"])

        input_texts = ["question:", question, "context:", context]
        label_texts = [answer] if type(answer) == str else answer

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metrics = [Accuracy, F1ScoreWithInvalid]
    metric_names = ["accuracy", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metrics = [MatthewCorrCoef, Accuracy]
    metric_names = ["matthews_correlation", "accuracy"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence:", example["sentence"]]
        label_texts = [str(example["label"])]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metrics = [Accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence", example["sentence"]]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metrics = [Accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "question:",
            example["question"],
            "sentence:",
            example["sentence"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metrics = [Accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        label_texts = [str(example["label"])]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    metrics = [Accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation_mismatched",
        "test": "validation_matched",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


# MNLI with text instead of numbers
class MNLITXT(AbstractTask):
    name = "mnli_txt"
    labels_list = ["entailment", "neutral", "contradiction"]
    label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    metrics = [SquadMetric, Accuracy]
    metric_names = ["SquadMetric", "accuracy"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation_mismatched",
        "test": "validation_matched",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        label_texts = [str(self.label_names[example["label"]])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metrics = [Accuracy, F1ScoreWithInvalid]
    metric_names = ["accuracy", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "question1:",
            example["question1"],
            "question2:",
            example["question2"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]

    metrics = [PearsonCorrCoef, SpearmanCorrCoef, Accuracy]
    metric_names = ["pearsonr", "spearmanr", "accuracy"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]

        label_texts = [str(round_stsb_target(example["label"]))]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


class SuperGLUEBoolQ(AbstractTask):
    name = "superglue-boolq"
    labels_list = ["0", "1"]
    metrics = [Accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset("super_glue", "boolq", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = ["question:", example["question"], "passage:", example["passage"]]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [MeanMulticlassF1(num_classes=3), Accuracy]
    metric_names = ["f1_multiclass", "accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset("super_glue", "cb", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        label_texts = [str(example["label"])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    labels_list = ["0", "1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [
        MultircF1,
        MeanGroupMetric(ExactMatch),
    ]
    metric_names = ["f1", "em"]

    def load_dataset(self, split):
        return datasets.load_dataset("super_glue", "multirc", split=split)

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub("<br>", " ", text)
        text = re.sub("<(/)?b>", "", text)
        return text

    def preprocessor(self, example, add_prefix=True):
        group = example["idx"]["question"]
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        input_texts = [
            "question:",
            self.remove_markup(example["question"]),
            "answer:",
            self.remove_markup(example["answer"]),
            "paragraph:",
            self.remove_markup(example["paragraph"]),
        ]
        label_texts = [str(example["label"])]
        return self.formater(
            self.name,
            input_texts,
            label_texts,
            add_prefix,
            extra_fields={"group": group},
        )


class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ["0", "1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [Accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset("super_glue", "wic", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
            "word:",
            example["word"],
        ]
        label_texts = [str(example["label"])]
        return self.formater(self.name, input_texts, label_texts, add_prefix)


class SuperGLUEWSCFixed(AbstractTask):
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    """Convert WSC examples to text2text format.
    WSC includes a sentence along with 2 'spans': the first denoting a noun and
    the other a pronoun. The 'label' specifies whether or not the pronoun is
    referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
    around the pronoun.
    For example, a typical example from WSC might look like
    {
        'text': 'This is a test sentence .',
        'span1_text': 'test',
        'span1_index': 3,
        'span2_text': 'This',
        'span2_index': 0,
        'label': 0
    }
    This example would be transformed to
    {
        'inputs': 'wsc text: # This # is a * test * sentence .',
        'targets': 'False'
    }
    """
    name = "superglue-wsc.fixed"
    labels_list = ["0", "1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [Accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset("super_glue", "wsc.fixed", split=split)

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r"^((?:\S+\s){N})(W)"
        pattern = re.sub("N", str(span_idx), pattern_tmpl)
        pattern = re.sub("W", span_str, pattern)
        return re.sub(pattern, r"\1{0} \2 {0}".format(mark), text)

    def preprocessor(self, example, add_prefix=True):
        # converts text as done in T5.
        text = example["text"]
        text = self._mark_span(text, example["span1_text"], example["span1_index"], "*")
        # Compensate for 2 added "words" added in previous step.
        span2_index = example["span2_index"] + 2 * int(
            example["span1_index"] < example["span2_index"]
        )
        text = self._mark_span(text, example["span2_text"], span2_index, "#")
        src_texts = ["text:", text]
        tgt_texts = [str(example["label"])]
        return self.formater(self.name, src_texts, tgt_texts, add_prefix)


class SuperGLUERecord(AbstractTask):
    name = "superglue-record"
    metrics = [SquadMetric]
    metric_names = ["SquadMetric"]

    def load_dataset(self, split):
        return datasets.load_dataset("super_glue", "record", split=split)

    def preprocessor(self, batch, add_prefix=True):
        new_batch = defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex["passage"]
            passage = re.sub(r"(\.|\?|\!|\"|\')\n@highlight\n", r"\1 ", passage)
            passage = re.sub(r"\n@highlight\n", ". ", passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if add_prefix:
                inputs = self.name + " " + inputs

            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["source"].extend([inputs] * num_duplicates)
            new_batch["target"].extend(ex["answers"] if num_answers > 0 else ["<unk>"])
            new_batch["task"].extend([self.name] * num_duplicates)
            new_batch["extra_fields"].extend(
                [{"answers": ex["answers"]}] * num_duplicates
            )

        return new_batch

    def map_dataset(self, dataset, add_prefix=True):
        return dataset.map(
            functools.partial(self.preprocessor, add_prefix=add_prefix),
            batched=True,
            remove_columns=dataset.column_names,
        )


TASK_MAPPING = OrderedDict(
    [
        # TODO implment all
        ("squad", Squad),
        ("mrpc", MRPC),
        ("cola", COLA),
        ("sst2", SST2),
        ("qnli", QNLI),
        ("rte", RTE),
        # ('wnli', WNLI),
        ("mnli", MNLI),
        ("mnli_txt", MNLITXT),
        ("qqp", QQP),
        ("stsb", STSB),
        ("superglue-boolq", SuperGLUEBoolQ),
        # ('superglue-rte', SuperGLUERTE),
        ("superglue-cb", SuperGLUECB),
        # ('superglue-copa', SuperGLUECOPA),
        ("superglue-multirc", SuperGLUEMultiRC),
        ("superglue-wic", SuperGLUEWIC),
        ("superglue-wsc.fixed", SuperGLUEWSCFixed),
        ("superglue-record", SuperGLUERecord),
        # ('multi_nli', MultiNLI),
        # ('snli', SNLI),
        # ('piqa', PIQA),
        # ('drop', DROP),
        # ('newsqa', Squad),
        # ('searchqa', Squad),
        # ('triviaqa', Squad),
        # ('nq', Squad),
        # ('hotpotqa', Squad),
        # ("social_i_qa", SocialIQA),
        # ("commonsense_qa", CommonsenseQA),
        # ("winogrande", WinoGrande),
        # ("scitail", SciTail),
        # ('yelp_polarity', YelpPolarity),
        # ('amazon_polarity', Amazon_Polarity),
        # ('paws', PAWS),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, config, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, seed)

        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
