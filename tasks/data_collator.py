import numpy as np
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq, DefaultDataCollator


@dataclass
class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def check_uniqueness(self, samples):
        # print(samples)
        assert len(np.unique(samples)) == 1

    def __call__(self, features):
        tasks = [d.pop("task") for d in features]
        extra_fields = [d.pop("extra_fields") for d in features if "extra_fields" in d]
        self.check_uniqueness(tasks)
        output = super().__call__(features)
        output["task"] = tasks[0]
        output["extra_fields"] = extra_fields
        return output


@dataclass
class ExtraDefaultDataCollator(DefaultDataCollator):
    # datacollator for extra fields
    def __call__(self, features):
        extra_fields = [d.pop("extra_fields") for d in features if "extra_fields" in d]
        # print(extra_fields)
        output = super().__call__(features)
        output["extra_fields"] = extra_fields
        return output
