import numpy as np
import torch
import scipy
import math
import sklearn
import collections

from .utils import normalize_squad, qa_metrics

from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix

# some are reused from https://github.com/AkariAsai/ATTEMPT/blob/main/attempt/metrics/metrics.py


def string_to_float(string, default=-1.0):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


# compute a f1 score from a list of strings
class F1ScoreWithInvalid(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def binary_reverse(targets):
        return ["0" if target == "1" else "1" for target in targets]

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        preds, targets = np.asarray(preds), np.asarray(targets)

        invalid_idx_mask = np.logical_and(preds != "0", preds != "1")
        preds[invalid_idx_mask] = self.binary_reverse(targets[invalid_idx_mask])

        preds, targets = torch.tensor(preds.astype(np.int32)), torch.tensor(
            targets.astype(np.int32)
        )

        conf_mat = confusion_matrix(preds, targets, num_classes=2, task="binary")
        self.tn += conf_mat[0, 0]
        self.fp += conf_mat[0, 1]
        self.fn += conf_mat[1, 0]
        self.tp += conf_mat[1, 1]

    def compute(self):
        if self.tp + self.fp == 0:
            return torch.tensor(0.0)

        if self.tp + self.fn == 0:
            return torch.tensor(0.0)

        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)

        if (precision * recall) == 0:
            return torch.tensor(0.0)

        if (precision + recall) == 0:
            return torch.tensor(0.0)

        return 100 * 2 * (precision * recall) / (precision + recall)


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        preds = np.asarray(preds)
        targets = np.asarray(targets)

        self.correct += np.sum(preds == targets)
        self.total += preds.size

    def compute(self):
        # print("internal states:", self.correct, self.total)
        return 100 * self.correct.float() / self.total


class SquadMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        if type(targets[0]) is list:
            targets = [[normalize_squad(t) for t in u] for u in targets]
        else:
            targets = [[normalize_squad(u)] for u in targets]

        preds = [normalize_squad(p) for p in preds]

        self.targets += targets
        self.preds += preds

    def compute(self):
        return qa_metrics(self.targets, self.preds)


class SpearmanCorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        targets = [string_to_float(t) for t in targets]
        preds = [string_to_float(p) for p in preds]

        self.targets += targets
        self.preds += preds

    def compute(self):
        spearman_corrcoef = 100 * scipy.stats.spearmanr(self.targets, self.preds)[0]

        if math.isnan(spearman_corrcoef):
            spearman_corrcoef = 0

        return spearman_corrcoef


class PearsonCorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        targets = [string_to_float(t) for t in targets]
        preds = [string_to_float(p) for p in preds]

        self.targets += targets
        self.preds += preds

    def compute(self):
        pearson_corrcoef = 100 * scipy.stats.pearsonr(self.targets, self.preds)[0]

        if math.isnan(pearson_corrcoef):
            pearson_corrcoef = 0

        return pearson_corrcoef


class MatthewCorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        self.targets += targets
        self.preds += preds

    def compute(self):
        return 100 * sklearn.metrics.matthews_corrcoef(self.targets, self.preds)


class MeanMulticlassF1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

        self.num_classes = 3

    def update(self, preds, targets):
        self.targets += targets
        self.preds += preds

    def compute(self):
        return 100 * sklearn.metrics.fbeta_score(
            self.targets,
            self.preds,
            beta=1,
            labels=range(self.num_classes),
            average="macro",
        )


class MultircF1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

        self.f1_score_with_invalid = F1ScoreWithInvalid()

    def update(self, preds, targets):
        self.targets += targets
        self.preds += preds

    def compute(self):
        # print(self.targets)
        # print([p["value"] for p in self.preds], [t["value"] for t in self.targets])
        return self.f1_score_with_invalid(
            [p["value"] for p in self.preds], [t["value"] for t in self.targets]
        )


class ExactMatch(Metric):
    def __init__(self, group_key="group", value_key="value"):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

        self.group_key = group_key
        self.value_key = value_key

    def update(self, preds, targets):
        self.targets += targets
        self.preds += preds

    def compute(self):
        return {"em": 100 * float(np.array_equal(self.targets, self.preds))}


class MeanGroupMetric(Metric):
    def __init__(self, group_key="group", value_key="value"):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

        self.group_key = group_key
        self.value_key = value_key
        self.metric_fn = ExactMatch()

    def update(self, preds, targets):
        self.targets += targets
        self.preds += preds

    def compute(self):
        grouped_values = collections.defaultdict(lambda: ([], []))

        for targ, pred in zip(self.targets, self.preds):
            g = targ[self.group_key]
            grouped_values[g][0].append(targ[self.value_key])
            grouped_values[g][1].append(pred[self.value_key])

        group_scores = collections.defaultdict(list)

        for targets, preds in grouped_values.values():
            for metric, score in self.metric_fn(targets, preds).items():
                group_scores[metric].append(score)

        return {metric: np.mean(scores) for metric, scores in group_scores.items()}
