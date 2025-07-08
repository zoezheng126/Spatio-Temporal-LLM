import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

from evaluator.build import EVALUATOR_REGISTRY
from evaluator.ngram_metrics.bleu.bleu import Bleu
from evaluator.ngram_metrics.cider.cider import Cider
from evaluator.ngram_metrics.meteor.meteor import Meteor
from evaluator.ngram_metrics.rouge.rouge import Rouge
import os

SYSTEM_PROMPT = "Give Explanation and reasoning for your answer. Answer in detail, and be specific. Do not random guess. If you don't know say 'I don't know'."

@EVALUATOR_REGISTRY.register()
class QwenEvaluator():
    def __init__(self, exp_dir, task_name):
        self.task_name = task_name

        self.cider_scorer = Cider(n=4)
        self.bleu_scorer = Bleu(n=3)
        self.meteor_scorer = Meteor()
        self.rouge_scorer = Rouge()

        self.best_result = -np.inf

        self.save_dir = os.path.join(exp_dir,'eval_results', task_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.reset()

    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'sentence_sim': [],
            'cider': 0, 'bleu': 0, 'meteor': 0, 'rouge': 0,
        }
        self.total_count = 0
        self.save_results = []
        self.pred_sentences = []
        self.gt_sentences = []

    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
    def batch_metrics(self):
        metrics = {}
        output_pred = [item['pred'] for item in self.data]
        output_gt = [item['gt'] for item in self.data]
        batch_size = len(output_pred)

        self.pred_sentences.extend(output_pred)
        self.gt_sentences.extend(output_gt)

        # Compute sentence similarity
        embed_pred = self.sentence_model.encode(output_pred, convert_to_tensor=True)
        embed_gt = self.sentence_model.encode(output_gt, convert_to_tensor=True)
        sims = pytorch_cos_sim(embed_pred, embed_gt).diag()

        metrics['total_count'] = batch_size
        metrics['sentence_sim'] = sims.mean().item()
        metrics['target_metric'] = metrics['sentence_sim']
        return metrics

    def update(self):
        metrics = self.batch_metrics()
        batch_size = metrics['total_count']
        self.total_count += batch_size

        for item in self.data:
            item['pred'] = item['pred'].replace('assistant\n', "")
            item['query'] = item['query'].replace('<pointcloud><image>', "")
            item['query'] = item['query'].replace(SYSTEM_PROMPT, "")
            save_dict = {
                'query': item['query'],
                'response_gt': item['gt'],
                'response_pred': item['pred'],
            }
            self.save_results.append(save_dict)

        for key in self.eval_dict.keys():
            if key not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        # N-gram metrics
        gt_sentence_mp = {i: [sent] for i, sent in enumerate(self.gt_sentences)}
        pred_sentence_mp = {i: [sent] for i, sent in enumerate(self.pred_sentences)}

        self.eval_dict['cider'] = self.cider_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0]
        self.eval_dict['bleu'] = self.bleu_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0][-1]
        self.eval_dict['meteor'] = self.meteor_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0]
        self.eval_dict['rouge'] = self.rouge_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0]

        # Compute other metrics
        for k, v in self.eval_dict.items():
            if k not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[k] = sum(v) / self.total_count

        if self.eval_dict['target_metric'] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict['target_metric']
        else:
            is_best = False

        if (is_best or split == 'test') and is_main_process:
            with open(os.path.join(self.save_dir,'results.json'), 'w') as f:
                json.dump(self.save_results, f, indent=2)

        return is_best, self.eval_dict

# Example usage:
# cfg = ...  # Configuration object with exp_dir attribute

if __name__ == "__main__":
    exp_dir = "" # ! change to your exp_dir
    json_path = "" # ! change to your json path
    task_name = "" # ! change to your task name

    evaluator = QwenEvaluator(exp_dir, task_name=task_name)
    evaluator.load_data(json_path)
    evaluator.update()
    is_best, results = evaluator.record(split='val', is_main_process=True)
    print(results)
