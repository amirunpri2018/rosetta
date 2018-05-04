import random

from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from sumeval.metrics.bleu import BLEUCalculator
from sumeval.metrics.rouge import RougeCalculator


def nltk_bleu(references, predictions):
    return corpus_bleu([[r.split()] for r in references],
                        [p.split() for p in predictions]) * 100


def sumeval_bleu(references, predictions):
    bleu = BLEUCalculator()
    bleus = [bleu.bleu(p, r) for p, r in zip(predictions, references)]
    return np.average(bleus)


def sumeval_rouge_n(references, predictions, n, alpha=0.5):
    rouge = RougeCalculator(stopwords=False, lang="en")
    rouges = [rouge.rouge_n(summary=p, references=r, n=n, alpha=alpha)
              for p, r in zip(predictions, references)]
    return np.average(rouges) * 100


def sumeval_rouge_l(references, predictions, alpha=0.5):
    rouge = RougeCalculator(stopwords=False, lang="en")
    rouges = [rouge.rouge_l(summary=p, references=r, alpha=alpha)
              for p, r in zip(predictions, references)]
    return np.average(rouges) * 100
