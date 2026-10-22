# Résultats de l'évaluation

Corpus : 949 documents BOFiP, 12310 passages. Questions : 24 dans le périmètre, 8 hors périmètre.
Modèles : `intfloat/multilingual-e5-small` (embeddings), `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (reranking).

## Recherche

| Configuration | Hit@1 | Hit@5 | MRR@10 | Latence médiane |
|---|---|---|---|---|
| dense | 38% | 83% | 0.58 | 227 ms |
| sparse | 54% | 96% | 0.68 | 420 ms |
| hybrid | 46% | 92% | 0.64 | 833 ms |
| hybrid + rerank | 71% | 92% | 0.79 | 3355 ms |

## Abstention (seuil sur le score de reranking)

- Seuil : 0.35 (seuil qui équilibre le mieux les deux erreurs sur ce jeu : 0.634, exactitude équilibrée 96%)
- Questions du périmètre où le système s'abstient à tort : 8%
- Questions hors périmètre où il s'abstient, comme attendu : 75%
- Score médian du meilleur passage : 0.98 dans le périmètre, 0.02 hors périmètre
