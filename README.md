# RAG fiscal

**Questions-réponses sourcées sur la doctrine fiscale française (BOFiP), pensées pour les conseillers en gestion de patrimoine.**

Le système répond uniquement à partir des textes du BOFiP, cite le passage exact derrière chaque affirmation,
et préfère ne pas répondre plutôt que d'inventer. Il est construit comme un service de production :
recherche hybride, reranking, garde-fous contre les hallucinations, prompts versionnés en A/B, évaluation
automatisée, supervision et traces.

> Pourquoi ce sujet : j'ai passé un an à la DGFiP, dans le service qui exploite les données fiscales pour
> sélectionner les dossiers à contrôler. La doctrine fiscale y est le texte de référence au quotidien.

## Résultats

<!-- RESULTS:START -->
Les résultats de l'évaluation sur le corpus réel sont publiés dans [eval/REPORT.md](eval/REPORT.md),
régénérés automatiquement par la CI à chaque changement du code, des prompts ou du corpus.
<!-- RESULTS:END -->

## Architecture

```mermaid
flowchart LR
    A[data.gouv.fr<br/>BOFiP en vigueur] -->|GitHub Actions| B[Corpus normalisé<br/>séries RPPM, PAT (IFI), ENR, RFPI]
    B --> C[Découpage par paragraphes<br/>numérotés + titre]
    C --> D[Vecteur dense<br/>multilingual-e5]
    C --> E[Vecteur creux<br/>BM25, IDF Qdrant]
    D & E --> F[(Qdrant)]
    Q[Question] --> G[Recherche hybride<br/>fusion RRF + filtres]
    F --> G
    G --> H[Reranking<br/>cross-encoder]
    H --> I{Score suffisant ?}
    I -- non --> X[Abstention]
    I -- oui --> J[LLM + prompt versionné<br/>A/B déterministe]
    J --> K{Citations valides ?}
    K -- oui --> R[Réponse citée]
    K -- non --> M[Mode dégradé :<br/>passages cités]
```

| Étape | Choix | Pourquoi |
|---|---|---|
| Découpage | Par paragraphes numérotés du BOFiP, recouvrement d'un paragraphe, titre en en-tête | Ne jamais couper une règle en deux ; chaque passage reste compréhensible seul |
| Recherche | Hybride : dense (sens) + creux (mots-clés), fusion RRF | Les questions fiscales mêlent vocabulaire exact (« IFI », « article 150-0 B ») et formulations libres |
| Filtres | Série du BOFiP, date de publication, tenant | Restreindre à un impôt ; isoler les documents de chaque client |
| Reranking | Cross-encoder multilingue sur 30 candidats | Meilleure précision sur les premiers résultats, et un score exploitable pour l'abstention |
| Génération | Gemini ou toute API compatible OpenAI, température 0 | Indépendance vis-à-vis du fournisseur |

## Garde-fous contre les hallucinations

1. **Abstention avant génération** : si le meilleur passage n'atteint pas le seuil de pertinence (calibré par l'évaluation), le système ne répond pas.
2. **Ancrage** : le LLM ne voit que des extraits numérotés et doit citer `[n]` après chaque affirmation, ou répondre `ABSTENTION`.
3. **Vérification des citations** : une réponse sans citation, ou qui cite un extrait inexistant, n'est jamais renvoyée. L'API bascule sur une réponse extractive, qui ne peut pas inventer.
4. **Mode dégradé** : si le LLM est indisponible (délai dépassé, quota, erreur), l'API répond quand même avec les passages cités, et une alerte se déclenche.

## LLMOps

- **Prompts versionnés** (`prompts/v1.yaml`, `prompts/v2.yaml`) et **A/B testing** déterministe : un même utilisateur garde la même version. La répartition se règle sans redéploiement (`RAG_PROMPT_AB=v1:90,v2:10`), et le retour arrière se fait en remettant `v1:100`.
- **Évaluation continue** : `eval/run_eval.py` mesure Hit@1, Hit@5 et MRR pour chaque configuration (dense, creux, hybride, hybride + reranking), l'abstention à tort et à raison, et, avec une clé LLM, le taux de réponses rejetées par la vérification des citations, par version de prompt. La CI régénère le rapport.
- **Supervision** : métriques Prometheus (latence par étape, réponses par mode et par version de prompt, abstentions, rejets de citations, score du meilleur passage), tableau de bord Grafana provisionné, alertes (taux d'abstention, mode dégradé, latence), traces OpenTelemetry vers Jaeger.

## Sécurité et multi-tenant

- Chaque client s'authentifie par une clé d'API, associée à un tenant (`RAG_API_KEYS=cle:tenant`).
- Le tenant filtre la recherche dans Qdrant : un client voit les documents publics et les siens, jamais ceux d'un autre. Ce cloisonnement est testé.
- Une passerelle OIDC (Keycloak, par exemple) peut se placer devant l'API et transmettre l'identité du client. Elle n'est pas incluse ici.
- Les questions ne sont pas journalisées : seules des métriques agrégées sont exposées.

## Démarrer

En local, sans LLM (mode extractif : l'API renvoie les passages cités). Les modèles d'embeddings et de reranking
sont téléchargés au premier lancement, puis l'index est construit en mémoire :

```bash
pip install -r requirements-dev.txt -r requirements-ml.txt
uvicorn rag_fiscal.api.main:app --app-dir src
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Quel abattement s'\''applique à la résidence principale pour l'\''IFI ?", "series": ["PAT"]}'
```

Pile complète (Qdrant, API, Prometheus, Grafana, Jaeger) :

```bash
cp .env.example .env              # renseigner RAG_LLM_PROVIDER et RAG_LLM_API_KEY pour la génération
docker compose up -d qdrant
docker compose run --rm api python scripts/build_index.py
docker compose up -d              # API :8000, Grafana :3000, Prometheus :9090, Jaeger :16686
```

Tests (encodeur sans modèle, Qdrant embarqué, aucun accès réseau) : `pytest -q`

## Structure

```
src/rag_fiscal/
  ingest/chunking.py     chargement du corpus et découpage
  index/encoders.py      vecteurs denses (e5) et creux (BM25)
  index/store.py         Qdrant : index, recherche hybride, filtres, multi-tenant
  retrieve/rerank.py     reranking (cross-encoder, lexical, aucun)
  generate/answer.py     prompts versionnés, A/B, citations, abstention, mode dégradé
  generate/llm.py        clients Gemini et compatibles OpenAI
  pipeline.py            orchestration, métriques et traces par étape
  api/main.py            FastAPI : /ask, /health, /metrics
scripts/fetch_bofip.py   téléchargement et normalisation du BOFiP (data.gouv.fr)
eval/                    questions annotées, évaluation, rapport
monitoring/              Prometheus (alertes), Grafana (tableau de bord)
```

## Limites

- Le corpus couvre les séries patrimoniales du BOFiP, soit 949 documents : revenus et plus-values de capitaux mobiliers (RPPM), patrimoine et IFI (PAT), enregistrement, successions et donations (ENR), revenus fonciers et plus-values immobilières (RFPI). Pas l'ensemble de la doctrine.
- Le jeu d'évaluation est annoté à la main et reste de taille modeste.
- Pas de mémoire de conversation ni d'agents : une question, une réponse sourcée.
- Ce n'est pas un conseil fiscal : l'outil aide à retrouver la doctrine applicable, il ne la remplace pas.

Données : BOFiP-Impôts, publications en vigueur, jeu de données public de [data.gouv.fr](https://www.data.gouv.fr/datasets/bofip-impots-publications-en-vigueur).
