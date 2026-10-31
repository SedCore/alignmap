# Method comparison

18,809 papers from lesswrong (11,960), arxiv (3,787), alignmentforum (3,062).

| Arm | Embedder | Clustering | Clusters | Noise | Stability (ARI) | Tag agreement (AMI) | V-measure | Tagged posts clustered | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| lsa-baseline | tfidf+lsa | no reduction → k-means (k=20) | 20 | 0.0% | 0.607 | 0.257 | 0.268 | 100.0% | 42 |
| 2022-style | sentence-transformers/allenai-specter | UMAP(5d, n_neighbors=250) → k-means (k=5) | 5 | 0.0% | 0.831 | 0.169 | 0.173 | 100.0% | 13444 |
| modern-fast | BAAI/bge-small-en-v1.5 | UMAP(10d, n_neighbors=30) → HDBSCAN (min_cluster_size=1.0% of documents) | 11 | 59.2% | 0.993 | 0.410 | 0.424 | 38.6% | 38 |

**How to read this.** *Tag agreement* is the adjusted mutual information between a method's clusters and the topic tags that LessWrong / Alignment Forum / EA Forum authors attached to their posts. The method never sees the tags, so a higher value means its clusters match human topic judgements better. AMI corrects for chance and for different numbers of clusters; it is computed on tagged posts the method placed in a cluster, so read it together with *Tagged posts clustered* and *Noise*. *Stability* is the mean adjusted Rand index between runs on random 80% subsamples; *Noise* is the share of papers a density-based method leaves unassigned. Tags are weak supervision: unevenly applied, multi-label (reduced to one), and coarser than some clusters.

Tags used as reference labels: `Interpretability (ML & AI)`, `AI Risk`, `Language Models (LLMs)`, `AI Governance`, `AI Control`, `Machine Learning  (ML)`, `AI Timelines`, `MATS Program`, `AI Evaluations`, `AI Alignment Fieldbuilding`, `GPT`, `Existential risk`, `Outer Alignment`, `Inner Alignment`, `Deceptive Alignment`, `AI Takeoff`, `Agent Foundations`, `Fiction`, `OpenAI`, `Sparse Autoencoders (SAEs)`, `Ethics & Morality`, `Reinforcement learning`, `Forecasting & Prediction`, `Logic & Mathematics`, `Corrigibility`, `AI Capabilities`, `ChatGPT`, `Economics`, `Chain-of-Thought Alignment`, `Agency`.
