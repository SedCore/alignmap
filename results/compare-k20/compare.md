# Method comparison

18,809 papers from lesswrong (11,960), arxiv (3,787), alignmentforum (3,062).

| Arm | Embedder | Clustering | Clusters | Noise | Stability (ARI) | Tag agreement (AMI) | V-measure | Tagged posts clustered | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| lsa k=20 | tfidf+lsa | no reduction → k-means (k=20) | 20 | 0.0% | 0.607 | 0.257 | 0.268 | 100.0% | 5 |
| specter UMAP k=20 | sentence-transformers/allenai-specter | UMAP(5d, n_neighbors=250) → k-means (k=20) | 20 | 0.0% | 0.644 | 0.235 | 0.245 | 100.0% | 158 |
| bge-small UMAP k=20 | BAAI/bge-small-en-v1.5 | UMAP(10d, n_neighbors=30) → k-means (k=20) | 20 | 0.0% | 0.617 | 0.298 | 0.309 | 100.0% | 35 |

**How to read this.** *Tag agreement* is the adjusted mutual information between a method's clusters and the topic tags that LessWrong / Alignment Forum / EA Forum authors attached to their posts. The method never sees the tags, so a higher value means its clusters match human topic judgements better. AMI corrects for chance and for different numbers of clusters; it is computed on tagged posts the method placed in a cluster, so read it together with *Tagged posts clustered* and *Noise*. *Stability* is the mean adjusted Rand index between runs on random 80% subsamples; *Noise* is the share of papers a density-based method leaves unassigned. Tags are weak supervision: unevenly applied, multi-label (reduced to one), and coarser than some clusters.

Tags used as reference labels: `Interpretability (ML & AI)`, `AI Risk`, `Language Models (LLMs)`, `AI Governance`, `AI Control`, `Machine Learning  (ML)`, `AI Timelines`, `MATS Program`, `AI Evaluations`, `AI Alignment Fieldbuilding`, `GPT`, `Existential risk`, `Outer Alignment`, `Inner Alignment`, `Deceptive Alignment`, `AI Takeoff`, `Agent Foundations`, `Fiction`, `OpenAI`, `Sparse Autoencoders (SAEs)`, `Ethics & Morality`, `Reinforcement learning`, `Forecasting & Prediction`, `Logic & Mathematics`, `Corrigibility`, `AI Capabilities`, `ChatGPT`, `Economics`, `Chain-of-Thought Alignment`, `Agency`.
