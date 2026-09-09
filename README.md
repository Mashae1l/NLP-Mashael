Bayan | بيان

SDAIA— Natural Language Processing with Transformers

Student Name: Mashael Almuflih

Project Description

Bayan is a bilingual Natural Language Processing project for analysing Arabic and English citizen feedback. The project was developed progressively through the course labs, starting with text preprocessing and tokenisation, followed by Transformer models, Arabic-aware modelling, semantic search, evaluation, and CPU inference benchmarking.

Completed Labs

Lab 1 — Bilingual Preprocessing and Tokenisation

* Built a shared preprocessing pipeline for Arabic and English text.
* Implemented text cleaning, sentence segmentation, and PII masking.
* Audited and compared multiple tokenizers.
* Recorded tokenizer measurements and selected the most suitable tokenizer based on evidence.

Lab 2 — Transformer Attention and Diagnostics

* Implemented Transformer attention components.
* Inspected attention behaviour and model dimensions.
* Added parameter and shape diagnostics.
* Verified the implementation using the provided tests.

Lab 3A — Topic Classification

* Created a TF-IDF baseline for topic classification.
* Trained a Transformer-based topic classifier.
* Evaluated the classifier and compared it with the baseline.
* Saved the trained model, tokenizer, configuration, and evaluation metrics.

Lab 3B — NER and Extractive QA

* Trained a Named Entity Recognition model.
* Implemented entity extraction for bilingual text.
* Tested extractive Question Answering.
* Verified honest no-answer behaviour for questions without valid answers.

Lab 4 — Arabic Normalisation and Dialect-Aware Modelling

* Implemented Arabic text normalisation.
* Audited Modern Standard Arabic and Gulf dialect data.
* Compared Arabic and multilingual model checkpoints.
* Recorded the selected Arabic model and supporting evaluation evidence.

Lab 5 — Bilingual Semantic Search

* Built a FAISS semantic-search index containing 20,000 vectors.
* Implemented bi-encoder retrieval and cross-encoder reranking.
* Added Arabic and English query support.
* Implemented calibrated no-answer handling.
* Evaluated Recall@10, MRR@10, latency, and cross-lingual performance.

Lab 5 Results

Configuration	Recall@10	MRR@10
Bi-encoder	1.0000	1.0000
Cross-encoder reranking	1.0000	1.0000

The no-answer evaluation correctly returned empty results for all 20 no-answer queries.

Lab 6 — Evaluation Reports and Model Cards

* Implemented sliced evaluation by language, dialect, class, and text length.
* Added bootstrap confidence intervals.
* Implemented behavioural test templates for invariance, directional, and minimum-functionality tests.
* Created an error taxonomy from reviewed model errors.
* Generated the final evaluation report.
* Generated model cards for the topic classifier, Arabic dialect model, and bilingual search model.
* Passed all six Lab 6 tests.

Lab 6 Results

* Aggregate macro-F1: 0.8333
* Arabic/MSA macro-F1: 0.6000
* English macro-F1: 1.0000
* Invariance template checks: 200/200
* Directional template checks: 200/200
* Minimum-functionality template checks: 400/400

Lab 7 — Part 1: CPU Baseline Benchmark

* Added a reproducible CPU inference benchmark.
* Used four CPU threads, ten warm-up runs, and 200 timed runs.
* Measured end-to-end tokenisation and inference latency.
* Compared FP32 padded and dynamic configurations.
* Recorded p50, p99, model size, and speedup.

Lab 7 Part 1 Results

Configuration	p50 latency	p99 latency	Model size
FP32 Torch @512 padded	1621.95 ms	2769.36 ms	1.1 GB
FP32 Torch @128 dynamic	107.21 ms	204.71 ms	1.1 GB

The FP32 dynamic configuration achieved a 13.53× speedup over the padded baseline.

Main Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Sentence Transformers
* FAISS
* Pandas
* NumPy
* Scikit-learn
* Pytest

Course Attribution

This project was completed as part of the SDAIE — Natural Language Processing with Transformers course provided by SDAIA Academy.
https://github.com/SDAIAAcademy
