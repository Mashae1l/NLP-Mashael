# Lab Notes


## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1
- Class: Personally Identifiable Information (PII)
- Example: 0551234567 / 1023456789
- Why it matters: This information can identify or expose a person and may cause privacy and security risks
- Decision: Clean — remove or mask the personal information

### Defect 2
- Class: Duplicate words
- Example: My My Licence licence request has been under review for 15 days
- Why it matters: Repeated words add noise and may negatively affect text analysis and NLP models.
- Decision: Clean — remove accidental duplicated words while preserving the intended meaning.

### Defect 3
- Class: Spelling / Typographical errors
- Example: ألطريق المؤدي إلى حي الياسمين يحتاج صيانة عاجلة
- Why it matters: Spelling errors can make the same word appear as different tokens and reduce NLP accuracy.
- Decision: Clean — correct clear spelling mistakes.

### Defect 4
- Class: Inconsistent capitalization
- Example: THERE IS A WATER LEAK AT DAMMAM; REFERENCE BYN-2026-000019
- Why it matters: Different capitalization can create unnecessary variation in the text and affect tokenization and matching.
-Decision: Clean — normalize capitalization, especially for English text.

### Defect 5
- Class: Emotional markers / emojis
- Example: Irrigation is not working in Tahlia Street park — very frustrating 😡
- Why it matters: Emojis and emotional expressions may introduce noise, but they can also contain useful sentiment information.
- Decision: Task-dependent — remove for some classification tasks, but preserve for sentiment/emotion analysis.

### Defect 6
- Class: Grammatical errors
- Example: الحاوية ممتلئة في حي النرجس ولم تُفرغ منذ 1 أيام
- Why it matters: Incorrect grammar creates linguistic inconsistency and may affect NLP understanding and model performance.
- Decision: Clean — correct obvious grammatical errors while preserving the original meaning.

## Lab 2 — Parameter audit

| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | 177,853,440 | 51.85% | Attention: 15.95%, FFN: 31.86% |
| CAMeLBERT | 109,081,344 | 21.49% | Attention: 26.01%, FFN: 51.95% |

- Embedding-share difference: mBERT has a much larger multilingual vocabulary, so its embedding matrix consumes 51.85% of the parameters, compared with 21.49% for CAMeLBERT. This is the multilingual vocabulary tax.

## Lab 4 — Dialect audit
- Arabic rows: 7,200.
- Distribution: Gulf 4,800 (66.67%); MSA 2,400 (33.33%).
- Split evidence: Gulf appears only in training, while the Arabic validation split contains 1,200 MSA examples and no Gulf examples.
- One-sentence implication for MSA-only evaluation: Evaluating only on MSA does not measure performance on the Gulf-dialect population, even though Gulf feedback forms the majority of the Arabic dataset.

## Lab 2 — Attention findings

- The attention implementation matched PyTorch within the required tolerance of 1e-6.
- The causal attention matrix was lower triangular, with zero future-attention mass.
- This masking behaviour corresponds to decoder-style causal attention.
- The diagnostic head showed adjacent-token attention mass of 0.3136 and SEP sink mass of 0.2136.
- Padding attention decreased from 0.1729 without a mask to 0.0000 with the correct mask, confirming that the padding mask eliminates pad-attention leakage.