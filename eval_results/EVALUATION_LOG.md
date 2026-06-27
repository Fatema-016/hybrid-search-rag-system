# Evaluation Log

## Baseline (hybrid retrieval, weights 0.6/0.4, k=5)
Faithfulness: 1.000 | Context Precision: 0.586 | Context Recall: 1.000

## A/B test: RRF weights 0.5/0.5
Faithfulness: 0.964 | Context Precision: 0.524 | Context Recall: 0.571
Conclusion: 0.6/0.4 wins on all three metrics. Kept as production default.

## Cross-encoder reranker experiment (cross-encoder/ms-marco-MiniLM-L-6-v2, wide_k=20, final_k=5)
Round 1 (eval dataset had a ground-truth bug -- bare fragments like "7.4%"
instead of full sentences):
  Faithfulness: 0.971 | Context Precision: 0.620 | Context Recall: 0.786

Round 2 (after fixing eval_dataset.py to use full-sentence ground truth):
  Faithfulness: 0.924 | Context Precision: 0.576 | Context Recall: 0.929

Conclusion: with the measurement bug fixed, reranking does NOT improve
Precision over baseline (0.576 vs 0.586) and reduces Recall (0.929 vs
1.000). Reranker is fully implemented and tested but NOT used as the
production default in the Streamlit app.

## Known retrieval/reranking gaps (confirmed via manual diagnostic + eval)
1. "What models were compared..." (DeepSeek-V3.2 / GPT-4o-mini) -- the
   chunk naming both models is never retrieved in top-5, with or without
   reranking. Root cause: the cross-encoder favors topical/abstract
   language over passages containing the specific named fact, in dense
   academic prose -- a known characteristic of MS-MARCO-trained,
   general-purpose web-search rerankers.
2. "Midjourney-10K ablation dataset" -- never retrieved in top-5 either
   way; reranker ranks it 18th of 20 candidates.
3. Bare keyword queries (e.g. "Virginia Tech") are out-of-distribution
   for the cross-encoder, which expects natural-language questions --
   confirmed by rephrasing to a full question, which fixed retrieval.

## Eval harness bugs found and fixed
1. Bare-fragment ground truth (e.g. "7.4%") gave ContextualRecallMetric
   nothing to semantically anchor on, producing a false Recall=0.00 on
   a question where the correct chunk was actually retrieved and used.
   Fixed by rewriting all expected_output entries as full sentences.
2. Judge showed possible lexical over-crediting on the Midjourney
   question (Recall scored 1.00 despite "Midjourney-10K" literally not
   appearing in the reranked top-5).


## Faithfulness sub-1.0 cases investigated (both judge pedantry, not real hallucinations)
1. "Models compared" question: judge flagged "in the context of Large
   Language Models" as unsupported, despite the retrieved chunk literally
   discussing "comparing models" and the paper's own title naming LLMs.
   The output correctly declined to guess specific model names.
2. AR-RAG frameworks question: judge flagged "two parallel frameworks" vs
   context's "separate strategies" -- a wording/paraphrase distinction,
   not a factual discrepancy (DAiD and FAiD are genuinely two methods
   either way).

Conclusion: across all manual and automated testing (Phases 3-8), the
RAG pipeline has not produced one genuine factual hallucination. Both
sub-1.0 Faithfulness scores trace to judge over-strictness, plausibly
related to disabling Qwen3's reasoning mode for TPM budget reasons.

