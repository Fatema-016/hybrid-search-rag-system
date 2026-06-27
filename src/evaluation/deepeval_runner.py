"""
DeepEval evaluation runner.

For each curated Q&A pair: retrieves via the current hybrid retriever
config, generates an answer via the production RAG pipeline, then scores
Faithfulness, Contextual Precision, and Contextual Recall using Gemini
as judge. Run with different `weights` to A/B test the RRF semantic/BM25
split -- results are saved to eval_results/, tagged by config and
timestamp, for direct before/after comparison.
"""

import os
import json
from datetime import datetime

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric

from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.generation.llm_service import generate_answer
from src.evaluation.groq_judge import GroqJudge
from src.evaluation.eval_dataset import EVAL_DATASET
from src.retrieval.hybrid_retriever import get_hybrid_retriever, get_reranked_retriever

RESULTS_DIR = "eval_results"


def run_evaluation(weights: tuple = (0.6, 0.4), k: int = 5, wide_k: int = 20,
                    use_reranker: bool = False, label: str = None):
    judge = GroqJudge()
    faithfulness_metric = FaithfulnessMetric(model=judge, threshold=0.5, include_reason=True)
    precision_metric = ContextualPrecisionMetric(model=judge, threshold=0.5, include_reason=True)
    recall_metric = ContextualRecallMetric(model=judge, threshold=0.5, include_reason=True)

    if use_reranker:
        retriever = get_reranked_retriever(wide_k=wide_k, final_k=k)
    else:
        retriever = get_hybrid_retriever(k=k, weights=weights)

    results = []

    for item in EVAL_DATASET:
        question = item["question"]
        expected = item["expected_output"]

        chunks = retriever.invoke(question)[:k]
        retrieval_context = [doc.page_content for doc in chunks]

        try:
            actual_output = generate_answer(question, chunks)
        except Exception as e:
            actual_output = f"[generation error: {e}]"

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected,
            retrieval_context=retrieval_context,
        )

        faithfulness_metric.measure(test_case)
        precision_metric.measure(test_case)
        recall_metric.measure(test_case)

        results.append({
            "question": question,
            "paper": item["paper"],
            "known_gap": item["known_gap"],
            "expected_output": expected,
            "actual_output": actual_output,
            "faithfulness_score": faithfulness_metric.score,
            "context_precision_score": precision_metric.score,
            "context_recall_score": recall_metric.score,
            "faithfulness_reason": faithfulness_metric.reason,
            "context_recall_reason": recall_metric.reason,
        })

        gap_tag = " [KNOWN GAP]" if item["known_gap"] else ""
        print(f"\nQ: {question}{gap_tag}")
        print(f"  Faithfulness: {faithfulness_metric.score:.2f} | "
              f"Precision: {precision_metric.score:.2f} | "
              f"Recall: {recall_metric.score:.2f}")

    n = len(results)
    avg_f = sum(r["faithfulness_score"] for r in results) / n
    avg_p = sum(r["context_precision_score"] for r in results) / n
    avg_r = sum(r["context_recall_score"] for r in results) / n

    print(f"\n=== Summary (weights={weights}, k={k}) ===")
    print(f"Average Faithfulness:       {avg_f:.3f}")
    print(f"Average Context Precision: {avg_p:.3f}")
    print(f"Average Context Recall:    {avg_r:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = label or f"weights_{weights[0]}_{weights[1]}_k{k}"
    filepath = os.path.join(RESULTS_DIR, f"{tag}_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump({
            "weights": weights, "k": k,
            "avg_faithfulness": avg_f,
            "avg_context_precision": avg_p,
            "avg_context_recall": avg_r,
            "results": results,
        }, f, indent=2)

    print(f"\nSaved to {filepath}")
    return results

if __name__ == "__main__":
    run_evaluation(weights=(0.6, 0.4), k=5, use_reranker=True, label="reranked_wide20_k5")