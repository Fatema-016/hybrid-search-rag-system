import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from src.evaluation.gemini_judge import GeminiJudge

judge = GeminiJudge()

# obvious case
test_case = LLMTestCase(
    input="What was the average rank gain for DeepSeek-V3.2 under instructive injection?",
    actual_output="The average rank gain for DeepSeek-V3.2 under instructive injection was 4.086.",
    retrieval_context=[
        "Under the instructive injection, DeepSeek-V3.2 remains highly vulnerable, "
        "with an average rank gain of 4.086 and success rate of 85.4%."
    ],
)

metric = FaithfulnessMetric(model=judge, threshold=0.5, include_reason=True)
metric.measure(test_case)

print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
print(f"Passed: {metric.is_successful()}")