"""
evaluation dataset -- verified ground-truth Q&A pairs drawn
from manual testing in Phases 3-5. `known_gap=True` marks questions
where the correct chunk was confirmed NOT to surface in top-5 retrieval
during manual testing.

Design note: expected_output entries are written as FULL, self-contained
sentences rather than bare fragments (e.g. "X was 7.4%" rather than just
"7.4%"). A bare fragment like "7.4%" gave ContextualRecallMetric nothing
to semantically anchor on, causing a false Recall=0.00 on a question
where the correct chunk was actually retrieved and used -- discovered
during Phase 8 analysis when a "fixed" result still scored as a failure.
"""

EVAL_DATASET = [
    {
        "question": "What was the average rank gain for DeepSeek-V3.2 under instructive injection in a homogeneous candidate pool?",
        "expected_output": "The average rank gain for DeepSeek-V3.2 under instructive injection in a homogeneous candidate pool was 4.086.",
        "paper": "Prompt Injection in Automated Résumé Screening",
        "known_gap": False,
    },
    {
        "question": "Was the difference between HQ and LQ candidates statistically significant under descriptive injection, and what was the p-value?",
        "expected_output": "The difference between HQ and LQ candidates under descriptive injection was statistically significant, with p = 1.64x10^-17.",
        "paper": "Prompt Injection in Automated Résumé Screening",
        "known_gap": False,
    },
    {
        "question": "What models were compared in the résumé screening study, and what job role was used?",
        "expected_output": "DeepSeek-V3.2 and GPT-4o-mini were compared; the job role was IT Support Specialist.",
        "paper": "Prompt Injection in Automated Résumé Screening",
        "known_gap": True,
    },
    {
        "question": "What was GPT-4o-mini's success rate under descriptive injection in a homogeneous candidate pool?",
        "expected_output": "GPT-4o-mini's success rate under descriptive injection in a homogeneous candidate pool was 7.4%.",
        "paper": "Prompt Injection in Automated Résumé Screening",
        "known_gap": False,
    },
    {
        "question": "What does AR-RAG propose for image generation?",
        "expected_output": "AR-RAG proposes Autoregressive Retrieval Augmentation, incorporating k-nearest-neighbor retrievals at the image patch level during autoregressive generation.",
        "paper": "AR-RAG",
        "known_gap": False,
    },
    {
        "question": "What dataset was used in the AR-RAG ablation study?",
        "expected_output": "The dataset used in the AR-RAG ablation study was Midjourney-10K.",
        "paper": "AR-RAG",
        "known_gap": True,
    },
    {
        "question": "Which university is one of the AR-RAG authors affiliated with?",
        "expected_output": "One of the AR-RAG authors is affiliated with Virginia Tech.",
        "paper": "AR-RAG",
        "known_gap": False,
    },
]