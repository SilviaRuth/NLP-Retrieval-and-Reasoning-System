# Error Cases and Failure Categories

The project already exposes heuristic error categories in `evaluation/error_analysis.py`. Use this file to turn raw mistakes into interview-ready examples.

## Current categories

- `negation`: premise and hypothesis disagree on negation cues.
- `lexical_overlap`: high token overlap but the relationship is still wrong.
- `long_sequence`: long examples that stress truncation or shallow pooling.
- `other`: residual failures worth manual review.

## What to capture after each benchmark run

- One false positive and one false negative per major category.
- Whether dense retrieval surfaced the right evidence before reranking.
- Whether the reranker improved rank or amplified a retrieval mistake.
- Whether robustness noise changed the predicted label or only confidence.

## Suggested write-up template

| Query / Hypothesis | Retrieved evidence | Gold | Predicted | Failure category | Takeaway |
| --- | --- | --- | --- | --- | --- |
| Example goes here | Example goes here | entailment | neutral | lexical_overlap | Reranker underweighted semantic match |

## Example investigation prompts

- Did the dense retriever miss the relevant premise entirely?
- Did the NLI model over-trust lexical overlap?
- Did truncation remove the decisive clause?
- Did typo/paraphrase noise move the prediction from entailment to neutral rather than contradiction?
