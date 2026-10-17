# Design notes - complaint-triage

## What it does, in three sentences

It pulls real consumer complaints from the CFPB public database, sends each free-text narrative to Claude Haiku 4.5
and asks it to assign the product category, rate urgency and explain itself, then scores the answers against the
category CFPB reviewers actually assigned. On a random sample the model was 86.5 % accurate, but 84.2 % of that sample
was one category, so a "always say credit reporting" baseline would score almost the same; on a balanced sample (60
complaints per category, 9 categories, 11.1 % baseline) accuracy was 76.5 %. A second prompt with a disambiguation rule
for the biggest confusion (debt collection vs. credit reporting) did *not* help - 73.7 % - which is kept as a result.

## Why each key decision was made (plain language)

- **CFPB labels as ground truth.** The labels were assigned by people in a regulatory process before this project
  existed - an external, independently produced answer key, which is rare for LLM evaluations.
- **A closed category list.** The model must pick from the exact CFPB product list, so "correct" is unambiguous.
- **Re-sampling to a balanced set.** The first 86.5 % looked great until the majority-class baseline (84.2 %) was
  computed. The balanced set drops the baseline to 11 % and makes the accuracy number honest.
- **Keeping the failed prompt v2.** A rule that targets the top confusion made accuracy worse. Reporting that is more
  useful than hiding it: it shows prompt changes need to be measured, not assumed.
- **Haiku, not a bigger model.** Triage is a high-volume, low-latency task; the point was to measure whether the
  cheap model is good enough, and to know where it is not.

## What I'd do differently

- Report per-category precision/recall and a confusion matrix in the README, not just overall accuracy.
- Add a few-shot prompt with real examples per category and measure it the same way.
- Run three seeds / re-samples to put an error bar on the 76.5 %.
- Compare against a cheap non-LLM baseline (TF-IDF + logistic regression) trained on CFPB data; it may be competitive.

## Design questions and answers

**1. Is 76.5 % good?**
Against an 11 % baseline over nine categories it is a large lift, but not deployable as an unattended router. The
right framing is "the model can pre-sort and prioritise; a human still confirms", and the urgency score is more useful
than the category for that.

**2. Why did accuracy fall from 86.5 % to 76.5 %?**
The first sample was 84 % one category; the model was mostly right by saying the common answer. The balanced sample
removes that crutch. The lower number is the truer one.

**3. Why did the disambiguation prompt make things worse?**
It told the model to prefer "debt collection" when a collector is mentioned, which fixed some errors and created more
in the other direction. Prompt rules move errors around; only measurement shows the net effect.

**4. How do you know the model isn't just recognising CFPB's own wording?**
The narratives are consumer-written free text with redactions and typos, and CFPB's category is not in the text.
The model does see the same product vocabulary a human reviewer sees, which is the task.

**5. What would you need to put this in front of a compliance team?**
Per-category error rates, a log of every model decision with its stated reason, a human-review queue for low-confidence
or high-urgency items, and a policy for how the category list is updated when CFPB changes theirs.

## Key figures

Random sample 571 complaints: 86.5 % vs 84.2 % majority baseline · balanced 540 (60 x 9): 76.5 % vs 11.1 % · prompt v2 73.7 % ·
model claude-haiku-4-5 · data fetched live from the CFPB API, cached in data/ (gitignored).
