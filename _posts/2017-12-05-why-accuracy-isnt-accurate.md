---
title: "Why accuracy isn't accurate"
layout: post
date: 2017-12-05 10:43
image: '/assets/images/'
description:
tag:
blog: true
jemoji:
author:
---

When it comes to measuring how well our machine learning models do, there's one metric we tend to reach for first; accuracy.

Accuracy can be thought of as the percentage of correct guesses out of our total number of things we're guessing...

```python
total_things = 100
correct_guesses = 70

accuracy_percentage = (correct_guesses / total_guesses) * 100
# Our accuracy is 70%
```

But there's a huge blind spot with accuracy as a single metric. Accuracy alone just looks at our correct guesses, but what if those were just chance? What if we had a classifier that just randomly guessed and as a result, it guessed 70 of our 100 examples correctly.

## Precision vs Recall

We need to be skeptical about accuracy, and our correct guesses on their own. If we were looking at images of cats and dogs and classifying them, how many of the images we guessed were cats, turned out to actually be cats? Did we miss any images that could've been classified as cats but weren't?

Although these two questions sound similar, take a minute to think them through and understand the difference...

- How many classification attempts were actually correct? (Precision)
- How much of the dataset did we classify correctly? (Recall)

These metrics are known as *Precision* and *Recall* and give us a better look at the performance of our model than just accuracy alone.

To understand these better we need to understand the four possible states our binary guess can be in...

- True Positives (TP): the number of positive examples, labeled correctly as positive.
- False Positives (FP): the number of negative examples, labeled incorrecly as positive.
- True Negatives (TN): the number of negative examples, labeled correctly as negative.
- False Negatives (FN): the number of positive examples, labeled incorrectly as negative.

## F1 Score

Now our model has three metrics; accuracy, precision and recall. Which one do we optimise for? Do we sacrifice precision, if we can improve recall? Guessing just a single cat picture correctly would give us a high precision (because we can demonstrate that out of all the guesses we make, we're very precise in classifying correctly), but we would have a terrible recall (because we've only classified one image out of our dataset).

Luckily we can combine precision and recall into a single score to find the best of both worlds. We'll take the harmonic mean of the two scores (We use the harmonic mean as that's best for rates and ratios). The harmonic mean calculates the average of the two scores but it also takes into account how similar the two values are. This is called the F1 score...

```python
precision = 0.84
recall = 0.72

f1_score = 2 * (precision * recall) / (precision + recall)

# Our F1 Score is 0.775
```

## Summary

Accuracy alone is a bad metric to measure our predictions by. It leaves out vital context like how many did we guess correctly by chance? How many were mislabelled and how much of the dataset did we actually predict correctly? This is where precision and recall can help us. As we may want to make a trade off between precision and recall to get a better and more balanced model, we can use the F1 score to tell us which model is better.

