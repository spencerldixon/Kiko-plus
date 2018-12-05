---
title: "Deep Q-Learning for Space Invaders"
layout: post
date: 2018-11-12 10:43
image: '/assets/images/'
description:
tag:
blog: false
jemoji:
author:
---

Over the last few posts we introduced the topic of Q-Learning and Deep Q-Learning in the field of reinforcement learning. We looked at how we can use Q-learning to assess the quality of what action to take given a particular state by using a Q-table to keep track of the most valuable actions at each state. We also looked at how we can replace our Q-table with a neural network to handle a larger state space by approximating our Q-values, rather than storing them for every possible state action pair.

We'll improve on our last tutorial of building a deep Q-network for the CartPole game, by throwing in a preprocessing step that allows us to learn from image data, rather than just the handy values we get back from OpenAI's gym library. We've covered convolutional neural nets before, but if you're not familiar, I would recommend brushing up on them first, as well as the past two posts on Q-Learning and Deep Q-Learning.

In this post, we'll combine deep Q-learning with convolutional neural nets, to build an agent that learns to play Space Invaders.

## Space Invaders
