# Final Assignment

**Name:** 林小蓮　**Student ID:** 111210552

> **AI Tool Used:** Claude (claude.ai) — claude-sonnet-4-6
> **How I used it:** Asking questions, understanding concepts, generating initial code, debugging
> **My contribution:** Understanding core concepts of each assignment, testing and verifying outputs, integrating and adjusting AI-generated code
> **Disclosure:** This work is not fully original. AI was involved in code generation. Full conversation logs are linked for reference.

---

## Assignment Overview

| # | Assignment | Code | AI Conversation |
|---|-----------|------|----------------|
| Midterm | Midterm Assignment | [midterm/](https://github.com/julianalidya/_ml/tree/master/midterm) | [link](https://claude.ai/share/c9970641-f2fd-46c2-ad66-f2f7f2066b5b) |
| HW1 | Hill Climbing | [hw1_hillclimbing.py](https://github.com/julianalidya/_ml/blob/master/hw1/hw1_hillclimbing.py) | [link](https://claude.ai/share/e598a238-ec27-475d-8974-9c3d86ee200f) |
| HW2 | Backpropagation | [hw2_backprop.py](https://github.com/julianalidya/_ml/blob/master/hw2/hw2_backprop.py) | [link](https://claude.ai/share/313761f6-f252-4f3f-9adf-cf367be9179a) |
| HW3 | nn0.py Example | [hw3_nn0.py](https://github.com/julianalidya/_ml/blob/master/hw3/hw3_nn0.py) | [link](https://claude.ai/share/dc511fc0-6950-47e6-826c-47a92695ee05) |
| HW4 | MicroGPT | [hw4_microgpt.py](https://github.com/julianalidya/_ml/blob/master/hw4/hw4_microgpt.py) | [link](https://claude.ai/share/9960e734-db64-4ebe-88b0-1547b8bd5ca8) |
| HW5 | Agent Security | [agent0_v2.py](https://github.com/julianalidya/_ml/blob/master/hw5/agent0_v2.py) · [agent0_v3.py](https://github.com/julianalidya/_ml/blob/master/hw5/agent0_v3.py) | [link](https://claude.ai/share/91fca7a1-ac2b-4dd5-8714-f9524cc2d108) |
| HW6 | Text Generation | [hw6_generate.py](https://github.com/julianalidya/_ml/blob/master/hw6/hw6_generate.py) | [link](https://claude.ai/share/81fed11a-cd0a-491f-a02d-a7a0606e2f74) |

---

## Assignment Details

### HW1 — Hill Climbing
Implemented the Hill Climbing algorithm to solve an optimization problem. Used AI to understand the algorithm logic, then implemented and tested it independently.

### HW2 — Manual Backpropagation
Manually computed gradients for two functions `f(x,y,z) = (x*y)+z` and `f(x,y,z,t) = ((x*y)+z)*t` using the chain rule, then verified results with code.

### HW3 — nn0.py Learning Example
Built a machine learning example using the nn0.py framework, implementing XOR neural network training with Autograd, Adam optimizer, and a full training loop.

### HW4 — MicroGPT
Based on Andrej Karpathy's microGPT, implemented a complete GPT training and inference pipeline in pure Python, including Transformer architecture, Adam optimizer, and text generation.

### HW5 — Agent with Security Controls
Extended agent0.py with security features: restricted file access to the working directory, outside-path access requires user approval, and all tool calls are reviewed by a second LLM.

### HW6 — Non-Transformer Text Generation
Trained a language model using Markov Chain (Bigram + Trigram) without Transformer/Attention, using `tw.txt` as training data to generate text.
