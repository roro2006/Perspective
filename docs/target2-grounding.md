# Target 2: a VLM that sees what you saw when you spoke

Target 1 learns *whose* "left" you mean. Target 2 handles what rules can't:
"that one", "the one by the tape", a request with no colour. It does that by
showing a vision-language model the human camera's frame from the moment you
said the key word, with every candidate object marked by a letter, and reading
back a probability for each letter. Those probabilities are fused with the
target-1 belief, so the model can settle descriptions without overturning a
learned habit.

Work breakdown and status: [`tasks/`](../tasks/README.md) (cards T2-xx).
Serving decision: vLLM in WSL2 on the RX 7900 XT. Training: LoRA on a ~4B
Qwen-VL on the same card; rented GPUs if that doesn't fit; H100s if available.

## 1. Pipeline

```
 human cam ──► FrameHistory (T2-01) ──┐
 Whisper words ──► anchor_time (T2-02)┴► frame at the key word
                                          │
 live map + frame ──► place_marks / render_marks (T2-03): letters A, B, C on the image, cropped
                                          │
 text + marks + belief ──► build_request (T2-04): one image, one prompt, answer = one letter
                                          │
                    VllmGrounder (T2-05): max_tokens=1, guided choice, top logprobs
                                          │  P(letter) → P(object)
 belief.posterior ──► fuse (T2-06) ──► decide (target 1) ──► act / hedge / ask
```

## 2. Why letters and one token
Object ids like `red_2` tokenize into several tokens, so their probabilities
are awkward to read. Single capital letters are one token each in Qwen's
tokenizer **[VERIFY during bring-up]**, so one decoding step with
`top_logprobs` gives a distribution over every candidate at once. Decode cost
is one token; the latency is the image prefill.

## 3. Latency plan
- The vision encoder and prefill run on a frame chosen at the key word, which
  usually arrives before you finish the sentence. The request can be sent on
  the partial transcript that contains the key word and cancelled if the
  final transcript differs.
- The crop keeps image tokens low: candidates plus a margin, longest side
  ≤ 448 px.
- The arm is already moving on the target-1 decision; the grounder confirms,
  tips or redirects (fusion decides how much it may).

## 4. Fusion
`p(o) ∝ belief(o) · (floor + vlm(o))^w`. With an explicit "my left" the
belief is 1 for one object and the VLM can't move it. With a learned habit the
VLM needs real confidence to win. With no readings (an open description the
rules can't parse), the VLM alone decides. `w` and `floor` are tuned on the
eval set (T2-08).

## 5. Data and evaluation
- **Sim data** (T2-07): rendered human-view frames with marks, three kinds of
  request: `relation` (left/right, marked or not), `deictic` ("that one" while
  the person looks at it) and `color` (a unique colour). Labels are exact.
- **Real data**: the decision recorder's log plus frames (a recorder card for
  frames comes with bring-up).
- **Gate** (T2-08): a model replaces the current one only if accuracy is no
  worse on every kind and p95 latency is inside the budget.

## 6. What needs the 7900 XT
T2-09 (LoRA training) and T2-10 (serving bring-up and latency) run on that
machine. Everything else is built and tested on any laptop against the sim and
a fake server.
