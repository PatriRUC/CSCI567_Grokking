# Grokking Beyond Weight Norm

**Research question:** If models follow the **same global parameter-norm trajectory at every training step**, does changing the magnitude of parameter-direction updates alter the delay before generalization?

### 1. Reference paper
Liu, Michaud & Tegmark (ICLR 2023), [*Omnigrok: Grokking Beyond Algorithmic Data*](https://arxiv.org/abs/2210.01117). See [Appendix A](https://arxiv.org/html/2210.01117v2#A1), [Appendix D](https://arxiv.org/html/2210.01117v2#A4), and the [official MNIST notebook](https://github.com/KindXiaoming/Omnigrok/blob/main/mnist/grokking/mnist-grokking.ipynb).

### 2. Main claim
**Grokking** is delayed generalization: training performance improves well before test performance. For MNIST, the paper relates this delay to initially large parameter norms and their gradual reduction under weight decay. We ask whether **norm shrinkage alone** explains the timing.

### 3. Original methods
The MNIST setup uses a depth-3 ReLU MLP (width 200), one-hot MSE, AdamW, learning rate **0.001**, and batch size **200**. Norm and direction evolve together. In separate *algorithmic-task* analyses (Appendix D), the authors hold certain norms fixed and optimize directions; a faster direction timescale is a simplifying assumption, **not an MNIST result**.

### 4. Alternative hypothesis
**Even with identical global norm trajectories, modifying direction updates may change the time to generalization.** A positive result would show that global norm trajectories are not a sufficient explanation; it would **not** prove that direction optimization is the sole or primary bottleneck.

### 5. Experimental plan
1. **Reproduce MNIST grokking.** Start from the official notebook (784–200–200–10 ReLU MLP; 1,000 fixed training examples; one-hot MSE; AdamW with weight decay 0.01; initialization scale 8; float64). Record sample indices and seeds.
2. **Branch after training fit.** Save parameters, optimizer/RNG states, and subsequent minibatch sequence at a preset training-fit threshold, without selecting by test performance.
3. **Record reference norms.** Continue standard AdamW and store the global parameter norm **at every step**, $R_t$.
4. **Intervene only on the tangential update.** Flatten all trainable parameters, set $u_t=\theta_t/\|\theta_t\|_2$, and decompose the *actual AdamW update* $\Delta_t$:

   $$\Delta_{\parallel,t}=(u_t^\top\Delta_t)u_t,\qquad \Delta_{\perp,t}=\Delta_t-\Delta_{\parallel,t}.$$

   $$\theta^*_{t+1}=\theta_t+\Delta_{\parallel,t}+c\Delta_{\perp,t},\qquad \theta_{t+1}=R_{t+1}\frac{\theta^*_{t+1}}{\|\theta^*_{t+1}\|_2}.$$

   Compare **$c=1$** (projection/replay control), **$c=2$** (larger tangential steps), and optionally **$c=0.5$**, using identical minibatches and retained AdamW state. Verify $c=1$ reproduces the reference before interpreting other groups.
5. **Evaluate.** Across ≥3 paired seeds (ideally 5), track train/test loss and accuracy, norm-matching error, actual angular change, and steps from the checkpoint to a prespecified generalization threshold. Report censored runs that never reach the threshold. Fix settings using validation data, not test results.

**Interpretation limit:** Matching *global* norm does not match layerwise norms or representations; changing tangential steps also changes later gradients and optimizer trajectories.

### 6. Progress
- [x] Reproduced **Figure 3** from *Omnigrok*.
- [ ] Implement and verify per-step norm replay ($c=1$).
- [ ] Run direction-intervention groups and paired-seed comparisons.
- [ ] Analyze generalization delays and robustness.
