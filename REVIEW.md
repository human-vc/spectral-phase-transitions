# Review of "Spectral Phase Transitions in the Loss Landscape"

## 1. Core Math Verification
**Formula:** $\gamma_\star(\delta) = \frac{4}{2+3\delta}$

*   **Verification:** The formula yields $\gamma_\star(1) = 4/5$, which matches the claim in Remark 4.3 and Figure 1.
*   **Derivation Issue:** There is a significant disconnect in the Proof of Proposition 6.1 (Section 6).
    *   The text derives a formula for $\delta < 1$ as $\gamma = \frac{2(1-\delta)}{3}$ using the equation $\frac{3\gamma}{2(1-\delta)} = 1$.
    *   This derived formula contradicts the claimed "unified" formula $\gamma_\star = \frac{4}{2+3\delta}$ at $\delta=0$ (gives $2/3$ vs $2$) and $\delta=1$ (gives $0$ vs $4/5$).
    *   The paper asserts the "unified formula" follows from the fixed-point equation but does not show the steps, and the intermediate "simple" derivation provided seems incorrect or refers to a different quantity.
    *   **Recommendation:** The proof in Section 6 needs to be corrected to show how $\frac{4}{2+3\delta}$ is derived from the self-consistent equation (Eq 14) for general $\delta$, removing the contradictory $\frac{2(1-\delta)}{3}$ derivation.

## 2. Figure Code Verification
*   **Figure 1 (`fig1_phase_boundary.pdf`):**
    *   Code checks out. The boundary is hardcoded as `4/5` (which matches $\gamma_\star(1)$).
    *   Domain/Range and plotting logic are correct.
*   **Figure 2 (`fig2_spectral_gap.pdf`):**
    *   Code checks out. It plots $\pm\sqrt{|\gamma - 0.8|}$, matching the square-root scaling law in Theorem 5.5.
*   **Figure 3 (`fig3_spurious_minima.pdf`):** (Note: filename in folder matches content described for Fig 3 in text, though label in text is `fig:isotropic`)
    *   Code checks out. Plots `4/(2 + 3*x)`, matching the main formula.

## 3. Experiment Code Verification
*   **`experiments/phase_boundary.py`:**
    *   **Correctness:** Implements `compute_gamma_star` correctly as `4.0 / (2.0 + 3.0 * delta)`.
    *   **Methodology:** Uses SGD to check for convergence. This is a valid proxy for the existence of spurious minima (if SGD fails, spurious minima likely exist).
    *   **Note:** The dataset seed is fixed (`data_seed = 10000`) outside the gamma loop. This means the phase boundary is tested on a *single* random dataset instance for each delta. While likely sufficient for high-dimensional concentration, varying the dataset seed would be more robust.

*   **`experiments/spectral_gap.py`:**
    *   **Correctness:** Implements `compute_gamma_star` correctly.
    *   **Methodology Issue (Critical):** The experiment uses SGD (`train_to_critical_point`) to find critical points. Theorem 5.5 and Figure 2 describe a "spectral gap" that becomes *negative* (saddles) for $\gamma < \gamma_\star$ scaling as $-\sqrt{\gamma_\star - \gamma}$.
        *   **Problem:** SGD naturally escapes saddle points and seeks local minima ($\lambda_{\min} \ge 0$). Therefore, this experiment is unlikely to reproduce the negative branch of the spectral gap curve shown in Figure 2. It will likely find spurious minima (with $\lambda_{\min} \ge 0$) or global minima.
        *   To verify the negative spectral gap of *saddles*, the experiment would need a saddle-finding algorithm (e.g., Newton method or specialized dynamics), not standard SGD.
    *   **Implementation Detail:** `compute_hessian_eigenvalues` reconstructs parameters assuming the order `[fc1.weight, fc2.weight]`. This is standard for `nn.Sequential` or simple modules but brittle. It appears correct for the defined `TwoLayerReLU`.

## Summary
The main formula and code are consistent with each other and the figures, **except for the proof text in Section 6** which contains a derivation that contradicts the final result. Additionally, the **spectral gap experiment** is designed to find minima (via SGD) and thus cannot empirically verify the theoretical prediction regarding the spectral gap of saddle points (negative eigenvalues) in the subcritical regime.
