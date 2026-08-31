# Manuscript self-review

Branch: `simplify-single-attempt-model`. Neither review authorizes a merge to `main`.

## Standard and reference reading

The target is a short economic argument that can be understood without reading the implementation. Every major result should state its question, mechanism, conditions, evidence, and practical implication. A reader should be able to distinguish a theorem from an illustrative curve and an illustrative curve from an empirical claim.

[Philip Trammell, *Is labor a luxury in the long run?*](https://www.forethought.org/research/is-labor-a-luxury-in-the-long-run) motivates a disputed conclusion, presents a strong counterexample, and asks which assumptions make it persuasive. Its examples expose the economics before the notation. The relevant standard here is to explain what could reverse a token-demand prediction, not merely enumerate parameters that change a plot.

[Cunningham et al., *The Economics of Recursive Self-Improvement*, July 13, 2026](https://elasticity.institute/rsi-paper.pdf) builds from a simple mechanism to richer cases, separates technical derivations from the main argument, and maps the decisive elasticities to measurements. The relevant standard here is a small number of interpretable conditions with corresponding empirical tests. Its growth model and empirical estimates are not imported into this static demand model.

The comparison concerns exposition and analytical usefulness. These papers address different questions and contain empirical and conceptual work that this manuscript does not replace. Copying their length, adding parameters, or claiming a quantitative forecast would not improve this paper.

## Round 1 — economic assumptions and simplification

### Findings and revisions

1. **Participation was inconsistent at the bottom of the adoption distribution.** An unrestricted logistic assigns some work to negative-surplus AI use. Condition the same distribution on positive hurdles. This adds no parameter, makes adoption continuous at zero, and eliminates uneconomic participation. Rename the distribution's location parameter so it is not described as the median after truncation. Hold its numerical value fixed; do not retune curves to recover preferred results.

2. **The two constraints needed a common allocation interpretation.** Taking the minimum of two capacities after optimizing per unit of work is not a joint optimum. A common scarcity price for review makes policy maximize surplus after charging `w + tau` for attention. Nonbinding attention recovers the work objective; abundant opportunities with low hurdles recover the attention objective. Ties may require allocating work across policies. The plotted results remain the two polar cases; no new intermediate-regime solver or hidden claim of one is added.

3. **The strongest general result was underemphasized.** Effective inference is `eta*x`. A finite efficiency improvement changes token demand by exactly the same proportion as the equivalent price cut changes token spending. This is more useful than saying efficiency has an ambiguous effect: a price experiment can identify the sign of rebound under explicit invariance assumptions.

4. **Capability needed an operational definition.** In the numerical specification it extends technical horizons proportionally. It is not an intelligence score, and real model releases need not follow that path. Feasibility is unknown at assignment; otherwise users could screen failures without paying the modeled costs. Adoption heterogeneity concerns outside options, not hidden variation in task difficulty.

5. **Failure accounting was already sufficient.** Keep one attempt and one review. Both costs are unconditional, and failed work contributes no completed output. No extra failure penalty or additional choice is needed. Repair, learning, partial completion, and imperfect verification remain explicit limitations.

### Which ingredients earn their place?

| Ingredient | What is lost if it is removed? | Decision |
|---|---|---|
| Choice of inference intensity | Users cannot save tokens or spend more inference as prices change | Keep |
| Choice of delegated scope | Capability cannot change the amount of work supervised per hour | Keep |
| Work adoption or an equivalent extensive margin | A finite workload cannot undergo adoption takeoff and saturation | Keep a general hurdle distribution in the argument |
| Review overhead and scope-dependent review | Cannot distinguish amortizing checkpoints from review that grows with the task | Keep; state which results need each part |
| Separate feasibility and execution notation | Not needed for demand accounting or the price-efficiency result | Use one success function in the main argument; retain the flexible numerical specification in the appendix |
| Particular exponential and logistic functions | Not needed for the general price, efficiency, and uniform-review results | Confine to numerical illustrations |
| A full joint allocation, dynamic diffusion, or endogenous supply model | Needed for intermediate constraints or an aggregate time forecast, not these comparative statics | Explain the boundary; do not add these models |

The zero-overhead benchmark is especially informative: with a homogeneous review function and horizon-scaling capability, optimal scope scales with capability, inference intensity stays fixed, and attention-limited demand scales with `m^(1-beta)`. The unusual valley needs additional structure; it should not receive the same prominence as the general price-efficiency identity.

### Validation and disposition

Completed: 64 Python tests and 11 JavaScript control tests pass. The Python checks cover positive-hurdle adoption, both optimality conditions, recovery of the attention policy by charging its scarcity price, price-efficiency equivalence, uniform-review scaling, and the zero-overhead benchmark. All 18 existing plots were regenerated without moving the adoption location or selecting new numerical cases. There were no binding action bounds; 282 main, 44 focused, and 50 intervention checks agreed with independent solutions to relative objective error below `1.3e-12`. The manuscript's numerical examples remain consistent with the regenerated data.

## Round 2 — presentation, figures, and conclusions

The economic revisions and their numerical validation are complete. The initial presentation failures are clear: the main result arrives after large calibration tables, fourteen-line comparisons obscure mechanisms, some captions label shapes without explaining them, and the conclusion does not turn conditional results into measurement priorities. The second round will rewrite and inspect the paper against these specific deficiencies.
