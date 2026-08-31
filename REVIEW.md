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

### Revisions after the second review

1. **Put the question and answers before the machinery.** The opening contrasts a finite reporting workload with a backlog of candidate designs. It states the three conclusions before introducing notation. The result sequence is price, efficiency, capability, then verification: a monotone price benchmark makes the later ambiguous quantity responses easier to understand.

2. **Separate the economic core from the numerical specification.** The main argument uses one success function, a review function, and an adoption distribution. Exponential hazards, the truncated logistic, shape parameters, first-order conditions, and allocation details move to Appendix A. The two policy choices remain unchanged.

3. **Show the mechanism behind a curve.** Four new main figures reuse the already audited outcomes. The price view separates adoption, purchases, and spending. The two capability views separate activity from tokens per work unit and show their product. The verification view separates token use from completed work. Each plot has at most three case curves, an explicit baseline, and a stated resource constraint. The full parameter comparisons remain in Appendix B.

4. **Reduce the emphasis on selected shapes.** The valley moves to the supplement with a statement of its extra assumptions and nearby-parameter checks. There is no claim that the examples establish industry frequencies. Exact numerical readouts support reproducibility in the appendix; rounded magnitudes suffice for the main prose.

5. **Make conclusions operational.** The paper identifies three measurements: the spending response to price, review growth with delegated scope, and the response to additional review capacity. These connect each conclusion to an observation or experiment. A declining token bill is separated from declining economic usefulness, and token purchases are separated from supplier revenue.

6. **Read the rewritten argument for remaining ambiguity.** Clarify that failed work produces no completion during the modeled period; future repair is excluded. Distinguish pure price changes from efficiency changes. State that the attention-value elasticity requires the horizon-scaling capability path. Rename the parameter-table columns so “high” unambiguously means a high constraint or difficulty, rather than high capability. Clarify that beta describes the variable review component, while theta is total review elasticity.

7. **Inspect the reading edition, not just the source.** Use complete numeric labels on logarithmic axes so 0.2 cannot appear as an ambiguous “2.” Preserve independent axes for outcomes with different denominators. Fix the linked notebook and review record so a packaged reading edition contains its advertised reproducibility files without exposing arbitrary checkout files.

### Claims and evidence check

| Claim | What supports it | What would invalidate the stated application? |
|---|---|---|
| Cheaper tokens weakly raise purchases | Revealed preference under the appropriate objective; monotonicity audits | Prices also change quality, feasible choices, fees, or other inputs |
| Efficiency response equals the equivalent spending response | Change of variables in either objective; independent numerical equivalence tests | Efficiency affects feasibility or review separately, or relevant action bounds bind |
| Uniformly faster review scales attention-limited use | Multiplicative capacity change leaves the conditional policy unchanged; throughput checks | Work runs out, the attention constraint no longer binds, or operation is uneconomic |
| Capability can raise or lower demand | Audited examples expose both activity and intensity; their product is verified | Not a universal sign prediction; a different capability path can give different responses |
| Scope and inference are both useful choices | Removing either eliminates one of the competing margins | A narrower application may legitimately hold one of them fixed |

### Validation and final disposition

The rewritten main argument is approximately 3,350 words with four figures. The paper retains eight supplementary figures and the notebook regenerates 22 figures in total. The new figures add no solver, calibrated cases, or economic parameters. Tests verify that their coordinates and denominators reproduce the audited economic outcomes.

The final economic read also makes the existence condition for optimized policies explicit and distinguishes the variable token charge from total invoices with fixed fees.

Validation: 69 Python tests and 11 JavaScript tests pass. The notebook regenerates all 22 figures and passes its independent optimizer audits. All 12 manuscript figures mount in the browser, with 44 interactive panels, no console warnings or errors, and no unrendered or overflowing equations. All seven tables fit the reading width. The 20 local reading resources and internal section links resolve, and the exported chart fingerprint matches the current numerical sources.

The reference-paper comparison is recorded above so the editorial standard is inspectable rather than an unsupported ranking. The manuscript now leads with its strongest result, makes its counterexamples intelligible, and ties its conclusions to measurements. It remains a qualitative model without empirical calibration, a joint-regime numerical solution, or a general-equilibrium forecast.

## Round 3 — contribution and workflow revision

The user approved two changes after discussing the final draft: sharpen what the paper contributes beyond generic rebound reasoning, and carry concrete workflows through the argument. A subagent then performed an independent editorial and claim-evidence review without editing files.

### Findings and revisions

1. **The mechanism, rather than the price identity, should organize the paper.** Reframe the contribution as a decomposition of token demand into work assigned and inference per work unit. The active constraint identifies the expansion margin—adoption under finite work and supervisory leverage under scarce attention—while its size relative to inference savings determines the sign. Explicitly relate the price-efficiency result to prior direct-rebound work instead of claiming a new rebound principle.

2. **The main result order should answer the title sooner.** Move capability under the two constraints ahead of the price and efficiency diagnostic. Renumber the four main figures accordingly. Remove the repeated display of the demand decomposition and the technical attention-value benchmark from the main text; the latter already appears with its proof in Appendix A.

3. **Examples should fit one-attempt failure accounting.** Replace mandatory reports and generic search with two hypothetical, optional workflows. A fixed catalog of optional content improvements represents limited work; rejected revisions leave existing content in place. A deep queue of optional software improvements represents scarce review attention; rejected changes can be abandoned. Map the model variables to each workflow and state that neither example characterizes its industry as a whole.

4. **The examples expose conditions, not a need for more choices.** Make clear that candidate work is otherwise comparable, review detects success, the software backlog is abundant only relative to review capacity during the period, and review-tool compute and maintenance costs are outside the current specification. Mandatory work, task-dependent policies, partial reuse, repair, undetected damage, and learning remain outside scope.

5. **Add one exact, portable corollary.** Under fixed work, if chosen inference intensity falls to a fraction $r$ of its initial value, token demand must fall whenever initial adoption exceeds $r$, even if every remaining opportunity adopts. This follows from $D=WAx$ and does not rely on the illustrative adoption distribution.

6. **Make the empirical questions follow the contribution.** Ask first which constraint is active, then record assigned work, completed work, inference, and review together, and finally use price variation as a conditional efficiency diagnostic. The conclusion ends with the four objects a forecast must identify: work that can enter, whether review binds, review scaling with scope, and inference per work unit.

### Claim-evidence disposition

The independent review found the analytical claims internally consistent and identified no reason to restore retry choice or add numerical cases. It rejected any statement that a regime alone determines the sign of demand: the regime identifies the offsetting margin, whose magnitude must still be compared with inference savings. Capability humps and valleys remain illustrative; the price-efficiency identity, adoption-saturation bound, and uniform-review scaling retain their explicit conditions.
