# Manuscript self-review

Branch: `simplify-single-attempt-model`. No review authorizes a merge to `main`. Rounds 1–3 document superseded drafts; Round 4 describes the current short draft.

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
| Choice of model effort level | Users cannot save tokens or spend more inference as prices change | Keep |
| Choice of delegated scope | Capability cannot change the amount of work supervised per hour | Keep |
| Work adoption or an equivalent extensive margin | A finite workload cannot undergo adoption takeoff and saturation | Keep a general hurdle distribution in the argument |
| Review overhead and scope-dependent review | Cannot distinguish amortizing checkpoints from review that grows with the task | Keep; state which results need each part |
| Separate feasibility and execution notation | Not needed for demand accounting or the price-efficiency result | Use one success function in the main argument; retain the flexible numerical specification in the appendix |
| Particular exponential and logistic functions | Not needed for the general price, efficiency, and uniform-review results | Confine to numerical illustrations |
| A full joint allocation, dynamic diffusion, or endogenous supply model | Needed for intermediate constraints or an aggregate time forecast, not these comparative statics | Explain the boundary; do not add these models |

The zero-overhead benchmark is especially informative: with a homogeneous review function and horizon-scaling capability, optimal scope scales with capability, model effort stays fixed, and attention-limited demand scales with `m^(1-beta)`. The unusual valley needs additional structure; it should not receive the same prominence as the general price-efficiency identity.

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
| Capability can raise or lower demand | Audited examples expose both activity and effort; their product is verified | Not a universal sign prediction; a different capability path can give different responses |
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

5. **Add one exact, portable corollary.** Under fixed work, if the chosen model effort level falls to a fraction $r$ of its initial value, token demand must fall whenever initial adoption exceeds $r$, even if every remaining opportunity adopts. This follows from $D=WAx$ and does not rely on the illustrative adoption distribution.

6. **Make the empirical questions follow the contribution.** Ask first which constraint is active, then record assigned work, completed work, inference, and review together, and finally use price variation as a conditional efficiency diagnostic. The conclusion ends with the four objects a forecast must identify: work that can enter, whether review binds, review scaling with scope, and inference per work unit.

### Claim-evidence disposition

The independent review found the analytical claims internally consistent and identified no reason to restore retry choice or add numerical cases. It rejected any statement that a regime alone determines the sign of demand: the regime identifies the offsetting margin, whose magnitude must still be compared with inference savings. Capability humps and valleys remain illustrative; the price-efficiency identity, adoption-saturation bound, and uniform-review scaling retain their explicit conditions.

### Validation

The revised source builds as a 12-figure reading edition. The Python suite passes 68 tests with one environment-dependent live-server test skipped because the sandbox disallows binding a local socket; the JavaScript interaction suite passes all 11 tests. Browser checks confirm the intended section and figure order, 12 mounted figure hosts, valid images and internal links, rendered mathematics, and no horizontal overflow. The 27-page PDF identifies the exact manuscript commit in its metadata and footer; all pages were rendered and inspected for clipping, legibility, and layout integrity.

## Round 4 — short model-first draft

The user found the prior version too long and difficult to read. This round resets the exposition rather than polishing that structure.

### Structural decisions

1. **Begin with the model.** Remove the abstract and opening essay. Define scope, tokens, feasibility, conditional reliability, total reliability, review time, surplus, adoption, token demand, and token spending before presenting results.

2. **Use the same questions in both regimes.** The work-limited and attention-limited sections each study capability, token efficiency, and token price in that order. Each result heading states the observation instead of naming a parameter or method.

3. **Show absolute units.** Each of the six figures pairs token demand in trillions of tokens with token spending in millions of dollars. Price falls from left to right. Logarithmic vertical scales keep industries with very different market sizes visible.

4. **Use three intuitive cases per regime.** The work-limited figures compare gradual adoption, clustered adoption, and early saturation. The attention-limited figures compare reusable, balanced, and nearly proportional review. The manuscript states the parameter changes immediately before the results.

5. **Keep this pass small.** Remove the conclusion, literature review, empirical agenda, proofs, and supplementary figures from the manuscript. They remain available in version history and the numerical repository, but they no longer interrupt the first explanation of the model.

### Current limitations

This short draft deliberately stops after the six comparative statics. It does not yet explain which results are general, connect the model to evidence, or defend the functional forms against alternatives. Those are the next editorial questions. The current goal is a readable base that makes later additions easy to evaluate.

### Validation before PDF export

The manuscript is about 1,800 words and contains six figures. The notebook exports 24 figures and again passes all optimizer and boundary audits. The Python suite passes 70 tests with one environment-dependent live-server test skipped; all 11 JavaScript interaction tests pass. The HTML build contains six interactive figure hosts, no broken static images, no math rendering errors, and no horizontal overflow in the checked desktop layout.

The final print edition is 11 pages and identifies manuscript commit `48632d9` in its metadata and footer. All pages were rendered and inspected. Equations, tables, figures, headers, footers, and page transitions are legible, with no clipping or repeated browser controls. Each footer links to the exact committed manuscript source.

## Round 5 — external review of the intro/TL;DR draft (2026-09-02)

Scope: README.md at commit `2fade6f` (branch `codex/intro-tldr`). Every number quoted in the text was re-derived from `figures/paradigms.json` and by re-solving with the packaged optimizer. The browser port (`paper_assets/model.js`) reproduces all 48 audited curves to relative objective error below 1e-13 and demand/policy error below 2e-6, so the interactive builder and the paper agree.

### Verdict

Short, readable, mechanism clear: capability, efficiency, or price moves demand through (adoption or supervisory leverage) × tokens per work unit, and the active constraint picks the margin. The two-regime split is the contribution and it survives numerical audit. Three weaknesses. (1) The two most general results were dropped between Rounds 3 and 4 (price–efficiency identity, adoption-saturation bound); without them the draft reads as "here are curves" when it can state exact conditions. (2) Every long-run claim in Section 2 and the ceilings in Figures 4 and 8b are consequences of the effort floor $x\geq1$, a units convention; the text does not flag that. (3) Prose needs a copy-edit (list at the end).

### Quantitative check

All quoted numbers match the data: $c^W_{\rm res}(0.8)\in[0.29,0.61]$, $c^W_{\rm res}(30)=14.6/18.9/22.2$; $\rho^*(30)/\rho^*(1)=5.3/15.8/1.8$; $c^H_{\rm res}(0.8)\in[0.20,0.57]$, $c^H_{\rm res}(30)\in[39.4,70.7]$. Two things the text should say:

- Figure 1 stars: the low-hurdle peak is at $m=0.1$ and the high-hurdle peak at $m=30$, both edges of the plotted range, not interior maxima. Say so or drop those two stars.
- In both reservation-price curves, 45–50 of the 88 points per line sit on $x=1$. The Figure 4 "plateau" and the Figure 8b "bend toward $b$" are floor effects. Below are the closed forms.

### Results that can be stated exactly

1. **Price–efficiency identity.** $\eta$ and $x$ enter only through $\eta x$ in $P$ and $cx$ in cost, so $u(s,x;\eta,c)=bP(s,\eta x)-(c/\eta)(\eta x)-wh(s)/s$. Hence $x^*(\eta,c)=x^*(1,c/\eta)/\eta$, $D(\eta,c)=D(1,c/\eta)/\eta$, $R(\eta,c)=R(1,c/\eta)$; identical for $J$. Corollary: the demand index in Figure 2b equals the spending index in Figure 3d read at $c=1/\eta$ (checked: 1.0601 vs 1.0601 at $\eta=2$ reference; 1.8750 vs 1.8750 hard execution; both regimes). Exact wherever the floor does not bind (it binds for low inference returns at $c\geq2$). This is Round 1's strongest result; one displayed equation replaces most of the 2.2/3.2 prose and tells the reader that an efficiency release and a price cut are the same experiment for the customer, and differ for the provider only by the factor $\eta$ in quantity.
2. **Adoption-saturation bound** (Round 3 corollary). $D=WAx$ with $A\leq1$: if $x^*$ falls to a fraction $r$ of its baseline, demand falls whenever baseline adoption exceeds $r$. Reference: $A(1)=0.455$, $x^*(1)=2.85$, floor gives $r=0.35$, so $D(\infty)/D(1)\leq0.77$ (data: 0.74 at $m=30$). Functional-form free; explains Figure 1 without the logistic.
3. **Long-run work-limited demand is pinned by the floor.** $A\to1$, $x^*\to1$, so $D(\infty)=W$ in every industry; the level in Section 2 is set by $x_{\min}=1$. Say what the floor is (tokens needed to emit the output, which $\eta$ does not reduce). If minimal output tokens fall over time, long-run work-limited demand falls with them.
4. **Reservation-price ceilings.** Work-limited: $m\to\infty$ gives $P\to1$, $h(s)/s\to0$ ($\beta<1$), $x=1$, so $u^*\to b-c$ and $c^W_{\rm res}(\infty)=b-u^*(1,c_0)$: 19.7 reference, 30.6 hard execution, 18.1 slow review (data at $m=3000$: 18.6, 28.9, 17.6). The provider ends up charging the value of a work unit minus the surplus the user must keep. Attention-limited: $J^*=\ell(s)[bP-c]-w$ with $\ell\to\infty$ forces $bP-c\to0$, so $c^H_{\rm res}\to b$ (reference at $m=3000$: 91.3; $\beta=0.95$: 55.4, converging like $m^{0.05}$). One line each; they explain Figures 4 and 8b better than the current prose. 2.4 says "eventually plateaus" while the curve is at 14.6 of 19.7 at $m=30$; say "approaches $b-u^*_0$".
5. **Jevons condition, work regime.** Envelope: $du^*/dc=-x^*$. With logistic adoption, $d\ln A/d\ln c=-(1-A)\,c\,x^*/\sigma$. Spending rises as price falls iff $(1-A)\,c\,x^*/\sigma+|\varepsilon_x|>1$, $\varepsilon_x=d\ln x^*/d\ln c$. Reference at $c=1$: $0.39+0.77=1.16$ (finite differences agree to four digits): mild Jevons near baseline, gone once $A$ rises. This is the sentence in 2.3 in symbols: non-adopted share × token spend per work unit ÷ hurdle spread. Attention regime has no adoption term, so it needs $|d\ln\ell/d\ln c+\varepsilon_x|>1$; that is why 3.3 is "harder".
6. **At the floor, demand and attention value grow together.** $D=H\ell(s^*)$ and $\rho^*\approx b\,\ell(s^*)$, both $\propto m^{1-\beta}$. One sentence links 3.1 and 3.4.

### Modeling assumptions

- Capability as a horizon multiplier ($q$, $r$ depend on $s/m$) is the METR task-horizon framing (Kwa et al. 2025; horizon doubling roughly every 7 months). Cite it: it anchors the functional form and maps $m$ to time ($m=30$ is about five doublings).
- $h(s)=h_0+h_1[(1+s)^\beta-1]$ is $h_0+h_1\beta s$ for $s\ll1$ regardless of $\beta$; the $s^\beta$ behaviour appears only for $s\gg1$. Reference $s^*(1)=0.41$, so the figures live in the transition zone and the $m^{1-\beta}$ asymptotics in 3.4 are a large-$s$ statement. The parameter table's "through the term $s^\beta$" is inaccurate.
- Adoption depends on $u^*$ alone and every adopter uses the same $(s^*,x^*)$; heterogeneity is only in the hurdle. Clean, but say that task heterogeneity in $s$ or $b$ is deliberately absent, otherwise readers ask why adoption does not change the task mix.
- One attempt, review paid regardless, no retry (Round 1 decision). Keep one sentence; retries would change tokens per success and the Figure 5 shapes.
- Work-limited ignores review capacity; attention-limited ignores $W$. Round 1's shadow-price formulation (charge $w+\tau$ per review hour) is the bridge. No need to solve it; one paragraph on when each regime applies (compare $W A h(s^*)/s^*$ with $H$) makes the two sections one model.
- $H$ fixed while $\rho^*$ rises 5–16×: a firm would hire reviewers. That is the complementarity result (labor demand for review rises with $m$ when $\beta<1$) and the link to the labor-share question; one sentence in 3.4 or the conclusion.
- $\phi$ is a random variable, not a parameter; "19 parameters" counts $s$, $x$, $\phi$.
- 1.7 heading says revenue, text says spending, 2.3 uses both. Define once: spending by users equals revenue to providers.

### Exposition

- TL;DR: bullet 1 lacks a period after the link; bullets 4 and 6 overlap; each bullet should carry its mechanism in one clause.
- Sections 2 and 3 run capability → efficiency → price symmetrically. Close with a six-cell sign table (regime × lever: demand, spending, ceiling); that is the figure people will screenshot.
- Normalization is explained in three captions; once suffices.

### Line edits

"user's adoption" → "users'"; "has an organizational mandate" → "have"; "userwould" → "user would"; "Figure 1a) marks" → "Figure 1(a) mark"; "token savings effect dominate" → "the token-saving effect dominates"; "significant adoptions" → "adoption"; "(green and orange line)" → "lines"; "low-ish" and "pretty striking" are informal; "Similar to Figure 4, here we also ask the question of" → "As in Figure 4, we ask"; "made explicitly shortly" → "made explicit shortly"; last sentence of 1.4 lacks a period.

### Project

- Quoted numbers are typed by hand and have drifted before; have the build inject them from `paradigms.json` (placeholder syntax in README.md, resolved by `paper.py`).
- Preview server rebuilt in-process, so edits to `paper.py` were not picked up until restart (root cause of the "disclosure not collapsible" report). Fixed this round: rebuilds run in a subprocess, and `<details>` pass-through is a strict allowlist in the renderer instead of string surgery on rendered HTML.
- Added: browser-side model (`model.js`, Node-tested against the audited Python curves), plot builder section (`explore.js`), GitHub Pages workflow. Suites: 81 Python, 83 Node.
