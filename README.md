# Modeling Token Demand

## Delegation, reliability, retries, verification, and adoption

**Working draft**

## Abstract

AI token demand is not mechanically increasing in model capability. A user chooses how to use a model: how much work to delegate before the next checkpoint, how much inference to spend on each attempt, and how many attempts to allow. Better models can reduce the tokens needed for a fixed task, but they can also unlock longer autonomous tasks, induce adoption, and let each hour of human attention supervise more machine work.

This paper develops an industry-level model of those choices. Each industry has its own task-difficulty distribution, returns to inference, verification technology, value of successful work, and adoption hurdles. The model separates a task's capability frontier from stochastic execution reliability, so repeated attempts improve pass rates without forcing success to converge to one. Users choose delegation horizon, inference intensity, and retry count to maximize surplus relative to the resource that constrains them: underlying work in one polar case and human attention in the other. Aggregate demand is then determined by both intensive margins---tokens used by existing industry work---and extensive margins such as adoption and newly economical work.

The central result is deliberately ambiguous. Model progress can raise token demand in capability-threshold industries while lowering it in mature, work-limited industries. Token efficiency can produce an inverted-U demand curve when it triggers adoption before eventually reducing tokens per unit of work. Verification improvements are also not one-dimensional: cheaper checkpoints can increase adoption while encouraging shorter, safer tasks and less inference per attempt.

## 1. The economic object

Consider an industry indexed by $i$. Illustrative industries include software development, professional writing, routine business processing, and scientific research.

A unit of underlying work is measured in **human-hour-equivalents**. The user chooses three features of an AI interaction policy:

- $s$: the **delegation horizon**, or human-hours of work delegated before the next human checkpoint;
- $x$: **inference tokens per human-hour-equivalent, per attempt**;
- $k$: the maximum number of attempts allowed for one delegated chunk.

One attempt therefore consumes

```math
T=sx.
```

This decomposition matters. Increasing $s$ means asking the model to carry a larger project between checkpoints. Increasing $x$ means making the model reason, search, test, or deliberate more intensely within an attempt. Autocomplete and an autonomous agent differ mainly in $s$; a standard run and a high-compute run differ mainly in $x$.

### Core variables

| Symbol | Meaning |
|---|---|
| $i$ | Industry |
| $s$ | Delegated human-hour-equivalents per checkpoint |
| $x$ | Tokens per delegated hour, per attempt |
| $k$ | Maximum attempts per delegated chunk |
| $m$ | General model capability |
| $\eta$ | Effective inference obtained per token |
| $q_i$ | Probability that a task lies within the model's capability frontier |
| $r_i$ | One-attempt success probability, conditional on the task being solvable |
| $P_i$ | Success probability after at most $k$ attempts |
| $E_i$ | Expected number of attempts actually consumed |
| $h_i$ | Human verification time per attempt |
| $b_i$ | Value per successfully completed hour-equivalent of work |
| $w_i$ | Cost of one hour of human attention |
| $c$ | Price per token |
| $A_i$ | Fraction of potential industry work volume delegated to AI |
| $W_i$ | Total potential industry work volume in human-hours per period |
| $H_i$ | Human verification hours available per period |
| $N_i$ | Number of delegated chunks attempted per period |
| $J_i$ | Expected net surplus per human verification hour |
| $D_i$ | Token demand per period |

## 2. Capability and execution are different

A naive retry model starts with a single-attempt success probability $p$ and writes pass@$k$ as $1-(1-p)^k$. That expression approaches one. It is inappropriate when some tasks are systematically beyond the model, because no number of similar retries will solve them.

The model therefore separates two kinds of uncertainty.

### 2.1 The capability frontier

Let

```math
q_i(s;m)
=
\exp\left[-\left(\frac{s}{\lambda_i m}\right)^{\nu_i}\right].
```

$q_i$ is the probability that a randomly drawn task of horizon $s$ in industry $i$ is within the model's capability frontier at all.

- $\lambda_i>0$ is industry tractability. Larger values mean that longer tasks are feasible for a model of a given capability.
- $\nu_i>0$ controls how sharply feasibility deteriorates with task horizon.
- $m>0$ shifts the frontier outward.

This is a reduced-form representation of latent task difficulty. An industry with routine, structured tasks may have a high $\lambda_i$. An industry requiring novel discoveries may have a low one.

### 2.2 Execution reliability conditional on solvability

Even a solvable task can fail on a particular trajectory. Define

```math
r_i(s,x;m,\eta)
=
\exp\left[
-\frac{s}{a_i m(\eta x)^{\alpha_i}}
\right].
```

Here:

- $a_i>0$ measures industry-specific execution ease;
- $0<\alpha_i<1$ gives diminishing returns to additional inference;
- $\eta>0$ is token efficiency: effective inference per token.

Conditional reliability falls with delegation horizon and rises with capability, inference intensity, and token efficiency.

The unconditional success probability on one attempt is

```math
p_{1,i}=q_i r_i.
```

Capability $m$ and token efficiency $\eta$ are intentionally distinct. In this specification, $m$ both expands the set of solvable tasks through $q_i$ and improves execution through $r_i$. Token efficiency mainly lets each token buy more execution effort through $r_i$.

## 3. A bounded retry model

Conditional on a task being solvable, suppose attempts explore sufficiently different trajectories that each succeeds with probability $r_i$. With at most $k$ attempts,

```math
P_i(s,x,k;m,\eta)
=
q_i\left[1-(1-r_i)^k\right].
```

Consequently,

```math
\lim_{k\to\infty}P_i=q_i,
```

not one. Retrying can repair execution failures but cannot cross the capability frontier.

The user stops after the first success. Expected attempts are therefore

```math
E_i(s,x,k)
=
(1-q_i)k
+q_i\frac{1-(1-r_i)^k}{r_i}.
```

The first term is the cost of fundamentally unsolvable tasks, which consume all $k$ attempts. The second is the truncated geometric expectation for solvable tasks.

Expected tokens consumed by one delegated chunk are

```math
T_i^{\mathrm{exp}}=sxE_i.
```

This remains a deliberately minimal retry model. Empirical pass@$k$ curves could later replace the conditional-independence assumption without changing the rest of the framework.

## 4. Human verification

Let verification time after one attempt be

```math
h_i(s)=h_{0,i}+h_{1,i}s^{\beta_i}.
```

- $h_{0,i}$ is fixed checkpoint overhead: reading a summary, reconstructing context, approving a result, or issuing the next instruction.
- $h_{1,i}$ is the variable verification burden.
- $\beta_i$ is the verification scaling exponent.

$\beta_i$ is a central source of industry heterogeneity:

- $\beta_i\approx 0$: outcome verification, automated tests, strong formal checks, or cheap sampling;
- $0<\beta_i<1$: spot checks and summaries cause review to grow sublinearly;
- $\beta_i\approx 1$: the human must inspect nearly all of the output.

Verification cost per unit of underlying work is

```math
w_i\frac{h_i(s)}{s}.
```

For $\beta_i\leq 1$, this normally falls with $s$: larger chunks amortize checkpoint overhead. Reliability moves in the opposite direction, creating an interior delegation choice.

## 5. The user's problem

The correct objective depends on what is scarce. We first state the fixed-work problem, then derive the abundant-work, attention-constrained problem separately.

Assume a successfully completed hour-equivalent of work in industry $i$ produces value $b_i$. Expected value per unit of underlying work is $b_iP_i$.

Define the cost of one attempted unit of work as

```math
C_i(s,x)
=
cx+w_i\frac{h_i(s)}{s}.
```

Expected surplus per hour-equivalent is

```math
u_i(s,x,k)
=
b_iP_i(s,x,k)
-E_i(s,x,k)C_i(s,x).
```

When underlying work is the binding resource, the user chooses

```math
(s_i^{\star},x_i^{\star},k_i^{\star})
=
\arg\max_{s>0,\;x>0,\;k\in\mathbb{N}}u_i(s,x,k).
```

The outside option is not to use AI. Adoption is introduced below.

### 5.1 The optimal retry count in the fixed-work problem

The retry choice has a clean marginal solution. After $k$ allowed attempts, adding attempt $k+1$ raises success probability by

```math
\Delta P_i=q_i r_i(1-r_i)^k.
```

The extra attempt is actually made only if the first $k$ attempts have not succeeded, an event with probability

```math
1-P_{i,k}.
```

The next retry is worthwhile if and only if

```math
b_iq_ir_i(1-r_i)^k
>
(1-P_{i,k})C_i(s,x).
```

This rule generally produces a finite retry count. As retries accumulate, the remaining pool contains a growing share of fundamentally unsolvable tasks, so the marginal return falls faster than it would under naive independent retries.

An equivalent expression makes the cutoff explicit. If $b_ir_i>C_i$ and $q_i<1$, another retry is worthwhile when

```math
(1-r_i)^k
>
\frac{C_i(1-q_i)}{q_i(b_ir_i-C_i)}.
```

The optimal $k_i^{\star}$ is the number of successive attempts satisfying the marginal condition. If even the first attempt fails it, AI use is not economically viable under that $(s,x)$ policy.

### 5.2 Inference intensity

For a fixed retry cap, an interior optimum in $x$ satisfies

```math
b_iP_{i,x}=E_{i,x}C_i+E_ic.
```

The left side is the marginal value created by more inference. On the right, $E_ic$ is the direct token cost and $E_{i,x}C_i$ captures the fact that a more reliable attempt can change expected retry costs. Because higher $x$ often lowers expected attempts, this second term can be negative.

### 5.3 Delegation horizon

For a fixed retry cap, an interior optimum in $s$ satisfies

```math
b_iP_{i,s}
=
E_{i,s}C_i
+E_iw_i\frac{h_i'(s)s-h_i(s)}{s^2}.
```

The reliability terms favor shorter tasks. The last term captures checkpoint amortization. For

```math
h_i(s)=h_{0,i}+h_{1,i}s^{\beta_i}
```

with $\beta_i\leq1$,

```math
h_i'(s)s-h_i(s)
=
-h_{0,i}-(1-\beta_i)h_{1,i}s^{\beta_i}
\leq0.
```

Thus verification cost per unit falls as the delegated chunk grows. The optimum balances that saving against a lower capability probability, lower execution reliability, and potentially more wasted attempts.

The economic meaning of $s_i^{\star}$ is direct: it is the user's optimal degree of agenticness.

### 5.4 The attention-constrained user

Now suppose useful work is abundant but the industry has only $H_i$ human verification hours per period. Let $N_i$ be the number of delegated chunks. The user solves

```math
\max_{s,x,k,N_i\geq0}
N_i s\,u_i(s,x,k)
```

subject to

```math
N_iE_i(s,x,k)h_i(s)\leq H_i.
```

If AI use creates positive value and attention binds, then

```math
N_i^{\star}
=
\frac{H_i}{E_i(s,x,k)h_i(s)}.
```

Substitution reduces the policy problem to

```math
(s_i^{\star},x_i^{\star},k_i^{\star})
=
\arg\max_{s>0,\;x>0,\;k\in\mathbb N}
J_i(s,x,k),
```

where

```math
J_i(s,x,k)
=
\frac{s\,u_i(s,x,k)}{E_i(s,x,k)h_i(s)}
=
\frac{s}{h_i(s)}
\left[
\frac{b_iP_i(s,x,k)}{E_i(s,x,k)}-cx
\right]
-w_i.
```

$J_i$ is expected net surplus per human verification hour. The size of the attention endowment $H_i$ scales activity but does not change the optimal policy in this single-industry polar case. The explicit human attention cost $w_i$ enters $J_i$ as the constant $-w_i$: it determines whether AI use is worthwhile but does not change $(s_i^{\star},x_i^{\star},k_i^{\star})$ once the hard attention constraint binds. A shadow price on the constraint would add to that opportunity cost.

This objective changes retry behavior sharply. At any fixed $(s,x)$,

```math
\frac{P_{i,k}}{E_{i,k}}
\leq
q_ir_i
=
\frac{P_{i,1}}{E_{i,1}},
```

because $1-(1-r_i)^k\leq kr_i$. The token-cost term in $J_i$ does not depend on $k$, so $k=1$ weakly dominates every larger retry cap. After a failure, a fresh task is another draw from the prior, while the failed task is more likely to belong to the unsolvable tail. Retrying can return if work is finite, abandoning a task is costly, task values differ, or later attempts learn from earlier failures.

## 6. Industry adoption and induced work

Let $A_i\in[0,1]$ be the fraction of potential work volume in industry $i$ that is economically delegated to AI. This is a reduced-form adoption measure that summarizes heterogeneity across the industry's work in suitability, integration difficulty, regulation, trust, switching costs, and other barriers.

Let $\phi$ denote an adoption hurdle measured in dollars per human-hour-equivalent of work. If these per-unit hurdles have a work-volume-weighted cumulative distribution $G_i$, the adopted share of industry work is

```math
A_i=G_i(u_i^{\star}).
```

A logistic example used later is

```math
A_i
=
\frac{1}{1+\exp[-(u_i^{\star}-\mu_i)/\sigma_i]},
```

where $\mu_i$ is the surplus per unit of work at which half of potential industry work is delegated to AI, and $\sigma_i$ determines how gradually adoption spreads across the industry's work.

This extensive margin can be nonlinear. If a large share of industry work has adoption hurdles near $\mu_i$, a modest technology improvement can move $A_i$ from nearly zero to nearly one.

The volume of useful work can also be endogenous. One optional reduced form is

```math
W_i(C_i^{\mathrm{AI}})
=
W_i^0
\left(
\frac{C_i^{\mathrm{legacy}}}{C_i^{\mathrm{AI}}}
\right)^{\epsilon_i},
```

where $\epsilon_i$ is the elasticity of work creation. Compliance processing may have low $\epsilon_i$ because the number of required filings is fixed. Software experiments, personalized content, scientific search, and continuous monitoring may have much higher elasticities.

## 7. Aggregate token demand

There are two useful limiting regimes.

### 7.1 Work-limited demand

Suppose the industry has $W_i$ hour-equivalents of potential work per period. There are $W_i/s_i^{\star}$ chunks. Each adopted chunk consumes $s_i^{\star}x_i^{\star}E_i^{\star}$ expected tokens. Therefore

```math
D_i^{\mathrm{work}}
=
W_iA_ix_i^{\star}E_i^{\star}.
```

The delegation horizon cancels mechanically. Merely grouping the same fixed work into larger chunks does not create token demand. It matters indirectly because it changes reliability, inference intensity, retry behavior, adoption, and the size of the economically useful work pool.

### 7.2 Human-attention-limited demand

Suppose instead that useful work is abundant but only $H_i$ human verification hours are available. The policy is now chosen to maximize $J_i$, as derived in Section 5.4. Each chunk uses $E_i^{\star}h_i(s_i^{\star})$ expected human hours, so the industry can supervise $H_i/[E_i^{\star}h_i(s_i^{\star})]$ chunks. Multiplying by expected tokens per chunk gives

```math
D_i^{\mathrm{attention}}
=
H_i\frac{s_i^{\star}x_i^{\star}}{h_i(s_i^{\star})}.
```

The expected retry count cancels because retries consume both tokens and checkpoints. The key ratio is

```math
\frac{D_i^{\mathrm{attention}}}{H_i}
=
\underbrace{\frac{s_i^{\star}}{h_i(s_i^{\star})}}_{\text{AI work launched per human hour}}
\times
\underbrace{x_i^{\star}}_{\text{tokens per unit of work}}.
```

This is the main agentic-demand channel: better models can let each human checkpoint launch far more machine work.

For any *given* policy, the two capacity limits imply the accounting equation

```math
D_i(s,x,k)
=
\min\left\{
W_iA_i(u_i)xE_i,
\;H_i\frac{sx}{h_i(s)}
\right\}.
```

This minimum is not, by itself, a solved equilibrium. If both work and attention can bind, the user must choose $(s,x,k,N_i)$ jointly subject to both capacity constraints. In particular, one should not take the minimum of demands generated by two policies optimized under different polar assumptions. The two polar cases isolate the mechanisms cleanly; the joint problem is the natural extension.

After solving the relevant user problem, aggregate demand is

```math
D=\sum_iN_i^{\star}s_i^{\star}x_i^{\star}E_i^{\star}.
```

## 8. Comparative statics

### 8.1 Model capability

When $m$ rises, both $q_i$ and $r_i$ improve. For a fixed task and policy, this usually reduces wasted retries. Once the user reoptimizes, the typical but not universal response is

```math
m\uparrow
\quad\Rightarrow\quad
s_i^{\star}\uparrow,
```

because longer tasks become feasible. But $x_i^{\star}$ and $k_i^{\star}$ are ambiguous. A stronger model may need less inference and fewer retries for existing work, or the user may move into harder work and spend more.

Token demand rises when adoption, work creation, or supervisory leverage grows faster than tokens per completed unit fall. It falls when adoption is already saturated and efficiency dominates.

The attention-constrained case has a particularly transparent scaling force. Because $q_i$ depends on $s/m$ and $r_i$ depends on $s/[m(\eta x)^{\alpha_i}]$, choosing $s_i^{\star}$ roughly proportional to $m$ can hold the reliability arguments approximately fixed. If variable verification time dominates fixed overhead, then

```math
\frac{s_i^{\star}}{h_i(s_i^{\star})}
\propto
m^{1-\beta_i}.
```

Thus capability can raise attention-limited demand even when $x_i^{\star}$ barely changes. This channel is strong in machine-verifiable industries with low $\beta_i$ and weak when review grows nearly linearly with scope.

### 8.2 Token efficiency

When $\eta$ rises, each token buys more effective inference. Holding behavior and reliability fixed, the required $x$ falls. That is the engineering effect.

Demand can nevertheless rise near an adoption threshold. A higher $\eta$ increases optimized surplus, raises the share of industry work delegated to AI, and may expand $W_i$. The likely lifecycle is an inverted U:

```math
\text{efficiency improvement}
\rightarrow
\text{adoption surge}
\rightarrow
\text{saturation}
\rightarrow
\text{declining tokens per unit}.
```

A change in token price $c$ is a different experiment. Lower prices normally raise token demand, but that is an ordinary movement along a demand curve rather than Jevons paradox by itself. Define token spend as

```math
S_i(c)=cD_i(c).
```

Spend rises as token prices fall only when token demand increases more than proportionally:

```math
\frac{d\log D_i}{d\log c}<-1.
```

This spending rebound is important for provider revenue and customer expenditure, but the direct Jevons-style test in this model is whether higher token efficiency $\eta$ raises total token use.

### 8.3 Verification cost and verification technology

Write

```math
h_i(s)\rightarrow v h_i(s),
```

where a lower $v$ means cheaper verification. At a fixed interaction policy, attention-limited demand is proportional to $1/v$. But reoptimization is subtler.

Cheaper checkpoints can make AI viable for a larger share of industry work. They can also reduce the need to amortize a checkpoint across a long task, leading the user to choose a smaller $s$, lower $x$, and possibly more attempts. Thus a uniform fall in verification cost need not make behavior more agentic.

Changes in the **shape** of verification are different from changes in its level. Automated outcome checks that make $\beta_i$ small allow verification to grow slowly with task scope and can greatly increase $s/h_i(s)$. Better failure detection, lower escaped-error losses, and formal verification would add further channels not modeled in the minimal version.

## 9. Illustrative numerical solution

This section demonstrates regimes the model can generate. It is not an empirical forecast. One unit of $x$ represents 100,000 tokens per delegated human-hour. The token price is normalized to $c=1$ dollar per token unit. Baseline technology has $m=\eta=v=1$.

For each industry and scenario, the solution enumerates $k\in\{1,\ldots,8\}$ and searches logarithmic grids over $s\in[0.02,80]$ hours and $x\in[0.02,12]$. Adoption uses the logistic specification above. Work-limited demand is reported with $W_i=1$ and normalized to each industry's own baseline.

### 9.1 Calibration

| Industry | $\lambda$ | $\nu$ | $a$ | $\alpha$ | Interpretation |
|---|---:|---:|---:|---:|---|
| Software | 15 | 1.25 | 5 | 0.55 | Tractable, with sublinear review |
| Review-bound writing | 12 | 1.15 | 4 | 0.50 | Capable execution, but near-linear review |
| Routine processing | 40 | 1.35 | 8 | 0.45 | Easy tasks and strong verification |
| Frontier research | 2 | 1.40 | 1.5 | 0.60 | A tight capability frontier |

| Industry | $h_0$ | $h_1$ | $\beta$ | $b$ | $w$ | $\mu$ | $\sigma$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Software | 0.035 | 0.025 | 0.35 | 125 | 100 | 112 | 6 |
| Review-bound writing | 0.025 | 0.130 | 0.95 | 125 | 100 | 100 | 7 |
| Routine processing | 0.025 | 0.012 | 0.25 | 100 | 75 | 30 | 8 |
| Frontier research | 0.040 | 0.080 | 0.50 | 220 | 150 | 152 | 10 |

The baseline optimum is:

| Industry | $s^{\star}$ (hours) | $x^{\star}$ | $k^{\star}$ | $P^{\star}$ | Expected attempts | Adoption |
|---|---:|---:|---:|---:|---:|---:|
| Software | 0.89 | 0.88 | 4 | 97.0% | 1.29 | 48% |
| Review-bound writing | 0.47 | 1.13 | 3 | 97.5% | 1.16 | 45% |
| Routine processing | 1.36 | 0.32 | 6 | 98.9% | 1.37 | ~100% |
| Frontier research | 0.32 | 3.69 | 2 | 91.7% | 1.16 | 48% |

Routine processing receives long chunks and little inference per unit. Frontier research receives short chunks, high inference, and few retries because the unsolvable tail makes repeated attempts unattractive. Review-heavy work remains interaction-intensive despite relatively capable execution.

### 9.2 Scenario A: model capability rises

The table reports work-limited token demand relative to each industry's $m=1$ baseline, with users reoptimizing at every point.

| Capability $m$ | Software | Review-bound | Routine | Frontier research |
|---:|---:|---:|---:|---:|
| 0.5 | 0.76 | 0.83 | 1.60 | 0.14 |
| 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 0.91 | 0.96 | 0.60 | 1.31 |
| 4 | 0.62 | 0.82 | 0.38 | 0.83 |

Routine processing is efficiency-dominated. Adoption is already saturated, so stronger models use fewer tokens on a fixed work pool.

Frontier research is threshold-dominated. Demand rises as adoption moves from 5% at $m=0.5$ to 48% at $m=1$ and 87% at $m=2$. By $m=4$, adoption is near saturation and lower inference per unit starts to dominate, producing an inverted U.

Software shows the interaction-policy response especially clearly:

| $m$ | $s^{\star}$ | $x^{\star}$ | $k^{\star}$ | Adoption |
|---:|---:|---:|---:|---:|
| 0.5 | 0.59 | 1.26 | 4 | 25% |
| 1 | 0.89 | 0.88 | 4 | 48% |
| 2 | 1.36 | 0.62 | 4 | 65% |
| 4 | 2.06 | 0.36 | 5 | 75% |

The user delegates more work per checkpoint while spending fewer tokens per unit of that work. Total demand depends on whether adoption and work creation offset that intensive-margin efficiency.

### 9.3 Scenario B: token efficiency rises

| Token efficiency $\eta$ | Software | Review-bound | Routine | Frontier research |
|---:|---:|---:|---:|---:|
| 0.5 | 1.30 | 1.20 | 1.33 | 1.12 |
| 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 0.78 | 0.85 | 0.75 | 0.81 |
| 4 | 0.68 | 0.85 | 0.60 | 0.65 |

In the base calibration, higher token efficiency lowers token demand in every industry. Adoption moves, but not enough to offset fewer tokens per unit of work.

That conclusion is not structural. Consider an industry with the same software-like technology but tightly clustered adoption hurdles around the baseline surplus. Reoptimizing gives:

| $\eta$ | Adoption | Relative token demand |
|---:|---:|---:|
| 0.5 | 8.7% | 0.25 |
| 1 | 49.8% | 1.00 |
| 2 | 84.1% | 1.24 |
| 4 | 95.4% | 1.18 |

This is the rebound case. Efficiency initially raises token demand by triggering adoption. Once adoption saturates, the engineering effect reasserts itself.

### 9.4 Scenario C: verification becomes cheaper

Let $v$ multiply verification time. A value of $v=0.5$ means verification takes half as long. The table again reports work-limited token demand relative to $v=1$.

| Verification multiplier $v$ | Software | Review-bound | Routine | Frontier research |
|---:|---:|---:|---:|---:|
| 2 | 0.79 | 0.18 | 1.73 | 0.05 |
| 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.5 | 0.88 | 1.06 | 0.58 | 1.01 |
| 0.25 | 0.57 | 0.64 | 0.34 | 0.68 |

The review-bound and frontier industries initially show an adoption rebound: halving verification time sharply increases adoption and keeps demand near or above baseline. Further reductions eventually lower demand as users choose cheaper, shorter interactions and less inference.

In this calibration, cheaper verification often reduces $s^{\star}$. For software, $(s^{\star},x^{\star})$ moves from $(0.89,0.88)$ at $v=1$ to $(0.65,0.58)$ at $v=0.5$ and $(0.49,0.32)$ at $v=0.25$. Cheap checkpoints make short, reliable tasks economical; they do not mechanically force longer autonomy.

This table is work-limited. In the attention-limited regime, the direct $1/v$ increase in supervisory capacity pushes the other way. Which constraint binds is therefore empirically important.

## 10. Industry regimes

The model suggests that industries may occupy several distinct token-demand regimes.

| Regime | Typical parameters | Response to better models | Token-demand tendency |
|---|---|---|---|
| Mature, fixed workload | High existing $A_i$, low $\epsilon_i$ | $x^{\star}$ and retries fall | Declining |
| Capability threshold | Low $\lambda_i$ or high adoption threshold | $q_i$ and $A_i$ rise sharply | Rising, then potentially falling |
| Review-bound | $\beta_i\approx1$, high $w_ih_i$ | Autonomy grows slowly | Flat or hump-shaped |
| Machine-verifiable | Low $\beta_i$, abundant backlog | High $s_i^{\star}/h_i(s_i^{\star})$ | Potentially very large |
| Highly elastic work | High $\epsilon_i$ | New work appears as cost falls | Strong rebound possible |

This taxonomy explains why aggregate token demand can rise even while demand falls in many mature industries. The sum is dominated by industries crossing capability, adoption, or verification thresholds.

## 11. What would need to be estimated

The model is useful only if its industry surfaces can be measured. The main empirical objects are:

1. **Capability curves:** $q_i(s;m)$ across task horizons and task distributions.
2. **Inference-response curves:** $r_i(s,x;m,\eta)$ and empirical pass@$k$ behavior.
3. **Verification functions:** $h_i(s)$, including how tests, judges, and formal checks change its level and exponent.
4. **Economic primitives:** $b_i$, human attention cost $w_i$, and the cost of detected and undetected failures.
5. **Adoption distributions:** $G_i$ and the share of industry work near the adoption frontier.
6. **Work elasticity:** how $W_i$ changes when AI lowers the cost of producing an outcome.
7. **Binding constraints:** whether an industry is limited by available work, human attention, budgets, latency, data, or something else.

The highest-value behavioral data records the chosen policy $(s,x,k)$ together with success, human review time, repair time, and final acceptance. Tokens per task alone are not enough.

## 12. Limitations and extensions

The minimal model intentionally leaves out several important features:

- **Correlated retries within the solvable set.** An empirical pass@$k$ curve can replace conditional independence.
- **Imperfect verification.** Failures may be missed. A fuller model would choose review effort and include detectability, repair costs, and escaped-error losses.
- **Parallel agents and model routing.** Users may allocate inference across planners, workers, judges, and smaller models.
- **Lumpy value.** Completing an end-to-end project may be more valuable than the sum of small suggestions.
- **Dynamic learning.** Users, organizations, and models learn from prior attempts; integration costs can fall over time.
- **Frontier versus aggregate demand.** Tasks may graduate to cheaper models even as the capability frontier expands.
- **Compute rather than tokens.** Input, cached input, reasoning, and output tokens can have different hardware costs. Infrastructure forecasts ultimately require joules per token by model and token type.

These extensions should be added only when the data can identify them. The minimal model already captures the central economic ambiguity.

## Conclusion

Inference demand is an equilibrium outcome of a human interaction policy, not a fixed engineering requirement. Users decide how much work to delegate, how hard the model should think, and how many times it should try. Industries differ in capability barriers, inference returns, verification scaling, economic value, and adoption hurdles.

The most useful forecasting equation is

```math
D
=
\sum_i
N_i^{\star}s_i^{\star}x_i^{\star}E_i^{\star},
```

where the starred activity and policy must be solved under the constraint that actually binds in each industry. This makes the ambiguity transparent. Better models can reduce tokens per unit of existing work while increasing delegation horizons, adoption, supervisory leverage, and the amount of work worth doing. Different industries can therefore move in opposite directions at the same time. That is not a flaw in the model; it is the central phenomenon the model is designed to explain.

## Numerical implementation

The code mirrors the economics rather than hiding it inside a plotting notebook:

- `Industry` contains primitives that differ across industries, such as the capability frontier, verification technology, value of work, and adoption hurdles.
- `Scenario` contains the technology and market variables we want to vary: model capability, token efficiency, token price, and verification speed.
- `Policy` is the user's choice $(s,x,k)$, while `PolicyOutcome` records reliability, surplus per work-hour, surplus per attention-hour, adoption, and token demand.
- `IndustryModel` evaluates a candidate policy. It contains no optimization logic.
- `PolicyOptimizer` maximizes surplus per work-hour for the fixed-work polar case.
- `AttentionConstrainedOptimizer` maximizes surplus per human verification hour for the abundant-work, binding-attention polar case. Both optimizers enumerate the discrete retry cap and numerically optimize delegation horizon and inference intensity for each value of $k$.

This separation makes it difficult to accidentally optimize token demand itself. The user maximizes expected surplus; adoption and industry token demand are downstream outcomes.

```python
from modeling_token_demand import (
    AttentionConstrainedOptimizer,
    IndustryModel,
    PolicyOptimizer,
    Scenario,
)
from modeling_token_demand.calibrations import SOFTWARE

model = IndustryModel(SOFTWARE)
scenario = Scenario(
    model_capability=1.0,
    token_efficiency=1.0,
    token_price_per_million=10.0,
)

work_limited_optimum = PolicyOptimizer().solve(model, scenario)
attention_limited_optimum = AttentionConstrainedOptimizer().solve(model, scenario)
print(work_limited_optimum.policy)
print(attention_limited_optimum.policy)
```

The package measures $x$ and demand directly in tokens. `token_reference` on an industry is only a numerical normalization inside the execution-reliability curve; it does not change the unit of $x$ or reported demand.

### Running the code

```bash
uv sync --extra notebook --extra dev
uv run pytest
uv run jupyter lab notebooks/comparative_statics.ipynb
```

The [comparative-statics notebook](notebooks/comparative_statics.ipynb) documents the industry calibrations, re-solves the attention-constrained policy at every point, and writes six figures. Its demand and spending figures use the abundant-work, binding-attention polar case with 100,000 human verification hours per industry; this normalization scales the vertical axis without changing the curve shapes. A separate capability figure compares that solution with the earlier procedure that maximized surplus per work-hour before applying the attention cap.

![Attention-limited token demand versus token price](figures/token-demand-vs-price.png)

![Attention-limited token spend versus token price](figures/token-spend-vs-price.png)

![Attention-limited token demand versus token efficiency](figures/token-demand-vs-efficiency.png)

![Attention-limited token demand versus model capability](figures/token-demand-vs-capability.png)

![Capability demand under alternative policy objectives](figures/token-demand-vs-capability-objectives.png)

![Optimized user surplus versus model capability](figures/optimized-surplus-vs-model-capability.png)

The plotted industry calibrations are designed to expose qualitatively different attention and verification regimes, not to serve as empirical estimates. The package retains a separate near-adoption-threshold calibration for later work-limited comparisons, but it is intentionally excluded here because adoption parameters do not affect demand when attention binds and useful work is abundant.
