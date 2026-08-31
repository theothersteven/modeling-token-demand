# Modeling Token Demand

## Delegation, inference, retries, and scarce human attention

**Working draft**

## Abstract

A common forecast of future inference demand begins with a large pool of economic activity, such as aggregate labor income or corporate spending, and assumes that sufficiently capable models will capture some fraction of it. Such calculations can describe a potential market, but they do not explain the behavior that turns model progress into token purchases. In particular, they often leave implicit how much work a model can actually complete, how often execution fails, and how much human attention is still required to specify, guide, and verify the work.

We instead model inference demand from the user's interaction policy. A user chooses how much work to delegate before the next checkpoint, how many tokens to spend on each attempt, and how many attempts to allow. The model does not assume that an entire job or unit of work becomes automatable. It separates whether a task lies inside the model's capability frontier from whether one attempt executes a solvable task successfully, and it makes human verification an explicit resource. We characterize the resulting demand when available work is scarce, when human attention is scarce, and when both resources can bind.

The constraint determines the result. With fixed work, a finite retry cap can be optimal. With abundant interchangeable work and scarce attention, one attempt weakly dominates retries. A uniform reduction in verification time increases attention-limited token demand in exact inverse proportion and leaves the optimal interaction policy unchanged. Greater model capability raises the optimized value of scarce attention, with an elasticity determined by how verification scales with delegated scope. Token demand itself remains ambiguous: it rises only when greater adoption or supervisory leverage outweighs lower token use per unit of work.

## 1. Introduction

Many top-down forecasts start with an aggregate pool of wages, profits, or corporate spending and ask what fraction a capable AI system might capture. This approach can provide an upper bound on economic value, but it does not provide a demand curve for inference. It does not say which parts of the underlying work are feasible, how many attempts the work requires, how token use changes with delegated scope, or how much human attention is needed to keep the process moving.

Forecasting inference demand therefore requires more than estimating how many tokens a model uses on today's tasks. That calculation holds the user's interaction policy fixed. A user can instead respond to a better or cheaper model by changing the scope of delegated work, the inference devoted to each attempt, and the number of attempts made before abandoning a task. These choices determine both tokens per unit of work and the amount of work that can be supervised.

This creates a basic ambiguity. Model progress reduces the cost of producing a given result. The same progress can make larger delegated tasks worthwhile, move more work onto AI systems, and increase the amount of machine work supported by a fixed human review budget. A useful model of token demand must represent both effects within the same decision problem.

We study an industry-level user who chooses an interaction policy with three components:

- a delegation horizon, which is the amount of work assigned before the next human checkpoint;
- inference intensity, which is the number of tokens used per unit of delegated work in each attempt; and
- a retry cap, which is the maximum number of attempts allowed for one delegated chunk.

In practical systems, inference intensity corresponds to the amount of computation allocated to an attempt. In ChatGPT, it is similar to choosing a reasoning-effort setting such as low, medium, high, or xhigh; a higher-compute mode such as Pro is another example of moving along this margin. Choosing among smaller and larger models in the same family can also be represented through $x$ after converting raw tokens to a common compute-equivalent unit, so that one token from a larger model counts as multiple smaller-model-token equivalents. Any remaining difference in the set of tasks the models can solve belongs in capability $m$, while differences in effective inference per compute-equivalent token belong in efficiency $\eta$.

The analysis proceeds in the order needed to determine demand. Section 2 defines the policy, the capability and execution model, retry behavior, verification, and surplus. Section 3 derives optimal policies and token demand under a work constraint, an attention constraint, and both constraints together, taking technology and prices as given. Section 4 examines how those choices and demand respond to changes in capability, efficiency, token prices, and verification. Section 5 asks how faster verification, slower review growth, expanded task feasibility, and large token-efficiency gains change demand and automation. The numerical exercise is designed to expose model mechanisms; it is not an empirical forecast.

The main results are as follows. First, separating capability from execution prevents pass rates from converging mechanically to one and makes the value of retries depend on the unsolvable share of work. Second, the resource constraint changes the policy objective. Applying an attention cap after optimizing surplus per unit of work generally does not recover the attention-constrained solution. Third, when useful tasks are abundant and interchangeable, retrying a failed task cannot improve expected value per scarce verification hour. Fourth, model capability can raise the value of an additional attention hour even when optimized token demand falls. These results identify the objects that must be estimated before aggregate token demand can be forecast.

## 2. Model

Industries are indexed by $i\in\mathcal I$. One unit of underlying work is measured in human-hour-equivalents. The unit fixes the scale of the model; it does not assume that a human must perform the work.

### 2.1 Interaction policy and technology

A user in industry $i$ chooses a policy $(s,x,k)$:

- $s>0$ is the **delegation horizon**, measured in units of underlying work between human checkpoints;
- $x>0$ is **inference intensity**, measured in tokens per unit of work in one attempt; and
- $k\in\mathbb N$ is the maximum number of attempts allowed for one delegated chunk.

One attempt on a chunk of size $s$ consumes

```math
T(s,x)=sx
```

tokens. Increasing $s$ changes the scope of work assigned between checkpoints. Increasing $x$ changes the computation devoted to each unit of that work. The distinction is important because a model can support longer delegation without requiring the same proportional increase in tokens per unit of work.

Four scenario variables describe technology and cost. Model capability is $m>0$, token efficiency is $\eta>0$, the price of one token is $c>0$, and the verification-time multiplier is $v>0$. A larger $\eta$ means that each token produces more effective inference. The constant $x_{\mathrm{ref}}>0$ is a numerical reference level used only to make inference intensity dimensionless inside the execution function.

### 2.2 Capability and execution

A failed attempt can have two different causes. The task may be outside the model's capability frontier, or the task may be solvable but poorly executed on that attempt. The model represents these events separately.

The share of tasks of horizon $s$ that lie inside industry $i$'s capability frontier is

```math
q_i(s;m)
=
\exp\left[-\left(\frac{s}{\lambda_i m}\right)^{\nu_i}\right].
```

The parameter $\lambda_i>0$ sets the horizon at which capability begins to bind, and $\nu_i>0$ controls how sharply the solvable share falls with delegated scope. Greater capability shifts the frontier outward.

Conditional on a task being solvable, one attempt succeeds with probability

```math
r_i(s,x;m,\eta)
=
\exp\left[
-\frac{s}
{a_i m\left[\eta(x/x_{\mathrm{ref}})\right]^{\alpha_i}}
\right].
```

The parameter $a_i>0$ measures execution ease. The parameter $0<\alpha_i<1$ gives diminishing returns to inference intensity. Conditional reliability falls with the delegation horizon and rises with capability, token efficiency, and inference intensity. The unconditional success probability of the first attempt is $q_i r_i$.

Capability and token efficiency play different roles. Capability $m$ expands the solvable set through $q_i$ and improves execution through $r_i$. Token efficiency $\eta$ changes how much effective inference is obtained from a token, but it does not directly move the capability frontier.

There is no single industry parameter for the "return on intelligence." The scale parameters $\lambda_i$ and $a_i$ describe baseline task tractability: $\lambda_i$ sets how far the capability frontier extends, while $a_i$ measures how easily a solvable task is executed. They are not return coefficients for $m$; larger values make a given task easier before capability improves. At a fixed policy, the local responses to capability are

```math
\frac{\partial\log q_i}{\partial\log m}
=
\nu_i\left(\frac{s}{\lambda_i m}\right)^{\nu_i},
\qquad
\frac{\partial\log r_i}{\partial\log m}
=
\frac{s}{a_i m[\eta(x/x_{\mathrm{ref}})]^{\alpha_i}}.
```

Thus the return to higher $m$ is greatest where the capability or execution constraint is locally tight. The shape parameter $\nu_i$ directly governs the frontier response, while $\alpha_i$ is the direct return to additional inference intensity $x$. An industry's realized gain from higher model capability depends jointly on these parameters and on how the user changes $(s,x,k)$.

### 2.3 Bounded retries

Suppose attempts on a solvable task explore independent execution paths, each of which succeeds with probability $r_i$. The user stops after the first success or after $k$ failures. The probability of success within $k$ attempts is

```math
P_i(s,x,k;m,\eta)
=
q_i\left[1-(1-r_i)^k\right].
```

Therefore

```math
\lim_{k\rightarrow\infty}P_i=q_i.
```

Retries can repair execution failures, but they cannot move a task into the solvable set.

The expected number of attempts consumed by one delegated chunk is

```math
E_i(s,x,k;m,\eta)
=
(1-q_i)k
+q_i\frac{1-(1-r_i)^k}{r_i}.
```

The first term covers tasks outside the capability frontier, which consume all $k$ attempts. The second term is the truncated geometric expectation for solvable tasks. At $k=1$, exactly one attempt is consumed, so $E_i=1$. Expected token use per chunk is

```math
T_i^{\mathrm{exp}}=sxE_i.
```

The independence assumption is a minimal specification. An empirical pass-at-$k$ curve can replace it without changing the resource constraints or demand accounting below.

### 2.4 Verification, value, and surplus

Human verification time after one attempt is

```math
h_i(s;v)
=
v\left(h_{0,i}+h_{1,i}s^{\beta_i}\right).
```

Here $h_{0,i}\geq0$ is fixed checkpoint overhead, $h_{1,i}>0$ scales review that grows with delegated scope, and $\beta_i\geq0$ is the elasticity parameter governing that growth. The multiplier $v$ changes the level of verification time without changing its shape.

Let $b_i>0$ be the value of successfully completing one unit of work, and let $w_i>0$ be the opportunity cost of one hour of human attention. For a single attempt, token and verification cost per unit of delegated work are

```math
C_i(s,x;v)
=
cx+w_i\frac{h_i(s;v)}{s}.
```

Expected surplus per unit of work under policy $(s,x,k)$ is

```math
u_i(s,x,k)
=
b_iP_i(s,x,k)
-E_i(s,x,k)C_i(s,x;v).
```

Equivalently, a delegated chunk creates expected surplus $s u_i$. This is the common economic object used in each constraint regime. What changes across regimes is the resource against which that surplus is maximized.

The attention cost $w_i$ can represent a wage or the opportunity cost of the reviewer's time; it is not the value $b_i$ of the underlying work. Raising $b_i$ rewards successful output, whereas raising $w_i$ penalizes verification time, so the two can produce different policy responses when work is limited. With abundant work and binding attention, $w_i$ instead becomes a constant in the policy objective below: it affects whether operating is worthwhile, but not the chosen policy conditional on operating. The numerical comparisons hold $w_i$ fixed and vary economic value through $b_i$ alone.

## 3. User behavior under alternative constraints

This section derives the policy and token demand implied by each resource constraint at given technology and prices. Its accompanying figures reoptimize as those inputs vary, illustrating the demand paradigms explained analytically in Section 4.

The figures illustrate both regimes using **one shared set of fourteen cases**: the reference, five high/low pairs, and three singleton configurations. Each pair changes one related parameter group; the singletons expose additional shapes without implying a high/low ordering. Every unlisted parameter stays at its reference value. Gray denotes the reference; solid/dashed lines denote high/low settings within each color group, and dash-dot lines denote singletons. The industry comparisons in Sections 3 and 5 use this same set.

Parameters run down the rows; each condition has its own column. **Bold values differ from the reference.** The blocks share the same reference and are split only for readability. “Capability” means capability constraint, “execution” means execution difficulty, and “review” means verification burden.

**Technical conditions**

| Parameter | Reference | Capability low | Capability high | Execution low | Execution high | Review low | Review high |
|---|---:|---:|---:|---:|---:|---:|---:|
| Capability horizon $\lambda$ | 12 | **24** | **6** | 12 | 12 | 12 | 12 |
| Frontier shape $\nu$ | 1.25 | **1.5** | **1** | 1.25 | 1.25 | 1.25 | 1.25 |
| Execution ease $a$ | 4 | 4 | 4 | **8** | **2** | 4 | 4 |
| Inference returns $\alpha$ | 0.5 | 0.5 | 0.5 | **0.65** | **0.35** | 0.5 | 0.5 |
| Fixed review $h_0$ (hours) | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | **0.015** | **0.06** |
| Review scale $h_1$ | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | **0.025** | **0.1** |
| Review growth $\beta$ | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | **0.25** | **0.95** |

**Economic and adoption conditions**

| Parameter | Reference | Value low | Value high | Concentration low | Concentration high |
|---|---:|---:|---:|---:|---:|
| Work value $b$ (dollars) | 100 | **50** | **200** | 100 | 100 |
| Hurdle spread $\sigma$ (dollars) | 4 | 4 | 4 | **16** | **1** |
| Hurdle midpoint $\mu$ (dollars) | 84 | 84 | 84 | 84 | 84 |

**Singleton conditions**

| Parameter | Reference | Early saturation | Capability valley | Offsetting efficiency |
|---|---:|---:|---:|---:|
| Capability horizon $\lambda$ | 12 | **36** | **36** | **36** |
| Execution ease $a$ | 4 | **8** | 4 | **2** |
| Inference returns $\alpha$ | 0.5 | 0.5 | **0.25** | 0.5 |
| Fixed review $h_0$ (hours) | 0.03 | 0.03 | **0.001** | 0.03 |
| Review growth $\beta$ | 0.5 | 0.5 | **0.9** | **0.8** |
| Hurdle spread $\sigma$ (dollars) | 4 | **1** | 4 | 4 |

All cases hold $w_i=100$ dollars per verification hour, $W_i=1{,}000{,}000$ potential work hours in the work-limited regime, and $H_i=100{,}000$ verification hours in the attention-limited regime. The values $b_i$, $\mu_i$, and $\sigma_i$ are dollars per unit of underlying work. The common scenario baseline is $m=\eta=v=1$ and a token price of 10 dollars per million, with $x_{\mathrm{ref}}=100{,}000$ tokens per work hour. All cases use the common hurdle midpoint $\mu_i=84$; concentration changes only $\sigma_i$. The reference work value is now $b=100$, with low/high values 50 and 200. Lowering the reference value also lowers optimized baseline AI surplus to approximately 83, so the midpoint is recentered from 108 to 84. Keeping the old midpoint above $b$ would leave the concentrated market almost entirely unadopted even when tokens become cheap. This is an illustrative central calibration, not an estimated median industry.

"High" refers to the named feature, not to every parameter being numerically larger or to token demand necessarily being higher. In particular, **high concentration means a small spread** $\sigma$: high, reference, and low concentration use $1$, $4$, and $16$, respectively, with the same midpoint $\mu=84$. Below that midpoint, high concentration gives lower adoption; above it, high concentration gives higher adoption. The CDFs cross at 50% adoption, so concentration is not an ordering of hurdle difficulty. The notebook checks this crossing rather than imposing the old hurdle ordering.

Economic value uses equal multiplicative steps of $\times2$; adoption spread uses steps of $\times4$. The capability-horizon, execution-ease, and review-time scale settings already halve or double their reference values. Shape parameters such as inference returns and review growth are treated separately rather than mechanically doubled.

The technical pairs retain their same-policy difficulty ordering over the policies in each comparison; shape parameters such as $\nu$ and $\alpha$ are not globally ordered. The concentration pair changes only dispersion; the other paired comparisons change the listed parameters together. Singleton settings are explicitly outside those high/low comparisons.

The paired verification settings already span strong growth and a hump; simply making every high/low pair more extreme is not needed. The concentration pair supplies the adoption takeoff; the singletons supply early saturation, a smooth U-shape, and near-offsetting efficiency. Main plots cover $m=0.1$ to $30$, $\eta=0.25$ to $10$, and token prices from 1 to 80 dollars per million. The wider technology ranges expose both sides of turning points that the former 0.35–5 range could truncate. These are existence examples, not an exhaustive classification or empirical estimates of particular industries.

### 3.1 Limited work, nonbinding attention

First suppose industry $i$ has a fixed pool of potential work and enough human attention to review all economically adopted work. The user chooses

```math
(s_i^W,x_i^W,k_i^W)
\in
\arg\max_{s>0,\,x>0,\,k\in\mathbb N}
u_i(s,x,k).
```

Write $u_i^W$ for the optimized AI surplus. Adoption compares this surplus with the current alternative for each unit of work.

Assume that the minimum AI surplus $\phi$ required for a unit of work to switch is distributed logistically across work, with location $\mu_i$ and scale $\sigma_i>0$. This threshold can incorporate the surplus from human-only production together with switching, integration, and risk costs. The adoption share is therefore

```math
A_i
=
\Pr(\phi\leq u_i^W)
=
\frac{1}{1+\exp[-(u_i^W-\mu_i)/\sigma_i]},
```

where $\mu_i$ is the surplus at which half of potential work is adopted and $\sigma_i$ controls how gradually work switches to AI. More generally, write $A_i(s,x,k)$ for the same logistic expression evaluated at the candidate surplus $u_i(s,x,k)$.

#### Result 1: work-limited demand depends on adoption and tokens per unit of work

Token demand equals the amount of work adopted multiplied by expected tokens per unit of work. Let $W_i>0$ be potential work during the period. The number of adopted chunks is $W_iA_i/s_i^W$. Each chunk consumes $s_i^Wx_i^WE_i^W$ expected tokens. Work-limited token demand is therefore

```math
D_i^W
=
W_iA_ix_i^WE_i^W.
```

The delegation horizon cancels from this accounting identity. Grouping a fixed amount of work into larger chunks does not mechanically create more work. The horizon still matters because it changes capability, reliability, verification, retries, inference intensity, and adoption.

The numerical illustration uses all paired and singleton settings in the table above, holding potential work $W_i$ fixed. In particular, verification burden combines higher or lower checkpoint overhead, variable review time, and review growth in one group; it is not just a uniform rescaling of verification time.

Each curve reoptimizes $(s,x,k)$ and allows adoption to respond. Gray denotes the reference case. Each parameter group has one color, with a solid line for its high setting and a dashed line for its low setting; singletons have distinct colors and dash-dot lines. The first figure reports absolute demand under a common $W_i$. Both work-limited figures use independent y-axes: capability and efficiency are linear, while price is logarithmic. Otherwise, the concentrated market's near-zero adoption tail would compress the meaningful variation in every other curve.

![Work-limited token demand in levels](figures/work-limited-token-demand-levels.png)

The second figure divides each curve by its own demand at the common scenario baseline: $m=1$, $\eta=1$, or a token price of 10 dollars per million. For an axis variable $z$ with fixed baseline $z_0$, the plotted index is

```math
\widetilde D_i^W(z)=\frac{D_i^W(z)}{D_i^W(z_0)}.
```

The denominator is one constant for each curve, not the reference industry's demand evaluated at every $z$. This removes level differences but preserves monotonicity, turning points, percentage changes, and log slopes. In particular, it does not remove the shape of the reference industry.

![Indexed work-limited token demand](figures/work-limited-token-demand-indexed.png)

The indexed price panel includes a black dash-dot **constant-revenue benchmark**, $c_0/c$, where $c_0=10$ dollars per million tokens. Above that line, an industry's token revenue exceeds its revenue at the baseline price; below it, revenue is lower. For example, halving the token price requires twice the baseline demand to preserve revenue. Indexing lets the same benchmark apply to every industry despite their different demand levels. This compares revenue with the baseline, rather than predicting the effect of every further small price cut.

#### Adoption takeoff, saturation, and efficiency-induced backfire

All fourteen cases appear in the main figures above. The following focused view isolates the reference, high adoption concentration, and early saturation from the same table, making their shapes easier to read in print. Capability and efficiency range from $0.25$ to $10$; price ranges from 1 to 80 dollars per million tokens. Each curve is indexed to its own demand at the usual baseline. The capability and efficiency panels use linear y-axes to expose peaks without giving nearly unadopted market tails disproportionate visual weight; the price panel uses a logarithmic y-axis. Their y-axes are independent.

![Work-limited demand paradigms across capability, efficiency, and price](figures/paradigm-work-demand.png)

Halving price from 10 to 5 dollars per million in the high-concentration case raises adoption from **27.0% to 49.0%**, demand **3.10 times**, and revenue **1.55 times**. A price cut from 10 to 1 dollars per million raises adoption from approximately **27.0% to 79.4%**, token quantity by **12.7 times**, and token revenue by **1.27 times**. This is a genuine market-unlocking response under the existing model. The median hurdle sits just above baseline optimized surplus, and the narrow distribution places substantial work close to that threshold. The wider spread $\sigma=1$ makes this transition less abrupt than the previous $\sigma=0.4$ example, while preserving an economically meaningful rebound. The reference's broader adoption distribution spreads switching decisions over a much larger surplus range.

The same case has a visible efficiency hump: token demand peaks at approximately $\eta=2.88$, at **1.60 times** baseline demand, before falling to **1.27 times** baseline at $\eta=10$. Initial efficiency improvements induce enough adoption to more than offset fewer tokens per adopted work unit; after most work adopts, further efficiency increasingly saves tokens. Capability produces a separate hump, peaking near $m=1.58$ at **2.64 times** baseline demand. These are sampled turning-point locations, not exact analytical thresholds.

Increasing baseline task tractability does not necessarily strengthen this rebound. Early saturation has the same $(\mu,\sigma)$ but a broader capability frontier and easier execution. Its adoption is approximately **99.44%** at the scenario baseline: much of its market-unlocking response has already occurred. Capability therefore shifts its peak earlier, near $m=0.63$, and further capability growth reduces demand. Efficiency predominantly reduces token use, with discrete retry-policy jumps superimposed. High capability combined with narrow adoption spread can thus remove the remaining extensive margin rather than amplify it.

![Adoption saturation and token revenue as tokens become cheaper](figures/paradigm-adoption-and-revenue.png)

The adoption panel measures **work assigned to AI**, $WA$, as a percentage of potential work; successful AI output is $WAP$. Neither quantity is token demand. Adoption approaches a ceiling while tokens per adopted work unit can continue increasing as price falls. Revenue can first rise and then fall even while token quantity continues to rise. Thus an adoption plateau must not be read as a token-demand plateau, and a revenue hump is not a violation of downward-sloping token demand.

#### Result 2: retries can be worthwhile, but the optimal cap is finite

For fixed $(s,x)$, the decision after a failure is local: the user should make one more attempt only when its conditional expected benefit exceeds its cost. The relevant success probability changes after every failure because failure is evidence that the task may be outside the capability frontier.

Write $P_{i,k}=q_i[1-(1-r_i)^k]$. After the first $k$ attempts fail, the posterior probability that the task is solvable is

```math
\widehat q_{i,k}
:=
\Pr(\text{solvable}\mid k\text{ failures})
=
\frac{q_i(1-r_i)^k}{1-P_{i,k}}.
```

The conditional probability that the next attempt succeeds is therefore $\widehat q_{i,k}r_i$. At the decision point, one more attempt is worthwhile if and only if

```math
b_i\widehat q_{i,k}r_i
>
C_i(s,x;v).
```

At $k=0$, the posterior equals the prior, so the rule for starting a fresh task is $b_iq_ir_i>C_i$. After failures, $\widehat q_{i,k}<q_i$ whenever $q_i<1$. Multiplying the local condition by the probability $1-P_{i,k}$ of reaching the decision point gives

```math
b_iq_ir_i(1-r_i)^k
>
(1-P_{i,k})C_i(s,x;v).
```

This is also the ex ante comparison between retry caps $k$ and $k+1$, because

```math
P_{i,k+1}-P_{i,k}
=
q_ir_i(1-r_i)^k.
```

When $b_ir_i>C_i$ and $q_i<1$, either form of the condition is equivalent to

```math
(1-r_i)^k
>
\frac{C_i(1-q_i)}{q_i(b_ir_i-C_i)}.
```

The right side is positive whenever some tasks are unsolvable, while the left side tends to zero as $k$ grows. The inequality must therefore eventually fail. If it holds at $k=1$, allowing a second attempt improves surplus over stopping after one. Once it fails, the posterior solvable probability only falls further, so no later retry becomes worthwhile at the same $(s,x)$.

#### Optimal inference and delegation

For a fixed retry cap, suppose the optimal inference intensity is not pinned to a lower or upper constraint. Such an **interior choice** allows the user to make a small increase or decrease in $x$. Its marginal benefit must therefore equal its marginal cost:

```math
b_iP_{i,x}
=
E_{i,x}C_i+E_ic,
```

where a subscript denotes a partial derivative. The left side is the marginal value of more inference. The right side includes its direct token cost and its effect on the expected number of attempts. If the optimum is at a boundary instead, this equality need not hold.

If the optimal delegation horizon is also interior, it satisfies

```math
b_iP_{i,s}
=
E_{i,s}C_i
+E_iw_i\frac{s h_{i,s}-h_i}{s^2}.
```

For $h_i(s;v)=v(h_{0,i}+h_{1,i}s^{\beta_i})$ with $\beta_i\leq1$,

```math
s h_{i,s}-h_i
=
-v\left[h_{0,i}+(1-\beta_i)h_{1,i}s^{\beta_i}\right]
\leq0.
```

Longer chunks therefore reduce verification cost per unit of work, while capability and execution reliability move in the opposite direction. The optimal horizon balances these forces.

### 3.2 Limited attention, abundant work

Now suppose useful work is abundant and interchangeable, but only $H_i>0$ human verification hours are available during the period. Let $N_i\geq0$ be the number of delegated chunks. The user solves

```math
\max_{s,x,k,N_i\geq0}
N_i s u_i(s,x,k)
```

subject to

```math
N_iE_i(s,x,k)h_i(s;v)\leq H_i.
```

If operating creates positive value and attention binds, then

```math
N_i
=
\frac{H_i}{E_i(s,x,k)h_i(s;v)}.
```

Substitution gives the policy objective

```math
J_i(s,x,k)
=
\frac{s u_i(s,x,k)}{E_i(s,x,k)h_i(s;v)}
=
\frac{s}{h_i(s;v)}
\left[
\frac{b_iP_i(s,x,k)}{E_i(s,x,k)}-cx
\right]
-w_i.
```

Thus the attention-constrained policy is

```math
(s_i^H,x_i^H,k_i^H)
\in
\arg\max_{s>0,\,x>0,\,k\in\mathbb N}J_i(s,x,k).
```

The attention endowment $H_i$ scales activity but does not change this policy in the single-industry polar case. The opportunity cost $w_i$ enters as a constant. It determines participation but does not change the maximizing policy once attention binds.

#### Result 3: retries do not improve value per scarce attention hour

For any fixed $(s,x)$,

```math
\frac{P_{i,k}}{E_{i,k}}
\leq
q_ir_i
=
\frac{P_{i,1}}{E_{i,1}}.
```

The inequality follows from $1-(1-r_i)^k\leq kr_i$. The token-cost term in $J_i$ does not depend on $k$, because both expected token use and expected verification time scale with $E_i$. Therefore $k=1$ weakly dominates every larger retry cap.

The result depends on abundant interchangeable work. After a failure, beginning a fresh task draws again from the original capability distribution, while retrying the same task conditions on evidence that it may lie outside the capability frontier. Retries can return when work is finite, abandonment destroys value, tasks differ in value, later attempts learn from earlier failures, or setup costs can be reused.

#### Result 4: attention-limited demand depends on supervisory leverage

The industry attempts $H_i/[E_i^Hh_i(s_i^H;v)]$ chunks. Multiplying by expected tokens per chunk gives

```math
D_i^H
=
H_i\frac{s_i^Hx_i^H}{h_i(s_i^H;v)}.
```

Expected attempts cancel from demand because each additional attempt consumes both another token budget and another checkpoint. Demand per human attention hour is

```math
\frac{D_i^H}{H_i}
=
\underbrace{\frac{s_i^H}{h_i(s_i^H;v)}}_
{\text{work launched per attention hour}}
\times
\underbrace{x_i^H}_
{\text{tokens per unit of work}}.
```

This decomposition isolates the attention channel: a fixed human attention budget supports token use through both the work launched per attention hour and the inference devoted to each unit of work.

The following figures mirror Section 3.1: three panels vary model capability, token efficiency, and token price, with all other scenario inputs fixed. Each point solves the attention-constrained problem directly, using the same $H_i=100{,}000$ verification hours for every case. Gray denotes the reference industry; each parameter group has one color, with solid and dashed lines for its high and low settings, while singletons use dash-dot lines.

These figures use exactly the same five parameter-group pairs as Section 3.1, including the combined verification-burden settings. The first figure reports absolute token demand.

![Attention-limited token demand in levels](figures/attention-limited-token-demand-levels.png)

The second divides each curve by its own demand at $m=1$, $\eta=1$, or a token price of 10 dollars per million. It compares proportional changes from that fixed scenario baseline; a line above gray need not have higher absolute demand. Verification-burden changes now affect both levels and response shapes because they also change $\beta_i$. Adoption-concentration changes are inactive in this abundant-work regime, so those lines overlap the reference in both figures.

![Indexed attention-limited token demand](figures/attention-limited-token-demand-indexed.png)

As in Section 3.1, the black dash-dot line in the indexed price panel is $c_0/c$. A curve above it has more token revenue than at its own baseline price; a curve below it has less. To the right of the baseline, this shows whether demand growth offsets the price cut.

The verification-burden pair changes the capability response itself. Raising capability from $m=1$ to $m=5$ multiplies demand by about $3.63$ in the low-burden case, $2.40$ in the reference, and $0.89$ in the high-burden case. When fixed verification overhead is negligible, the model approximately scales as $D_i^H(m)\propto m^{1-\beta_i}$: larger delegated tasks create more supervisory leverage when review grows slowly with scope. With positive fixed overhead, the optimal inference response can also matter enough to reverse demand growth when review is nearly proportional to scope. These comparisons change all three verification parameters together; the separate uniform-time experiment in Section 4.4 isolates the pure level effect.

#### Attention-limited growth, a hump, and a smooth U-shape

The next panels isolate the low and high verification-burden settings and the capability-valley singleton from the same table, using independent y-scales to show their shapes rather than compare magnitudes. The first two cover $m=0.25$ to $10$. The capability-valley panel uses the explicitly wider range $m=0.1$ to $30$ to show both sides of its trough. Other scenario variables remain at baseline in every panel.

![Attention-limited capability paradigms: growth, a hump, and a valley](figures/paradigm-attention-capability.png)

The low-burden pair member displays sustained growth, while the high-burden member has a hump and then declining demand. These are exactly the verification settings in the main plots, not separate leverage and bottleneck calibrations. The capability-valley case falls approximately **12.9%** from $m=0.1$ to its sampled trough near $m=1.61$, then rises approximately **16.9%** by $m=30$.

The valley does not require an adoption threshold or a retry switch: $k=1$ throughout. Its local requirement is that

```math
\frac{d\log D_i^H}{d\log m}
=
[1-\theta_i(s_i^H)]\frac{d\log s_i^H}{d\log m}
+\frac{d\log x_i^H}{d\log m}
```

changes from negative to positive: inference savings dominate first, and supervisory leverage dominates later. Here $\theta_i(s)=d\log h_i(s)/d\log s$. With zero fixed overhead and an interior optimum, the exact scaling $D_i^H\propto m^{1-\beta_i}$ rules out a U-shape for $0\leq\beta_i\leq1$. Positive fixed overhead breaks that scale invariance, but is not by itself sufficient. The table's $(\lambda,\alpha,h_0,\beta)=(36,0.25,0.001,0.90)$ is one verified combination, not a necessary-and-sufficient characterization.

Small fixed checkpoint overhead and near-linear variable review initially allow reductions in inference intensity to dominate the growth in work launched per attention hour. At higher capability, the positive supervisory-leverage contribution dominates. Optimized value per attention hour increases throughout even when token quantity falls. The two-sided valley persists under separate 10% increases and decreases in fixed review overhead, inference returns, and the capability-horizon parameter.

The **offsetting-efficiency** singleton adds a near-flat case to the main efficiency panel: demand varies by less than 10% over $\eta=0.25$ to $10$. This is a finite-range offset between supervisory leverage and inference savings, not a hard token-demand ceiling. In the interactive main figure, isolate this case or the capability valley and use **Fit visible** to inspect small changes.

### 3.3 Work and attention can both bind

The two polar cases isolate the mechanisms, but they generally imply different policies. It is therefore incorrect to optimize surplus per unit of work and then apply an attention cap when attention changes the user's choices.

For a candidate policy, the maximum feasible number of chunks is

```math
N_i(s,x,k)
=
\min\left\{
\frac{W_iA_i(s,x,k)}{s},
\frac{H_i}{E_i(s,x,k)h_i(s;v)}
\right\}.
```

The joint problem is

```math
\max_{s,x,k,N_i\geq0}
N_i s u_i(s,x,k)
```

subject to

```math
N_i s
\leq
W_iA_i(s,x,k),
\qquad
N_iE_i(s,x,k)h_i(s;v)
\leq
H_i.
```

After solving this problem, token demand is

```math
D_i=N_i^*s_i^*x_i^*E_i^*.
```

The work-limited and attention-limited models are the two cases in which one of the capacity constraints is slack. Aggregate demand is

```math
D=\sum_{i\in\mathcal I}D_i.
```

## 4. How technology and prices affect token demand

Section 3 derived the user's policy for given technology and prices. Here we change one input at a time, hold the other inputs and resource endowments fixed, and compare outcomes after the user chooses a new optimal policy. These comparisons do not model how quickly users adjust.

The model gives strong conclusions about feasibility and optimized value. It gives weaker conclusions about token demand because users can change delegation, inference intensity, retries, and adoption.

| Change | Guaranteed effect at a fixed policy or effective inference level | Effect on optimized token demand |
|---|---|---|
| Capability $m\uparrow$ | $q_i$ and $r_i$ rise; $P_i$ rises and $E_i$ falls | Ambiguous |
| Token efficiency $\eta\uparrow$ | The same effective inference can be produced with fewer tokens; optimized surplus cannot fall | Ambiguous |
| Token price $c\downarrow$ | The feasible set is unchanged and optimized surplus cannot fall | Token quantity weakly rises in either pure regime under the standard revealed-preference comparison; spend is ambiguous |
| Verification multiplier $v\downarrow$ | Verification uses less human time; optimized surplus cannot fall | Exactly proportional to $1/v$ in the pure attention-limited regime; otherwise policy-dependent |

### 4.1 Model capability

For every fixed policy, greater $m$ raises the solvable share $q_i$ and conditional success $r_i$. It therefore raises eventual success $P_i$ and reduces expected attempts $E_i$. The user can respond by choosing a longer horizon, less inference per unit of work, or a different retry cap.

#### Result 5: capability raises the value of scarce attention

Define gross value per verification hour, after token spending but before subtracting $w_i$, as

```math
R_i(s,x,k)
=
J_i(s,x,k)+w_i.
```

Its optimized value is

```math
\rho_i^*(m)
=
\max\left\{0,\max_{s,x,k}R_i(s,x,k;m)\right\}.
```

If $h_{0,i}>0$, $0\leq\beta_i\leq1$, $\rho_i^*>0$, and the optimal attention-constrained values of $s$ and $x$ lie strictly inside their feasible ranges, then $k_i^H=1$ and, holding $c$, $\eta$, and $v$ fixed, the capability elasticity of the optimized gross value is

```math
\frac{d\log\rho_i^*}{d\log m}
=
1-\theta_i(s_i^H),
```

where

```math
\theta_i(s)
:=
\frac{s h_{i,s}(s;v)}{h_i(s;v)}
=
\frac{\beta_i h_{1,i}s^{\beta_i}}
{h_{0,i}+h_{1,i}s^{\beta_i}}.
```

Since $0\leq\theta_i(s)\leq\beta_i$, the elasticity lies between $1-\beta_i$ and $1$. Capability is most valuable at the margin when verification grows slowly with delegated scope.

#### The effect on token demand remains ambiguous

The value result does not determine token use: $\rho_i^*$ can rise while $D_i^H$ falls if the user reduces $x_i^H$ sufficiently. The two resource regimes make the relevant tradeoffs clear.

In the work-limited regime,

```math
D_i^W=W_iA_ix_i^WE_i^W.
```

Demand rises when adoption grows faster than inference intensity and expected attempts fall. Once adoption is saturated, the reduction in $x_i^WE_i^W$ can dominate.

In the attention-limited regime,

```math
D_i^H=H_i\frac{s_i^H}{h_i(s_i^H;v)}x_i^H.
```

Demand rises when capability expands supervisory leverage $s_i^H/h_i(s_i^H;v)$ faster than chosen inference intensity falls. The elasticity result for $\rho_i^*$ shows why low verification elasticity makes capability economically valuable, while the demand equation shows why that fact alone does not determine token use.

### 4.2 Token efficiency

Let effective inference intensity be $z=\eta x$. Holding $z$ fixed, a rise in $\eta$ allows the user to set $x=z/\eta$. Capability, reliability, and expected attempts are unchanged, while token cost falls. Optimized surplus therefore cannot decrease.

Token demand may increase through adoption or a longer delegation horizon. In the attention-limited regime,

```math
\frac{d\log D_i^H}{d\log\eta}
=
\frac{d\log[s_i^H/h_i(s_i^H;v)]}{d\log\eta}
+
\frac{d\log x_i^H}{d\log\eta}.
```

A Jevons-style increase in token use occurs only when the growth in supervisory leverage is larger than the decline in token intensity. In the work-limited regime, the corresponding extensive margin is adoption $A_i$.

### 4.3 Token price and token spend

A lower token price is a movement along the token demand curve. In either pure regime, the feasible set is fixed, so revealed preference implies that the optimized token quantity is weakly higher at a lower price. This result does not determine spending.

Let

```math
S(c)=cD(c)
```

be token spend, equivalently revenue received by token suppliers. For a small price reduction, spend rises only when token demand increases more than proportionally:

```math
\frac{d\log D}{d\log c}<-1.
```

For a finite change from baseline price $c_0$, the exact revenue comparison is

```math
\frac{S(c)}{S(c_0)}
=\frac{c}{c_0}\frac{D(c)}{D(c_0)}.
```

Revenue is therefore constant when indexed demand equals $c_0/c$, the black dash-dot line in the indexed price plots. A curve above this benchmark has higher revenue than at $c_0$, and one below it has lower revenue. For a price cut, being above the line means the demand increase more than offsets the lower price over that interval. It need not mean that the local elasticity exceeds one in magnitude at every point along the curve.

Price and efficiency answer different questions. A lower $c$ changes the budget tradeoff for the same token technology. A larger $\eta$ changes the amount of effective inference produced by each token. Jevons' paradox concerns efficiency improvements that increase total resource use; see [Gillingham, Rapson, and Wagner (2014)](https://media.rff.org/documents/RFF-DP-14-39.pdf). In this model, its counterpart is token demand rising when $\eta$ increases with $c$ and $m$ held fixed. Higher revenue after a pure token-price cut is an elastic demand response, not by itself evidence of Jevons' paradox.

### 4.4 Verification technology

The level and shape of verification have different effects. A change in $v$ multiplies all verification time by the same factor.

#### Result 6: uniformly faster verification scales up attention-limited demand

If verification takes half as long, the same attention budget supports twice as many tokens without changing the optimal policy, provided attention binds in both cases. In the pure attention-limited regime,

```math
J_i(s,x,k;v)
=
\frac{1}{v}R_i(s,x,k;1)-w_i.
```

The factor $1/v$ does not change the maximizing policy. Consequently,

```math
D_i^H(v)
=
\frac{1}{v}D_i^H(1).
```

This exact result applies only when useful work is abundant and attention is the binding constraint at both values of $v$. In the work-limited or joint problem, lower verification time changes per-unit cost, adoption, and the optimal policy.

#### Changes in how verification scales with scope

Changing $\beta_i$ is different. It changes how verification scales with the delegation horizon, alters the policy itself, and changes both supervisory leverage and the capability elasticity of scarce attention.

## 5. Which improvements increase token demand and automation?

What happens if verification gets faster, a harness makes more tasks feasible, or each token buys much more effective inference? Section 3 varies capability, efficiency, and price across industry configurations. This section instead isolates specific improvements and asks whether they increase **token demand, work delegated to AI, and work successfully completed**. Those outcomes need not move together.

Every figure has the same layout. The **top row fixes potential work** at $W=1{,}000{,}000$ hours and lets adoption respond; the **bottom row fixes attention** at $H=100{,}000$ verification hours with abundant work. The columns show token demand, work delegated, and work completed successfully. In the work-limited row, delegated work is the adoption share $A_i$, while completed work is $A_iP_i$, both expressed as percentages of potential work. In the attention-limited row, delegated and completed work are throughput, not market adoption rates:

```math
L_i^H=\frac{H_i s_i^H}{E_i^H h_i(s_i^H)},
\qquad
C_i^H=L_i^H P_i^H.
```

Token demand and attention-limited throughput are indexed to each curve's own baseline; the two work-share panels use percentages. Baselines are $v=1$, $\beta=0.5$, an improvement factor of 1, and $\eta=1$, respectively. All panels use independent linear y-axes. Except for the parameter varied on the x-axis and the explicitly labeled comparison, inputs stay at the Section 3 reference. Policies are reoptimized at every point. “Completed work” means expected successful AI output with human verification still required; it is not a measure of fully autonomous jobs or labor displacement.

### 5.1 What if verification took half as long?

Reduce the uniform verification-time multiplier $v$, holding $h_0$, $h_1$, and $\beta$ fixed. Unlike changing review growth, this makes every checkpoint faster regardless of task size. Moving right in the figure reduces verification time from twice the baseline to one tenth of it.

![What if verification took less time? Token demand, delegated work, and completed work under each constraint](figures/intervention-verification-speed.png)

With scarce attention, halving $v$ **doubles token demand, delegated work, and completed work**, without changing $(s,x,k)$ or success per chunk. This is the exact scaling result from Section 4.4. With fixed work, there is no such proportional response: cheaper verification changes the policy and adoption. In the reference calibration, halving $v$ raises adoption from **43.8% to 79.8%** and the successfully completed share from **42.2% to 77.8%**, while token demand rises only **9.6%**. More automation can therefore accompany a much smaller increase in tokens. The work-limited demand jumps reflect changes in the optimal retry cap.

### 5.2 What if review grew more slowly with task size?

Reduce $\beta$ while holding the verification-time level parameters fixed. Compare $m=1$ with $m=5$ to ask whether slower review growth matters more when the model supports longer tasks. Moving right lowers $\beta$ from 0.95 to 0.1; each curve is indexed at $\beta=0.5$.

![What if review grew more slowly? Varying beta at two capability levels](figures/intervention-review-growth.png)

With scarce attention, reducing $\beta$ from 0.5 to 0.25 raises token demand to **1.61 times** baseline at $m=1$ and **2.44 times** baseline at $m=5$. Completed work rises to **1.32 times** and **1.90 times** baseline, respectively. Slower review growth makes larger chunks less costly to supervise, especially when capability already supports a longer delegation horizon. The demand increase is larger than the completed-work increase because the optimized policy also changes inference intensity and reliability.

There is an important qualification to calling lower $\beta$ “easier verification.” In $h(s)=v(h_0+h_1s^\beta)$, review time is unchanged at $s=1$ hour. Lower $\beta$ reduces review time for longer chunks but increases it for shorter chunks. The work-limited reference chooses chunks shorter than one hour near baseline: lowering $\beta$ to 0.25 slightly **reduces** adoption, from 43.8% to 42.7%. At $m=5$, the chosen chunks are longer and adoption instead rises from 88.2% to 90.1%. This experiment isolates review growth; the preceding $v$ experiment is the appropriate comparison for an improvement that saves verification time on every task.

### 5.3 What if a better harness made more tasks feasible?

Represent a harness that expands the feasible task set by increasing $\lambda$ alone, with execution ease $a$, inference returns $\alpha$, and model capability $m$ held fixed. The x-axis is the proportional improvement: a factor of five raises $\lambda$ from 12 to 60 hours. Compare this with increasing $m$ by the same factor.

![What if a harness expanded feasibility? Increasing lambda alone versus increasing model capability](figures/intervention-harness-feasibility.png)

These interventions produce the **same capability share $q$ at a fixed horizon**, because it depends on the product $\lambda m$. They are not equivalent for demand: increasing $m$ also improves conditional execution $r$, while increasing $\lambda$ alone does not. The harness experiment consequently isolates feasibility; a real harness that also improves execution or verification would require changing those parameters separately.

With fixed work, a fivefold increase in $\lambda$ raises token demand to **2.36 times** baseline and adoption to **75.0%**. A fivefold increase in $m$ instead lowers token demand to **0.85 times** baseline even though adoption reaches **88.2%** and more work is completed. Expanding the feasible set can create demand for additional inference; improving execution as well can save enough tokens per delegated unit to outweigh greater adoption. With scarce attention, both interventions increase demand in this calibration, but the broader capability improvement produces substantially more completed work.

Lowering $\nu$ would answer a different question about the *shape* of the frontier. From $\log q=-(s/(\lambda m))^\nu$, lower $\nu$ raises feasibility when $s>\lambda m$ but lowers it when $s<\lambda m$; it leaves $q=e^{-1}$ unchanged at the threshold. It therefore cannot represent a uniform expansion of feasible tasks. The $\lambda$ sweep supplies that clean comparison without combining a frontier shift with a shape change.

### 5.4 What if each token became 100 times more efficient?

Increase $\eta$ from 1 to 100, holding capability and the price per raw token fixed. Compare $\alpha=0.25$, $0.5$, and $0.75$, changing no other industry parameter. This asks how the response to efficiency depends on diminishing returns to inference, without also changing execution ease $a$ as in Section 3's execution-difficulty pair. Each curve is indexed to its own outcome at $\eta=1$.

![What if tokens became much more efficient? Varying eta at three inference-return settings](figures/intervention-efficiency-returns.png)

In all three cases, 100-fold efficiency leaves **fewer tokens consumed but more work successfully completed** than at baseline. The size of the token saving depends strongly on $\alpha$:

| Inference returns $\alpha$ | Fixed-work token demand at $\eta=100$ (baseline = 1) | Scarce-attention token demand at $\eta=100$ (baseline = 1) | Successfully completed share of potential work, $\eta=1\to100$ |
|---|---:|---:|---:|
| 0.25 | 0.48 | 0.86 | 43.1% → 57.7% |
| 0.50 | 0.35 | 0.43 | 42.2% → 61.9% |
| 0.75 | 0.18 | 0.22 | 43.5% → 63.9% |

Efficiency improves execution at a given token budget, but it does not expand the capability frontier $q$. Users respond by reducing raw token intensity, changing retries, and delegating more work. They do not simply divide token use by 100: those behavioral responses offset much of the mechanical saving. Work-limited demand can jump upward when the user switches to fewer, more intensive attempts, even as its longer-run path declines. These results concern this controlled reference experiment, not a universal ranking of $\alpha$ or a claim that efficiency always lowers demand; Section 3's concentrated-adoption case already supplies a counterexample over a smaller efficiency range.

The [analysis notebook](notebooks/comparative_statics.ipynb) regenerates all four experiments. The [Section 5 diagnostics](figures/interventions.json) record the inputs, policies, unindexed outcomes, and numerical checks, including wider-bound checks at endpoints, baselines, and sampled extrema. No policy bound or maximum retry cap binds in these figures.

## 6. Measurement and limitations

The model separates the empirical objects needed for a forecast:

1. **Capability:** estimate $q_i(s;m)$ over task horizons and model versions.
2. **Execution:** estimate $r_i(s,x;m,\eta)$ and empirical pass-at-$k$ curves.
3. **Verification:** estimate $h_i(s;v)$, including its fixed cost and elasticity with respect to delegated scope.
4. **Value and cost:** estimate $b_i$, $w_i$, and the location $\mu_i$ and scale $\sigma_i$ of adoption thresholds.
5. **Available resources:** measure potential work $W_i$ and human attention $H_i$ to determine which constraint binds.
6. **Behavior:** observe the chosen $(s,x,k)$ together with success, review time, repair time, abandonment, and final acceptance.

Tokens per completed task are not sufficient. They combine policy, task selection, reliability, retries, and the binding resource constraint.

The current specification has several limitations. Attempts are conditionally independent within the solvable set. Verification is perfect and occurs after every attempt. Work within an industry has a common value apart from its reduced-form adoption hurdle. The model does not include learning across attempts, reusable setup costs, parallel attempts, latency, model routing, or a distinction among input, cached, reasoning, and output tokens. The joint work-and-attention problem is stated above, while the current numerical notebook focuses on the two polar objectives. These extensions should be introduced when data can identify them.

## 7. Conclusion

Token demand is the result of an optimized interaction policy. Users choose delegated scope, inference intensity, and retries subject to the resource that constrains activity. The same model improvement can therefore reduce tokens per unit of fixed work while increasing adoption or the amount of work supported by scarce human attention.

The constraint must be part of the optimization problem. With fixed work, retries can be valuable and demand depends on adoption and tokens per unit. With abundant work and scarce attention, retries are weakly dominated and demand factors into supervisory leverage and inference intensity. When both work and attention can bind, policy and throughput must be solved jointly. This sequence turns the apparent ambiguity of token demand into a set of measurable economic questions.

## Appendix A. Notation

| Symbol | Definition |
|---|---|
| $\mathcal I$, $i$ | Set of industries and industry index |
| $s$ | Delegation horizon: units of work per checkpoint |
| $x$ | Tokens per unit of work in one attempt |
| $k$ | Maximum attempts per delegated chunk |
| $N_i$ | Number of delegated chunks during the period |
| $m$ | Model capability |
| $\eta$ | Effective inference per token |
| $c$ | Price per token |
| $v$ | Multiplicative verification-time factor |
| $x_{\mathrm{ref}}$ | Reference token intensity used to normalize the execution function |
| $\lambda_i$, $\nu_i$ | Scale and shape of the capability frontier |
| $a_i$, $\alpha_i$ | Execution scale and inference-return elasticity |
| $q_i$ | Share of tasks inside the capability frontier |
| $\widehat q_{i,k}$ | Posterior probability that a task is solvable after $k$ failed attempts |
| $r_i$ | One-attempt success probability conditional on solvability |
| $P_i$ | Success probability within $k$ attempts |
| $E_i$ | Expected attempts consumed |
| $h_{0,i}$, $h_{1,i}$, $\beta_i$ | Fixed verification time, variable verification scale, and verification elasticity parameter |
| $h_i$ | Verification hours per attempt |
| $b_i$ | Value per successfully completed unit of work |
| $w_i$ | Opportunity cost per human attention hour |
| $C_i$ | Attempt cost per unit of delegated work |
| $u_i$ | Expected surplus per unit of work |
| $J_i$ | Expected net surplus per human verification hour |
| $R_i$ | Gross value per verification hour before subtracting $w_i$ |
| $\rho_i^*$ | Optimized gross value of an additional verification hour |
| $\theta_i$ | Elasticity of verification time with respect to delegated scope |
| $\phi$ | Minimum AI surplus required for a unit of work to switch from its current alternative |
| $\mu_i$, $\sigma_i$ | Location and scale of the logistic distribution of adoption thresholds |
| $A_i$ | Share of potential work assigned to AI |
| $W_i$ | Potential units of work during the period |
| $H_i$ | Human verification hours available during the period |
| $D_i^W$, $D_i^H$, $D_i$ | Work-limited, attention-limited, and joint token demand |
| $D$ | Aggregate token demand |
| $S(c)$ | Aggregate token spend at price $c$ |

A superscript $W$ denotes the work-limited solution, a superscript $H$ denotes the attention-limited solution, and a star denotes the solution to the applicable optimization problem. Subscripts $s$ and $x$ on a function denote partial derivatives.

## Appendix B. Numerical implementation

The code separates the economic model from the numerical optimizer:

- `Industry` contains capability, execution, verification, value, adoption, and capacity parameters.
- `Scenario` contains $m$, $\eta$, $c$, and $v$.
- `Policy` contains $(s,x,k)$, while `PolicyOutcome` records the implied reliability, surplus, adoption, and token demand.
- `IndustryModel` evaluates a candidate policy without choosing it.
- `PolicyOptimizer` maximizes $u_i$ for the work-limited polar case.
- `AttentionConstrainedOptimizer` maximizes $J_i$ for the attention-limited polar case. It also implements the exact one-dimensional interior characterization used to verify the capability result.
- `illustrative_industries()` in `calibrations.py` supplies all fourteen Section 3 cases. `include_singletons=False` returns the reference and ten paired variants; `singleton_industries()` returns the three additional table rows. `work_paradigms()` and `attention_paradigms()` select subsets of this same table for the focused views.
- `paradigms.py` generates the focused views through the notebook and independently audits endpoints, baselines, and extrema of the main curves. `scripts/scan_paradigms.py` reproduces the exploratory search without changing the economic model.

```python
from modeling_token_demand import (
    AttentionConstrainedOptimizer,
    IndustryModel,
    PolicyOptimizer,
    Scenario,
)
from modeling_token_demand.calibrations import REFERENCE_INDUSTRY

model = IndustryModel(REFERENCE_INDUSTRY)
scenario = Scenario(
    model_capability=1.0,
    token_efficiency=1.0,
    token_price_per_million=10.0,
)

work_limited = PolicyOptimizer().solve(model, scenario)
attention_limited = AttentionConstrainedOptimizer().solve(model, scenario)
```

The package measures $x$ and demand in tokens. `token_reference` is the code parameter corresponding to $x_{\mathrm{ref}}$ and is only a normalization inside the execution function.

Run the checks and notebook with

```bash
uv sync --extra notebook --extra dev
uv run pytest
uv run jupyter lab notebooks/comparative_statics.ipynb
```

To repeat the broader configuration scan, run `uv run python scripts/scan_paradigms.py`. It writes the full search diagnostics to `build/paradigm-scan.json`; this optional exploration is not required for ordinary paper builds. The Section 3 figures and diagnostics and the controlled Section 5 experiments are regenerated by the notebook or the paper refresh command below.

### Search coverage and numerical checks

The reproducible [parameter scan](scripts/scan_paradigms.py) first crosses three frontier scales, three execution scales, and four verification elasticities: **36 technical configurations**, each with **15 adoption distributions** for the work-limited regime. It then examines **512 additional attention configurations**, varying fixed overhead, frontier size, verification elasticity, and inference returns over the wider capability range. Candidates with binding policy bounds or nonpositive attention operating value are not used as attention-side evidence. These grid counts describe search coverage, not frequencies in the economy, and the search is not exhaustive.

The scan also finds retry-driven demand jumps and more complicated sampled reversals. These should not be conflated with smooth U-shaped responses: replacing several cheap attempts with fewer more intensive attempts can change total tokens discontinuously. The figures join actual optimized samples without fitting or smoothing a curve.

Not every imaginable shape belongs on every axis. A pure token-price reduction cannot reduce optimized token quantity in either polar regime. Capability and efficiency can produce nonmonotonic quantities, while price cuts can produce nonmonotonic spending. Near-flat behavior over a finite range is not evidence of an asymptotic demand ceiling. Adoption saturation is an explicit feature of the work-limited model; a token-demand ceiling imposed by an optimizer bound is not an economic finding.

The main and focused figures use 81 logarithmically spaced samples per axis plus exact baseline and comparison anchors. No continuous policy bound or maximum retry cap binds. Endpoints, baselines, and sampled extrema are rechecked with a denser multistart search, tenfold wider continuous bounds, and a higher retry cap; attention solutions are additionally compared with the exact scalar characterization. The main and focused configurations, outcomes, and audit results are saved in [the numerical diagnostics](figures/paradigms.json). The checks verify the selected examples rather than treating every coarse-scan classification as established.

### Interactive reading edition

`README.md` is the source of truth for the paper's prose, equations, tables, and figure placement. The HTML edition is generated from it; do not edit the generated page separately.

Install the optional paper dependencies and start the local live preview:

```bash
uv sync --extra notebook --extra dev --extra paper
uv run token-demand-paper serve
```

Open [the local paper preview](http://127.0.0.1:8000). While this command is running, saving the README rebuilds the page and refreshes the browser. Press Ctrl-C in the terminal to stop it. Use `--port 8001` if port 8000 is already in use.

If a rebuild fails, the preview keeps the last successful page visible and reports the error in the terminal. Correct the source and save again to retry.

Section 5 keeps the same two-row, three-column layout in print and in the interactive edition: constraint regimes run down the rows, and demand, delegation, and completion run across the columns.

Each figure supports line toggles, double-click isolation, hover values, zooming, and an expanded panel view. Line visibility is linked across the panels of that figure. The main attention-limited three-panel figures start with shared logarithmic y-scales; choose **Independent scales** to fit each separately. Work-limited and focused comparisons use independent axes, including mixed linear and logarithmic scales where labeled. Each subfigure also has its own **Fit visible** button: it fits only that panel's y-axis, switches to independent scales, and leaves the other panels unchanged. **Fit all visible** fits all panels using the selected scale mode. Hiding lines alone keeps the scale fixed. Adoption-concentration high and low retain separate toggles even when they coincide with the reference; double-click either label to isolate its curve. Printing uses the original static figures.

To produce a static HTML edition without starting a server:

```bash
uv run token-demand-paper build
```

The output is `build/paper/index.html`. Open it directly in a browser, or copy the entire `build/paper/` folder to share the edition and its assets. Plotly and chart coordinates are included locally; equation typesetting loads a pinned MathJax version from its CDN and requires an internet connection. If JavaScript is unavailable, the original PNG figures remain visible.

Chart coordinates are exported from the notebook's actual Matplotlib figures, not from a separate browser implementation of the model. The tracked cache `figures/interactive.json` makes prose-only rebuilds fast. Changing the notebook's code or the model sources invalidates that cache and reruns the notebook, including its numerical audits; this can take several minutes. To force a numerical refresh:

```bash
uv run token-demand-paper refresh
```

Editing a written equation changes the manuscript, not the Python model. Model changes belong in `src/modeling_token_demand/` or the notebook. The local preview only serves generated files on this computer; it does not publish the paper. A shared static copy updates only when you rebuild and replace it.
