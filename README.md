# Modeling Token Demand

## Introduction

AI progress does not translate mechanically into token demand. A more capable or token-efficient model can complete the same work with fewer tokens, but it can also make new work worthwhile, support larger assignments, or induce users to buy more inference. Price declines create a similar ambiguity for revenue: demand rises, yet spending rises only if usage expands faster than price falls. The direction of the response therefore depends on what prevents more AI work from being done.

We study these forces in a stylized model in which users jointly choose the scope of a task and the inference effort spent on it. Capability expands the set of feasible tasks, token efficiency changes the effective inference provided by each token, and human verification becomes more costly as assignments grow. We compare two limiting regimes. In the first, the supply of potential work is fixed, while review capacity is sufficient. In the second, worthwhile work is abundant, but human attention is scarce. This distinction separates adoption—the share of existing work assigned to AI—from supervisory leverage—the amount of work supported by each review hour.

The model is intended to organize the relevant comparative statics rather than produce a point forecast. Its main implication is that the same technical improvement can increase or decrease token demand depending on how much adoption remains to be unlocked and how verification costs scale with task scope.

### TL;DR

- [When work is limited, capability can make token demand peak and then fall.](#21-token-demand-peaks-as-adoption-nearly-saturates) Better models raise reliability and adoption but require less inference for a given task; once adoption saturates, the token-saving effect dominates.
- [Token efficiency has the same competing margins.](#22-token-efficiency-can-raise-demand-before-it-saves-tokens) It raises surplus and adoption but lowers tokens per task, so demand rises only while the expansion in adopted work is sufficiently strong.
- [Lower token prices always raise demand, but not necessarily revenue.](#23-cheaper-tokens-raise-demand-but-spending-eventually-falls) A Jevons-like rebound requires demand to expand faster than price falls, and this is [harder when human attention is already fully used](#33-jevons-paradox-is-harder-to-reproduce-in-the-attention-limited-regime).
- [When human attention is scarce, capability creates value by raising supervisory leverage.](#31-capability-raises-demand-when-verification-cost-grows-slowly) Larger assignments let each review hour support more work, especially when verification time grows slowly with task scope.
- [Pricing power and the value of attention obey different ceilings.](#34-capability-raises-the-value-of-scarce-attention) The [work-limited reservation token price eventually plateaus](#24-capability-raises-the-work-limited-reservation-token-price), while the value of a review hour can keep growing roughly as $m^{1-\beta}$ when review time grows sublinearly.

## 1. Modeling assumptions

### 1.1 The user chooses scope and effort level

For each task, the user chooses two things:

- $s$: the scope of the task as measured in work units. A work unit can be thought of as the work a reference human completes in one hour. "Implement this function" would be a relatively small task, while "Refactor this entire system and ensure the test coverage is good" would be a large task.
- $x$: the model's normalized effort level per work unit; a task of scope $s$ consumes $sx$ token units. One can understand $x$ as the effort levels in major AI products: "low" "medium" "high" "pro" etc. But things like specialized harness, "/loop" "/goal" etc can also be modeled as $x$. To prevent some degenerate results later we normalize minimum viable effort to $x=1$. The choice of $1$ is unimportant and it can be any positive constant and our qualitative results will stay the same.

After the model attempts each task, the user needs to verify the correctness of its output. All else being equal, the bigger the task, the less reliable the model is. On the other hand, assigning bigger tasks reduces the number of times the user has to manually review the model's output. For example, one can use AI in "tab complete" mode, where each task is simply to complete the next line of code. In this case, the success rate is likely very high and each verification step is quick, but the workflow requires many human verification steps over the course of a day. The other extreme is to assign the AI an entire project. This reduces the number of human verification steps, but each step takes longer, and the AI is much less likely to complete the work correctly.

For more difficult tasks, higher $x$ can significantly increase the success probability. For easy tasks, however, a large $x$ unnecessarily increases token usage without meaningfully improving the quality of the output.

The user chooses the optimal $s$ and $x$ to maximize their utility, which will be made explicitly shortly.

### 1.2 Model capability expands the set of feasible tasks

Let $m$ denote model capability. The share of tasks within the model's capability frontier is given by

```math
q(s;m)=\exp\left[-\left(\frac{s}{\lambda m}\right)^\nu\right],
```

where the parameters $\lambda$ and $\nu$ together define how *difficult* the set of possible tasks is. Different industries have different $\lambda$ and $\nu$. Given a fixed model capability $m$ and task scope $s$, $q(s;m)$ is the fraction of work in the industry that is feasible for the model.

Note that feasibility does not imply a 100% success rate, which we model next.

### 1.3 More tokens spent = higher probability of success

Conditional on a task being feasible, we model the probability of successful execution as

```math
r(s,x;m,\eta)
=\exp\left[-\frac{s}{a m(\eta x)^\alpha}\right].
```

The parameter $\alpha\in(0,1)$ determines the rate of diminishing returns to more inference. $\eta$ measures token efficiency: a model that is twice as token-efficient can achieve the same reliability, or success rate, with half as many tokens. We model token efficiency separately from model capability because capability expands the set of feasible tasks, whereas token efficiency does not. It is common practice for labs to publish model evaluation results on a reward vs tokens plot. Pushing the curve horizontally towards fewer tokens means the token efficiency is improving. Pushing the curve up means the capability ceiling is improving. 

Combined with the task feasibility set, a model can successfully complete a task with scope $s$ with probability

```math
P(s,x;m,\eta)=q(s;m)r(s,x;m,\eta).
```

### 1.4 Human reviews take longer as tasks get larger

We model the review time for one task as

```math
h(s)=h_0+h_1\left[(1+s)^\beta-1\right].
```

$h_0$ is the fixed time associated with one review. The second term grows with assignment size. Depending on the type of workflow, $\beta$ can be small, which means that human verification time is nearly fixed with respect to task horizon $s$. When $\beta$ is close to one, review grows almost in proportion to the work delegated. As $\beta$ approaches 0 the review cost approaches $h_0$, and as $\beta$ approaches 1 the review cost approaches $h_0 + h_1s$


Examples of workflows with small $\beta$ include software engineering work with automated tests and reliable benchmarks. Examples with large $\beta$ include insurance claims and legal cases, where each case still requires human supervision and the amount of human verification work grows nearly linearly with the task scope. 

Let $w$ be the dollar value of one hour of human attention. Review cost per unit of work is

```math
w\frac{h(s)}s.
```

### 1.5 The user maximizes expected surplus

Let $b$ be the value of one successfully completed work unit. Let $c$ be the dollar price of one normalized token unit. Expected surplus per assigned work unit is

```math
u(s,x)=bP(s,x;m,\eta)-cx-w\frac{h(s)}s.
```

Given a fixed set of model and industry parameters, let $s^*$ and $x^*$ be the scope and model-effort settings that maximize the user's surplus per work unit.

### 1.6 Adoption depends on the value of using AI

A positive optimized surplus, $u^*:=u(s^*,x^*)>0$, does not by itself guarantee that the user will adopt AI. This could be because completing the work manually generates even more surplus, or it could be due to inherent frictions in adopting a new technology. Conversely, some users may adopt even when measured surplus is negative because they value being AI-forward or has an organizational mandate to use AI. We capture these considerations with an adoption hurdle $\phi$, which can be positive or negative and is distributed logistically across tasks. A user adopts when surplus exceeds this hurdle. The fraction of work that adopts AI given some surplus is then given by

```math
A(u)=\Pr(\phi\leq u)=\frac{1}{1+\exp[-(u-\mu)/\sigma]}.
```

The location $\mu$ sets the typical hurdle. The spread $\sigma$ determines whether adoption is gradual or concentrated around a threshold.

### 1.7 Token revenue is proportional to number of tokens used

Let $Q$ be the number of work units assigned to AI during the period. Then

```math
D=Qx
```

is **token demand**, measured in normalized token units. **Token spending** is simply demand times unit price

```math
R=cD,
```

measured in dollars. 

Here is a summary of the modeling parameters that are using. In the next two sections we will study the different adoption and demand curves as a function of model capability, token efficiency, and token price in two distinct regimes.

| Parameter | Name | Category | How it is used in the model |
|---|---|---|---|
| $s$ | Delegated scope | User action | Work units assigned before the next review. It enters feasibility $q(s;m)$, execution reliability $r(s,x;m,\eta)$, and review time $h(s)$. |
| $x$ | Model effort level | User action | Normalized token units used per work unit, with minimum viable effort $x=1$. Effective inference is $\eta x$, token cost per work unit is $cx$, and one assignment uses $sx$ token units. |
| $m$ | Model capability | Model parameter | Expands both the capability horizon $\lambda m$ in $q(s;m)$ and the execution horizon in $r(s,x;m,\eta)$. |
| $\eta$ | Token efficiency | Model parameter | Converts token units into effective inference, $\eta x$, in the execution-reliability function. |
| $c$ | Token price | Model parameter | Dollar price of one normalized token unit. It determines token cost $cx$ and total token spending $R=cD$. |
| $\lambda$ | Capability horizon | Industry parameter | Sets the baseline assignment scope the model can feasibly handle in $q(s;m)=\exp[-(s/(\lambda m))^\nu]$. |
| $\nu$ | Capability shape | Industry parameter | Controls how sharply the feasible share falls as delegated scope grows in $q(s;m)$. |
| $a$ | Execution ease | Industry parameter | Scales the execution horizon $am(\eta x)^\alpha$ in $r(s,x;m,\eta)$. |
| $\alpha$ | Inference returns | Industry parameter | Controls the diminishing return from effective inference $\eta x$ in the execution horizon. |
| $h_0$ | Fixed review time | Industry parameter | Captures the fixed time required to open, understand, and close a review in $h(s)=h_0+h_1[(1+s)^\beta-1]$. |
| $h_1$ | Variable review scale | Industry parameter | Scales the part of human verification time that grows with delegated scope in $h(s)$. |
| $\beta$ | Review elasticity | Industry parameter | Controls how quickly human verification time grows with delegated scope through the term $s^\beta$. |
| $b$ | Value of successful work | Industry parameter | Dollar value of one successfully completed work unit. Expected benefit per assigned work unit is $bP(s,x;m,\eta)$. |
| $w$ | Value of human attention | Industry parameter | Dollar value of one hour of human review. Review cost per assigned work unit is $wh(s)/s$. |
| $\phi$ | Adoption hurdle | Industry parameter | Task-specific hurdle that optimized surplus must exceed for adoption. It can be positive or negative. |
| $\mu$ | Hurdle location | Industry parameter | Sets the location of the logistic distribution of adoption hurdles in $A(u)$. |
| $\sigma$ | Hurdle spread | Industry parameter | Controls whether adoption is gradual or concentrated around the hurdle location in $A(u)$. |
| $W$ | Potential work | Industry parameter | Total work available in the work-limited regime, where assigned work is $Q=WA$. |
| $H$ | Human attention | Industry parameter | Review hours available in the attention-limited regime, where assigned work is $Q=Hs/h(s)$. |




## 2. Work is limited

In this regime, there are only $W$ potential work units. Better or cheaper AI can increase token demand by increasing surplus for the user and thus bringing more of the users over the adoption threshold.

We start with a reference industry and vary one parameter at a time to produce a set of stylized, not quantitatively calibrated, industries. We then vary model capabilities, token efficiency, and token cost and study how adoption/token revenue changes. Every line uses $W=1{,}000{,}000$.

| Plot line | $\lambda$ | $\nu$ | $a$ | $\alpha$ | $h_0$ | $h_1$ | $\beta$ | $b$ | $w$ | $\mu$ | $\sigma$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference industry | 12 | 1.25 | 4 | 0.5 | 0.03 | 0.05 | 0.5 | 100 | 100 | 81 | 4 |
| Low adoption hurdle | 12 | 1.25 | 4 | 0.5 | 0.03 | 0.05 | 0.5 | 100 | 100 | **40** | 4 |
| High adoption hurdle | 12 | 1.25 | 4 | 0.5 | 0.03 | 0.05 | 0.5 | 100 | 100 | **95** | 4 |
| Hard execution | 12 | 1.25 | **1** | 0.5 | 0.03 | 0.05 | 0.5 | 100 | 100 | 81 | 4 |
| High capability requirement | **3** | 1.25 | 4 | 0.5 | 0.03 | 0.05 | 0.5 | 100 | 100 | 81 | 4 |


### 2.1 Token demand peaks as adoption nearly saturates

![Figure 1. Adoption and token demand as a function of model capability.](figures/work-capability-demand-spending.png)

*Each demand curve is normalized to 1 at $m=1$ to show its trend rather than its absolute level. A star on each adoption curve marks the value of $m$ where the corresponding token-demand curve reaches its maximum over the plotted range.*

Higher capability raises reliability and adoption but reduces tokens per task. In industries with already high adoption, token savings effect dominate and overall token spending decreases as models improve. The stars in Figure 1a) marks the model capability level where the token demand peaks.



### 2.2 Token efficiency can raise demand before it saves tokens

![Figure 2. Work-limited adoption and indexed token demand as token efficiency changes.](figures/work-efficiency-demand-spending.png)

*A value of $\eta=2$ means that each token provides twice as much effective inference as at $\eta=1$. Panel (b) normalizes each demand curve to 1 at $\eta=1$ to show its trend rather than its absolute level; the black horizontal line marks unchanged token revenue because token price is fixed. Panels (c) and (d) report the model effort level $x$ and effective inference $\eta x$. The three hurdle-only cases overlap in these panels because adoption hurdles do not change the optimal policy conditional on use.*

As models get more token efficient, users can spend fewer tokens for the same level of reliability, which drives down token demand. On the other hand, higher token efficiency leaves users with higher surplus which increases adoption. Which effect dominates depends on the current state of adoption and the industry setting: in industries with low adoption rate / tasks that are difficult to execute, the adoption effect dominates and the overall token demand grows (green and orange line). But in other industry settings, where token efficiency cannot drive enough additional adoption, the total token demand eventually falls as token efficiency improves.

### 2.3 Cheaper tokens raise demand, but spending eventually falls

![Figure 3. Work-limited adoption, indexed token demand and spending, and optimal model effort level, as token price changes.](figures/work-price-demand-spending.png)

*Price falls from left to right. Panel (a) reports adoption in levels. Panels (b) and (d) normalize every curve to 1 at $c=1$ to show its trend rather than its absolute level. Panel (c) reports the optimal model effort level $x$ in levels. The black line in panel (b) is the demand increase required to keep spending unchanged.*

Unlike in the case of increasing token efficiency, at a per task level, users actually choose to use more tokens (i.e. higher effort, see Figure 3c vs Figure 2c). As a result, the token demand increases universally, even in industries with already high adoption. However total token spending does not necessarily increase as token price drops. It's only in industries with still low-ish adoption rate that we observe Jevons paradox, where decreasing price increases overall token spending. In industries with already significant adoptions (reference industry and low adoption hurdle), the increase in token demand does not lead to overall increase in token revenue.

### 2.4 Capability raises the work-limited reservation token price
So far we have treated the token price as an exogenous variable that the model providers choose. We can also try to analyze the model providers' pricing power by solving for the "reservation price" of a smarter model. Intuitively, the reservation price represents "how much more the user is willing to pay for a smarter model relative to a baseline model". More concretely, let the optimized surplus per assigned work unit be $u^*(m,c)$. We define the work-limited reservation token price by

```math
u^*\!\left(m,c_{\rm res}^{W}(m)\right)=u^*(1,c_0).
```

It is the highest token price that preserves baseline optimized surplus and therefore adoption $A(u^*)$. Every candidate price reoptimizes $s$ and $x$. The superscript $W$ distinguishes it from the attention-limited $c_{\rm res}^{H}$ which we will study in the next section.

![Figure 4. Work-limited reservation token prices rise with model capability.](figures/work-capability-reservation-price.png)

*The vertical axis reports $c_{\rm res}^{W}(m)/c_0$ on a logarithmic scale. The adoption-hurdle-only cases coincide with the reference because the hurdle does not enter $u^*$; we retain them to make that invariance visible, with staggered markers in the static figure and separate toggles in the interactive figure. The user reoptimizes $s$ and $x$ at every point.*

We start with a model at capability level $m=1$ and price $c_0$. We can solve for $c_{\rm res}^{W}$ as a function of evolving $m$ and other industry parameters.
At $m=0.8$, preserving baseline surplus requires a price of $0.29c_0$ to $0.61c_0$ (i.e. user would demand a discount relative to the baseline model to use a dumber model). At $m=30$, it reaches $14.6c_0$ in the reference, $18.9c_0$ under high capability requirements, and $22.2c_0$ under hard execution.
Note that the reservation price eventually plateaus instead of increasing forever as models get smarter. The intuition is basically that once the model is "smart enough", it can do approximately ALL the work in that industry, and because token usage $x$ is bounded from below at $1$, it eventually will not make sense for user to pay for even smarter models.


In summary, we see that in this work limited regime, the returns on smarter model / more token efficiency eventually becomes negative (Figure 1, 2), and that the strategy of increasing total revenue by lowering token cost (Jevons paradox) only works in some settings where existing adoption level is still low (Figure 3). Even if the model company is able to extract ALL the surplus from the user, the return on more intelligence eventually plateaus (Figure 4).

## 3. Human attention is limited

In this regime, there is abundant worthwhile work but only $H$ human review hours. Let $n$ tasks use the same policy $(s,x)$. Because each task contains $s$ work units and requires $h(s)$ review hours, the attention constraint is

```math
n h(s)\leq H.
```

We define **supervisory leverage** as

```math
\ell(s)=\frac{s}{h(s)},
```

the work assigned to AI per human review hour. Once all $H$ hours are used, $n=H/h(s)$ and total assigned work is $Q=ns=H\ell(s)$. 

With this attention constraint, the user chooses $(n,s,x)$ to maximize

```math
ns\left[bP(s,x;m,\eta)-cx\right]-wnh(s)
\quad\text{subject to}\quad nh(s)\leq H.
```

Conditional on the objective being greater than 0, the constraint is binding, so we can substitute $n=H/h(s)$, which makes total surplus

```math
\Pi(s,x)=H\underbrace{\left\{\frac{s}{h(s)}\left[bP(s,x;m,\eta)-cx\right]-w\right\}}_{J(s,x)}.
```

$J(s,x)$ is expected surplus per review hour. Since $H>0$ is fixed, maximizing total surplus subject to the attention constraint is equivalent to maximizing $J(s,x)$.

These figures use $H=100{,}000$ review hours. Again, every alternative industry changes exactly one parameter from the reference and the difference is marked in bold.

| Plot line | $\lambda$ | $\nu$ | $a$ | $\alpha$ | $h_0$ | $h_1$ | $\beta$ | $b$ | $w$ | $\mu$ | $\sigma$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference industry | 12 | 1.25 | 4 | 0.5 | 0.03 | 0.05 | 0.5 | 100 | 100 | 81 | 4 |
| Hard execution | 12 | 1.25 | **1** | 0.5 | 0.03 | 0.05 | 0.5 | 100 | 100 | 81 | 4 |
| Low inference returns | 12 | 1.25 | 4 | **0.2** | 0.03 | 0.05 | 0.5 | 100 | 100 | 81 | 4 |
| Slow-growing review | 12 | 1.25 | 4 | 0.5 | 0.03 | 0.05 | **0.15** | 100 | 100 | 81 | 4 |
| Nearly proportional review | 12 | 1.25 | 4 | 0.5 | 0.03 | 0.05 | **0.95** | 100 | 100 | 81 | 4 |

The slow-growing-review case represents workflows where automated tests, benchmarks, or standardized checks keep verification time from growing quickly with task horizon $s$. The low-inference-returns case represents workflows where additional inference has sharply diminishing value.

### 3.1 Capability raises demand when verification cost grows slowly

![Figure 5. Supervisory leverage and indexed attention-limited token demand as model capability changes.](figures/attention-capability-demand-spending.png)

*Both outcomes are normalized to 1 at $m=1$ to show their trends rather than their absolute levels. Price is fixed, so spending has the same shape as demand.*

A better model uses fewer tokens for a given task, but can also handle larger tasks. In the reference, hard-execution, low-inference-returns, and slow-growing-review industries, work supervised per review hour expands enough that token demand rises. The increase is especially strong when $\beta$ is small, because review time grows slowly relative to task scope. Intuitively, this means the user can assign models longer and more valuable tasks, while keeping their verification time constant.  When $\beta=0.95$, human verification time grows nearly in proportion to task horizon. In this case the supervisory leverage expands too slowly to keep offsetting token savings from smarter models, so demand peaks and falls.

### 3.2 Token efficiency can expand work per review hour

![Figure 6. Supervisory leverage and indexed attention-limited token demand as token efficiency changes.](figures/attention-efficiency-demand-spending.png)

*Both outcomes are normalized to 1 at $\eta=1$ to show their trends rather than their absolute levels. Capability, review hours, and token price are fixed.*

Efficiency does not create more review hours, but it can change the size of the task that user decides to send to the model. In the reference, slow-growing-review, and nearly proportional-review industries, that expansion is too small to offset fewer tokens per work unit, so demand falls over most of the range. When execution is hard, greater efficiency makes larger assignments worthwhile and demand rises. When returns to inference are low, those forces nearly cancel and demand stays almost flat. These are intensive-margin responses: all review hours were already in use, but each hour can support a different amount of work.

Compared to the work limited regime, token efficiency seems to be a much worse lever for generating additional token revenue in the regime of limited human attention.

### 3.3 Jevons paradox is harder to reproduce in the attention limited regime

![Figure 7. Attention-limited token demand and spending as token price changes.](figures/attention-price-demand-spending.png)

*Price falls from left to right. Every curve is normalized to 1 at $c=1$ to show its trend rather than its absolute level. The black line is the demand increase required to keep spending unchanged.*

In this regime, all $H$ review hours are used, and token demand is $D=H[s/h(s)]x$. When tokens get cheaper, the user chooses a higher effort level $x$. The extra inference can also make larger assignments $s$ worthwhile, which may increase the work assigned per review hour, $s/h(s)$. Demand therefore grows through both more tokens per unit of work and more work per scarce review hour. Token revenue rises only if this combined increase in demand is larger than the inverse price decline shown in Figure 7a.

These forces are strongest when execution is hard: extra inference meaningfully improves success and supports larger assignments, so demand rises faster than price falls. In most other cases, the increase in demand is unable to compensate the lowered prices. It is interesting to note that compared to the work limited case in Figure 3, it is much more difficult to produce a Jevons paradox like effect. This is somewhat intuitive: it's easier to convert users from not using AI to using AI by lowering prices, than to increase the token usage of existing use cases.

### 3.4 Capability raises the value of scarce attention

Figures 5–7 study how technology and prices affect token demand and revenue. Similar to Figure 4, here we also ask the question of how much user surplus increases as a function of model capability (which in turn implies the total *potential* revenue the model producer can generate). Since attention is the bottleneck, we additionally ask the question of, how much the userwould be willing to pay to get one additional hour of human attention. This in theory informs the question of how the labor wage will change in the age of AI.

At token price $c$, we define optimized user surplus per review hour (i.e. the value of reviewer attention) as

```math
J^*(m,c)=\max_{s,x}\left\{\frac{s}{h(s)}\left[bP(s,x;m,\eta)-cx\right]-w\right\},
\qquad
\rho^*(m,c)=J^*(m,c)+w.
```

$J^*(m,c)$ is the user's net surplus after paying for tokens and accounting for the opportunity cost $w$ of the review hour. Adding $w$ back gives $\rho^*(m,c)$, the value created by one review hour after token spending. Equivalently, it is the maximum hourly price the user could pay for reviewer attention while still choosing to operate. Panel (a) evaluates this quantity at the baseline token price $c_0=1$.

The model provider may instead capture some of this value through the token price. Starting from a baseline model with capability $m=1$ and price $c_0$, define the attention-limited reservation token price $c_{\mathrm{res}}^{H}(m)$ by

```math
J^*\!\left(m,c_{\mathrm{res}}^{H}(m)\right)=J^*(1,c_0).
```

This is the highest token price that leaves the user with the same optimized surplus as the baseline model. At every candidate price, the user reoptimizes both task scope $s$ and inference effort $x$. The reservation price is therefore an upper bound on the user's willingness to pay for capability, not a prediction of the market price the provider will actually charge.

![Figure 8. Capability raises reservation prices for scarce attention and model tokens.](figures/attention-capability-value.png)

*Panel (a) normalizes each $\rho^*(m,c_0)$ curve by that same line's value at $m=1$, so every line equals 1 at the baseline capability. Panel (b) reports the fully reoptimized $c_{\rm res}^{H}(m)/c_0$, which also equals 1 at $m=1$. Both vertical axes are logarithmic, $\eta=1$, and $m$ ranges from $0.8$ to $30$.*

The increase in reviewer attention value is pretty striking. At $m=30$, the value of a review hour is 5.3 times its baseline in the reference industry and 15.8 times its baseline with slow-growing review, compared with only 1.8 times its baseline when review time grows nearly in proportion to task scope. The difference comes from supervisory leverage: larger tasks create much more work per review hour when $\beta$ is small, but much less when $\beta$ is close to one. As a back-of-the-envelope calculation, success depends on the ratio $s/m$, so keeping reliability roughly constant allows task scope to grow in proportion to capability, $s\propto m$. At large task scope, review time is approximately $h(s)\approx h_1s^\beta$. The total value created by one hour of human review after token cost is therefore roughly:

```math
\rho^*(m,c_0)\propto \frac{s}{h(s)}\propto m^{1-\beta}.
```

This means approximately $m^{0.5}$ growth in the reference industry, $m^{0.85}$ with slow-growing review, and only $m^{0.05}$ with nearly proportional review. At exactly $\beta=1$, $s/h(s)$ instead approaches $1/h_1$, and the value of attention approaches the finite ceiling $(b-c_0)/h_1$.

The reservation token price follows the same intuition. At $m=0.8$, a less capable model requires a discount, with $c_{\rm res}^{H}$ ranging from $0.20c_0$ to $0.57c_0$. By $m=30$, a more capable model supports prices between $39.4c_0$ and $70.7c_0$ while preserving baseline user surplus. Once the optimal effort level reaches its lower bound $x=1$, the reservation price bends toward the value $b$ of one successful work unit: each work unit creates at most $b$ in expected value and consumes at least one token. This limits the price per token, but not the value of scarce attention. When review time grows less than proportionally with task scope, a more capable model can continue increasing $s/h(s)$, so the same human hour supports more work even after token use per work unit reaches its minimum. This is the key difference from the work-limited regime, where the total amount of available work is fixed.

## 4. Conclusion

We studied how model capability, token efficiency, and token prices affect token demand and revenue under two limiting resource constraints: finite work and finite human attention. When work is limited, capability and efficiency raise adoption but reduce token use per task. As adoption saturates, token demand can peak and eventually fall. Similarly, lower token prices increase demand but raise revenue only when they unlock enough additional adoption. The work-limited reservation token price eventually plateaus because total work is fixed and inference effort cannot fall below its minimum.

When human attention is limited, the key limiting factor is the amount of work supported by each review hour. More capable models can make larger assignments worthwhile, so token demand and the value of reviewer attention can continue to grow even as token use per unit of work falls. This effect is strongest when review time grows slowly with task scope. If review time grows approximately as $s^\beta$, the value of attention grows roughly as $m^{1-\beta}$. The reservation token price also has a per-token ceiling, even when the value of scarce attention continues to rise.

Taken together, our results show that technical progress alone does not determine token demand or revenue. The outcome depends on which resource is scarce, how much adoption remains to be unlocked, and crucially how verification costs scale with task scope. An interesting avenue for future work is to study how the pricing dynamics change when there are multiple competing model providers.
