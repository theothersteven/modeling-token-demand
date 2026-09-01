# When does better AI increase token demand?

*A simple model of finite work and scarce human attention*

**Working draft · September 2026**

## 1. Modeling assumptions

### 1.1 The user chooses scope and tokens

A **work unit** is the amount of useful work a reference human completes in one hour. It measures task size. It does not measure how long the model runs.

The user chooses two things:

- $s$: work units delegated before the next human review;
- $x$: tokens used per work unit.

One assignment therefore contains $s$ work units and uses $sx$ tokens. Larger assignments require fewer checkpoints per work unit, but they are harder for the model to complete. More tokens can improve reliability, but they cost money.

The model attempts each assignment once. Success produces $s$ completed work units. Failure produces no completed work. Either outcome uses all $sx$ tokens and requires human review.

Review perfectly identifies whether the assignment succeeded. Review time depends on assignment size, not on the outcome or the number of tokens used. The model omits partial completion, undetected errors, and learning across attempts.

### 1.2 Capability determines which work is feasible

Let $m$ denote model capability. The share of assignments within the model's capability frontier is

```math
q(s;m)=\exp\left[-\left(\frac{s}{\lambda m}\right)^\nu\right].
```

The parameter $\lambda$ sets the baseline capability horizon. The parameter $\nu$ controls how sharply feasibility falls with assignment size. Higher $m$ lets the model handle larger assignments.

This is the **technically solvable share** of the market at scope $s$. It is not the share that actually buys tokens. Adoption also depends on value and cost.

### 1.3 Tokens determine reliability within the frontier

Let $\eta$ denote token efficiency. If $\eta$ doubles, each token supplies twice as much effective inference. Conditional on an assignment being feasible, the probability of successful execution is

```math
r(s,x;m,\eta)
=\exp\left[-\frac{s}{a m(\eta x/\bar x)^\alpha}\right].
```

The parameter $a$ measures how easy the work is to execute. The parameter $\alpha\in(0,1)$ gives diminishing returns to more inference. The constant $\bar x$ only fixes the unit in which tokens are measured.

Total reliability is

```math
P(s,x;m,\eta)=q(s;m)r(s,x;m,\eta).
```

Capability $m$ improves both feasibility and execution. Token efficiency $\eta$ improves execution for a given number of tokens.

### 1.4 Human review gets harder as assignments grow

Review time for one assignment is

```math
h(s)=h_0+h_1s^\beta.
```

The fixed term $h_0$ is the cost of opening, understanding, and closing a review. The second term grows with assignment size. When $\beta$ is small, much of the review can be reused across a larger assignment. When $\beta$ is close to one, review grows almost in proportion to the work delegated.

Let $w$ be the dollar value of one hour of human attention. Review cost per assigned work unit is

```math
w\frac{h(s)}s.
```

### 1.5 The user maximizes expected surplus

Let $b$ be the value of one successfully completed work unit. Let $c$ be the dollar price of one token. Expected surplus per assigned work unit is

```math
u(s,x)=bP(s,x;m,\eta)-cx-w\frac{h(s)}s.
```

Tokens and review time are paid whether the assignment succeeds or fails. Reliability therefore matters without a separate retry decision or failure penalty.

### 1.6 Adoption depends on the value of using AI

In the work-limited regime, opportunities have different switching hurdles $\phi\geq0$. These can represent integration costs, the value of the existing process, or reluctance to change. The user chooses $(s,x)$ to maximize surplus per work unit. A work opportunity adopts AI when that optimized surplus exceeds its hurdle.

For the numerical examples, hurdles follow a logistic distribution truncated at zero. With $\Lambda(z)=1/(1+e^{-z})$, the adopted share is

```math
A(u)=
\begin{cases}
0, & u\leq0,\\[4pt]
\dfrac{\Lambda((u-\mu)/\sigma)-\Lambda(-\mu/\sigma)}
{1-\Lambda(-\mu/\sigma)}, & u>0.
\end{cases}
```

The location $\mu$ sets the typical hurdle. The spread $\sigma$ determines whether adoption is gradual or concentrated around a threshold.

### 1.7 Token demand is a number of tokens; token spending is in dollars

Let $Q$ be the number of work units assigned to AI during the period. Then

```math
D=Qx
```

is **token demand**, measured as a number of tokens. Token spending is

```math
R=cD,
```

measured in dollars. Spending is the token supplier's gross revenue. It is not profit or total economic value.

The model studies two limiting cases:

| | Work is limited | Human attention is limited |
|---|---:|---:|
| Scarce resource | $W$ potential work units | $H$ review hours |
| Policy objective | Maximize $u(s,x)$ | Maximize $J(s,x)=\dfrac{s}{h(s)}[bP(s,x)-cx]-w$ |
| Work assigned to AI | $Q=WA$ | $Q=H\dfrac{s}{h(s)}$ |
| Token demand | $D=WAx$ | $D=H\dfrac{s}{h(s)}x$ |
| Token spending | $R=cWAx$ | $R=cH\dfrac{s}{h(s)}x$ |

In the first regime, there is enough review capacity to serve all work that adopts. In the second, worthwhile work is abundant and every available review hour is used. These are polar cases. A later draft can study the transition between them.

### 1.8 The figures compare simple industry types

The reference technology uses $\lambda=12$, $\nu=1.25$, $a=4$, $\alpha=0.5$, $h_0=0.03$, $h_1=0.05$, $\beta=0.5$, $\mu=76$, and $\sigma=4$. All examples use $b=w=100$ dollars, $\bar x=100{,}000$ tokens, and a baseline token price of 10 dollars per million tokens. Work-limited industries have $W=1{,}000{,}000$ potential work units. Attention-limited industries have $H=100{,}000$ review hours. These values illustrate the model. They are not estimates.

The work-limited figures compare:

| Industry type | What changes from the reference case? |
|---|---|
| Gradual adoption | Reference technology and $\sigma=4$ |
| Clustered adoption | Hurdles are concentrated: $\sigma=1$ |
| Early saturation | Clustered hurdles and easier work: $\lambda=36$, $a=8$ |

The attention-limited figures compare:

| Industry type | Review time $h(s)=h_0+h_1s^\beta$ |
|---|---|
| Reusable review | $h_0=0.015$, $h_1=0.025$, $\beta=0.25$ |
| Balanced review | $h_0=0.03$, $h_1=0.05$, $\beta=0.50$ |
| Proportional review | $h_0=0.06$, $h_1=0.10$, $\beta=0.95$ |

Unless the horizontal axis says otherwise, $m=\eta=1$ and the token price is 10 dollars per million tokens. Both vertical axes use logarithmic scales so industries with different market sizes remain visible. Each point reoptimizes both $s$ and $x$.

## 2. Work is limited

There are only $W$ potential work units. Better or cheaper AI can increase token demand by bringing more of this work into the market. That expansion stops when adoption approaches 100 percent.

### Token use peaks once adoption runs out of room

![Figure 1. Work-limited token demand and spending as model capability changes.](figures/work-capability-demand-spending.png)

*Demand is measured in trillions of tokens. Spending is measured in millions of dollars. The token price is fixed at 10 dollars per million tokens, so the two panels have the same shape.*

Higher capability raises reliability and adoption. It also lets the user spend fewer tokens on each work unit. At first, adoption grows faster than tokens per work unit fall, so total token demand rises. Later, adoption has little room left to grow and token savings dominate.

The peak arrives earliest in the early-saturation industry. This industry already has easy work and little unused market at the baseline. Further capability increases completed work while reducing token demand. The unintuitive result is that a more useful model can generate less token spending.

### Token efficiency can raise demand before it saves tokens

![Figure 2. Work-limited token demand and spending as token efficiency changes.](figures/work-efficiency-demand-spending.png)

*A value of $\eta=2$ means that each token provides twice as much effective inference as at $\eta=1$. Price is fixed, so demand and spending again have the same shape.*

Higher efficiency directly reduces the tokens needed to obtain a given reliability. It also lowers the cost of using AI and brings more work into the market. Demand rises when adoption expands faster than tokens per work unit fall.

The clustered-adoption industry has the largest rebound because many work opportunities cross the adoption threshold together. The early-saturation industry has almost no adoption margin left, so efficiency mainly saves tokens.

### Cheaper tokens raise demand, but spending eventually falls

![Figure 3. Work-limited token demand and spending as token price changes.](figures/work-price-demand-spending.png)

*Price falls from left to right. A point at 10 means 10 dollars per million tokens. Demand is a token quantity; spending equals price times that quantity.*

A lower token price always raises the optimized token quantity in these examples. Users buy more tokens per work unit, and lower operating cost can increase adoption.

Spending need not keep rising. Once adoption is close to its ceiling, an additional price cut produces too little new demand to offset the lower price. Spending therefore peaks and then falls. The clustered-adoption industry has the sharpest increase near its adoption threshold.

## 3. Human attention is limited

There is abundant worthwhile work but only $H$ review hours. The user chooses the policy that creates the most surplus per review hour. Assigned work is $H s/h(s)$, so the key question is whether a better model lets each reviewer supervise more work.

### Capability raises demand when review can be reused

![Figure 4. Attention-limited token demand and spending as model capability changes.](figures/attention-capability-demand-spending.png)

*Demand is measured in trillions of tokens. Spending is measured in millions of dollars. Price is fixed, so the two panels have the same shape.*

A better model can handle larger assignments. When review grows slowly with assignment size, a reviewer can supervise much more work. This expansion is strong in the reusable-review industry and token demand rises sharply.

When review is nearly proportional to assignment size, larger assignments release little attention. Token savings can then dominate. In the proportional-review industry, capability improves output even as token demand peaks and then falls.

### Token efficiency mostly reduces demand when attention is fixed

![Figure 5. Attention-limited token demand and spending as token efficiency changes.](figures/attention-efficiency-demand-spending.png)

*Capability, review hours, and token price are fixed. Every point reoptimizes assignment size and tokens per work unit.*

Efficiency does not create more review hours. It can change assignment size, but in these industries that response is too small to offset the reduction in tokens per work unit. Token demand therefore falls across most of the displayed range.

This differs from the work-limited case. There, lower cost can unlock adoption. Here, all review hours were already in use. Efficiency must increase work supervised per review hour to create a comparable rebound.

### Cheaper tokens raise demand, but spending eventually falls

![Figure 6. Attention-limited token demand and spending as token price changes.](figures/attention-price-demand-spending.png)

*Price falls from left to right. The vertical scales show absolute token quantities and dollar spending.*

Lower prices make extra inference worthwhile, so token demand rises. In all three plotted industries, demand eventually grows more slowly than price falls, so token spending declines at sufficiently low prices.

Reusable review supports much more token demand because each review hour covers more work. Proportional review keeps both demand and spending much lower. The review technology changes the level of the market and whether technical progress expands it.

---

The numerical model and figure code are in `src/modeling_token_demand`. Run `.venv/bin/python -m modeling_token_demand.paper refresh` to rebuild the audited figures and HTML reading edition. The figures are comparative statics for stylized industries, not forecasts.
