# CS229 Lecture 3: Locally Weighted & Logistic Regression

**Stanford CS229: Machine Learning - Autumn 2018**  
**Instructor:** Andrew Ng

---

## Table of Contents
1. [Overview](#overview)
2. [Notation Recap](#notation-recap)
3. [Feature Engineering for Linear Regression](#feature-engineering)
4. [Locally Weighted Linear Regression (LWR)](#locally-weighted-regression)
5. [Probabilistic Interpretation of Linear Regression](#probabilistic-interpretation)
6. [Logistic Regression](#logistic-regression)
7. [Newton's Method](#newtons-method)
8. [Key Takeaways](#key-takeaways)

---

## Overview

This lecture builds upon linear regression concepts and introduces:
- **Locally Weighted Regression**: A non-parametric method for fitting non-linear data
- **Probabilistic Interpretation**: Why we use squared error (justification via Maximum Likelihood Estimation)
- **Logistic Regression**: First classification algorithm for binary outcomes
- **Newton's Method**: Fast optimization algorithm for logistic regression

**Dependencies:**
- Locally Weighted Regression → depends on Linear Regression
- Logistic Regression → depends on Probabilistic Interpretation
- Newton's Method → optimization for Logistic Regression

---

## Notation Recap

From Lecture 2, we use the following notation:

| Symbol | Meaning |
|--------|---------|
| $(x^{(i)}, y^{(i)})$ | Training example $i$ |
| $m$ | Number of training examples |
| $n$ | Number of features |
| $x^{(i)}$ | Input features ($(n+1)$-dimensional with $x_0 = 1$) |
| $y^{(i)}$ | Output/target variable |
| $\theta$ | Parameter vector (weights) |
| $h_\theta(x)$ | Hypothesis function |

**Hypothesis Function:**
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n = \theta^T x$$

**Cost Function (Linear Regression):**
$$J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

---

## Feature Engineering

### Problem: Linear Models Don't Fit Curved Data

When data doesn't follow a straight line, we need to choose appropriate features.

**Example:** Housing prices vs. size might follow:
- Straight line: $h(x) = \theta_0 + \theta_1 x$
- Quadratic: $h(x) = \theta_0 + \theta_1 x + \theta_2 x^2$
- Square root: $h(x) = \theta_0 + \theta_1 x + \theta_2 \sqrt{x}$

### Feature Engineering Approach

To fit non-linear functions, define new features:
- Let $x_1 = x$ (original feature)
- Let $x_2 = x^2$ (squared feature) OR $x_2 = \sqrt{x}$ (square root feature)

Then apply linear regression to the new feature vector $[x_0, x_1, x_2]$.

**Challenge:** How do you choose which features to use?
- Feature selection algorithms (covered later in course)
- OR use Locally Weighted Regression (next section)

---

## Locally Weighted Regression

### Parametric vs. Non-Parametric Learning

**Parametric Learning Algorithm:**
- Fixed set of parameters $\theta$ that you fit to data
- Example: Linear regression with $\theta_0, \theta_1, \theta_2$
- Once trained, you can discard training data
- Memory requirement: constant (independent of training set size)

**Non-Parametric Learning Algorithm:**
- Amount of data/parameters grows with training set size
- Must keep entire training set to make predictions
- Memory requirement: $O(m)$ where $m$ is training set size
- Can fit complex patterns without manual feature engineering

### Algorithm

For linear regression, to predict at query point $x$:
1. Fit $\theta$ to minimize $J(\theta) = \frac{1}{2}\sum_{i=1}^{m}(y^{(i)} - \theta^T x^{(i)})^2$
2. Return $\theta^T x$

For **locally weighted regression**, to predict at query point $x$:
1. Fit $\theta$ to minimize: $$J(\theta) = \frac{1}{2}\sum_{i=1}^{m} w^{(i)}(y^{(i)} - \theta^T x^{(i)})^2$$
2. Return $\theta^T x$

### Weight Function

$$w^{(i)} = \exp\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)$$

**Properties:**
- If $|x^{(i)} - x|$ is small → $w^{(i)} \approx 1$ (weight examples close to query point highly)
- If $|x^{(i)} - x|$ is large → $w^{(i)} \approx 0$ (ignore examples far from query point)
- Shape: Gaussian bell curve (but NOT a probability distribution - doesn't integrate to 1)

### Bandwidth Parameter $\tau$

Controls the width of the weight function:
- **Large $\tau$**: Wider window, smoother fit (may underfit)
- **Small $\tau$**: Narrower window, more jagged fit (may overfit)
- This is a **hyperparameter** of the algorithm

**Vector-valued generalization:**
$$w^{(i)} = \exp\left(-\frac{(x^{(i)} - x)^T(x^{(i)} - x)}{2\tau^2}\right)$$

### When to Use LWR

✅ **Good for:**
- Low-dimensional problems ($n$ is small: 2, 3, or a few features)
- Plenty of training data
- Don't want to manually engineer features

❌ **Not ideal for:**
- Very high-dimensional data
- Millions of training examples (computationally expensive)
- Need for fast real-time predictions

---

## Probabilistic Interpretation

### Why Squared Error?

**Question:** Why minimize $\frac{1}{2}\sum(h_\theta(x^{(i)}) - y^{(i)})^2$?  
Why not $|h_\theta(x^{(i)}) - y^{(i)}|$ or $(h_\theta(x^{(i)}) - y^{(i)})^4$?

### Assumptions

Assume the target variable is related to inputs by:
$$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$$

where:
- $\epsilon^{(i)}$ = error term (unmodeled effects + random noise)
- $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ (Gaussian/Normal distribution)
- $\epsilon^{(i)}$ are **IID** (Independently and Identically Distributed)

**IID Assumption:** Error for one house is independent of error for another house  
*(Not perfectly true in reality, but often good enough)*

### Probability Density

Error distribution:
$$p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi\sigma}} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)$$

This implies:
$$p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi\sigma}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$

Or more compactly:
$$y^{(i)} | x^{(i)}; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$$

**Notation:** Semicolon means "parameterized by" (not conditioning, since $\theta$ is not a random variable)

### Maximum Likelihood Estimation (MLE)

**Likelihood of parameters:**
$$L(\theta) = p(\vec{y} | X; \theta) = \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}; \theta)$$

Due to IID assumption, joint probability = product of individual probabilities.

Substituting the Gaussian density:
$$L(\theta) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$

**Log-likelihood** (easier to maximize):
$$\ell(\theta) = \log L(\theta) = m \log \frac{1}{\sqrt{2\pi\sigma}} - \frac{1}{2\sigma^2} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$$

### MLE ⟹ Least Squares

To maximize $\ell(\theta)$:
1. First term is constant (doesn't depend on $\theta$)
2. Second term: maximize $-\frac{1}{2\sigma^2}\sum(y^{(i)} - \theta^T x^{(i)})^2$
3. Equivalent to minimizing $\sum(y^{(i)} - \theta^T x^{(i)})^2$

**Result:** Under Gaussian IID error assumptions, MLE gives us least squares regression!

### Likelihood vs. Probability

**Terminology:**
- **Likelihood of parameters $\theta$**: View as function of $\theta$ with data fixed
- **Probability of data**: View as function of data with $\theta$ fixed
- They evaluate to the same mathematical expression, but used in different contexts

---

## Logistic Regression

### Classification Problem

- **Binary classification:** $y \in \{0, 1\}$
- $0$ = negative class (e.g., benign tumor)
- $1$ = positive class (e.g., malignant tumor)

### Why Not Linear Regression?

**Problem with applying linear regression to classification:**
1. Outputs can be $< 0$ or $> 1$ (doesn't make sense for probabilities)
2. Very sensitive to outliers
3. A single far-away positive example can shift the decision boundary dramatically

**Example:** Adding one more positive example far to the right causes the linear fit to shift, changing predictions for many examples.

### Logistic Function (Sigmoid)

To ensure $0 \leq h_\theta(x) \leq 1$, use:

$$g(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- $g(z) \to 1$ as $z \to \infty$
- $g(z) \to 0$ as $z \to -\infty$
- $g(0) = 0.5$
- Always outputs values between 0 and 1

**Useful derivative property:**
$$g'(z) = g(z)(1 - g(z))$$

### Hypothesis Function

$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

### Probabilistic Model

Assume:
$$P(y = 1 | x; \theta) = h_\theta(x)$$
$$P(y = 0 | x; \theta) = 1 - h_\theta(x)$$

**Compact notation:**
$$p(y | x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$$

**Why this works:**
- If $y = 1$: $p = h_\theta(x)^1 \cdot (1-h_\theta(x))^0 = h_\theta(x)$
- If $y = 0$: $p = h_\theta(x)^0 \cdot (1-h_\theta(x))^1 = 1 - h_\theta(x)$

### Maximum Likelihood for Logistic Regression

**Likelihood:**
$$L(\theta) = \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^{m} (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}}$$

**Log-likelihood:**
$$\ell(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

### Gradient Ascent

**Update rule:**
$$\theta_j := \theta_j + \alpha \frac{\partial}{\partial \theta_j} \ell(\theta)$$

**Note:** Plus sign because we're maximizing (gradient *ascent*)

**After taking derivatives** (see lecture notes for full derivation):
$$\theta_j := \theta_j + \alpha \sum_{i=1}^{m} (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$$

**Stochastic gradient ascent** (one example at a time):
$$\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$$

### Interesting Observation

The update rule *looks identical* to linear regression:
- **Linear regression:** $\theta_j := \theta_j + \alpha(y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}$
- **Logistic regression:** $\theta_j := \theta_j + \alpha(y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}$

**Key difference:** Definition of $h_\theta(x)$ is different!
- Linear: $h_\theta(x) = \theta^T x$
- Logistic: $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$

This similarity is not a coincidence - both are instances of **Generalized Linear Models** (GLMs), covered later.

### Properties of Logistic Regression

✅ **No local optima:** The log-likelihood $\ell(\theta)$ is concave, guaranteeing a single global maximum

✅ **Why sigmoid?** Choice of logistic function ensures:
- Outputs between 0 and 1
- Concave log-likelihood (no local optima)
- Natural interpretation from exponential family distributions

---

## Newton's Method

### Motivation

**Gradient Ascent:**
- Takes small steps in the direction of steepest ascent
- Many iterations needed (100s or 1000s)
- Each iteration is cheap

**Newton's Method:**
- Takes much larger, more intelligent steps
- Far fewer iterations needed (often <10-20)
- Each iteration is more expensive (requires computing and inverting Hessian)

### One-Dimensional Case

**Problem:** Find $\theta$ such that $f(\theta) = 0$

**Newton's update:**
$$\theta := \theta - \frac{f(\theta)}{f'(\theta)}$$

**Intuition:** 
1. Draw line tangent to $f$ at current $\theta$
2. Find where tangent line crosses zero
3. Set next $\theta$ to that point

### For Optimization

To maximize $\ell(\theta)$, we want $\ell'(\theta) = 0$

Set $f(\theta) = \ell'(\theta)$, then:
$$\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}$$

### Multidimensional Newton's Method (Newton-Raphson)

When $\theta \in \mathbb{R}^{n+1}$:
$$\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)$$

where:
- $\nabla_\theta \ell(\theta)$ = gradient vector (first derivatives)
- $H$ = **Hessian matrix** (second derivatives)

**Hessian:**
$$H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}$$

$H$ is an $(n+1) \times (n+1)$ symmetric matrix.

### Quadratic Convergence

**Property:** Number of correct digits roughly doubles with each iteration
- Iteration 1: Error = 0.01
- Iteration 2: Error = 0.0001
- Iteration 3: Error = 0.00000001

This is called **quadratic convergence** - extremely fast near the optimum!

### When to Use Newton's Method

**Use Newton's Method when:**
- ✅ Number of features is moderate ($n < 10,000$)
- ✅ Want very accurate solution
- ✅ Can afford computing/inverting Hessian

**Use Gradient Descent when:**
- ✅ Very high-dimensional ($n > 10,000$)
- ✅ Large-scale problems
- ✅ Cannot afford $(n \times n)$ matrix operations

**Rule of thumb:**
- $n \leq 50$: Strongly consider Newton's method
- $n > 10,000$: Use gradient descent
- In between: Depends on computational resources

### Fisher Scoring

**Note:** When Newton's method is applied to logistic regression log-likelihood, it's also called **Fisher scoring**.

---

## Key Takeaways

### Locally Weighted Regression
- **Non-parametric** algorithm (keeps all training data)
- Fits local straight lines using weighted examples
- No need for manual feature engineering
- Bandwidth $\tau$ controls smoothness (bias-variance tradeoff)
- Best for low-dimensional data with sufficient examples

### Probabilistic Interpretation
- Least squares emerges naturally from:
  - Gaussian errors: $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$
  - IID assumption
  - Maximum Likelihood Estimation
- Provides theoretical justification for squared error

### Logistic Regression
- **Classification algorithm** for binary outcomes
- Uses sigmoid function: $h_\theta(x) = \frac{1}{1+e^{-\theta^T x}}$
- Concave log-likelihood (no local optima)
- Update rule similar to linear regression (but different $h_\theta$)
- Both are Generalized Linear Models

### Newton's Method
- Much faster convergence than gradient descent (quadratic vs. linear)
- Requires computing and inverting Hessian matrix
- Practical for moderate dimensions ($n < 10,000$)
- Trade-off: fewer iterations but more expensive per iteration

---

## Coming Next

**Lecture 4: Generalized Linear Models (GLMs)**
- Unifying framework for linear and logistic regression
- Exponential family distributions
- Deriving new learning algorithms systematically
- Softmax regression for multi-class classification

---

## Additional Resources

- **Lecture Notes:** CS229 Lecture Notes 1 (pages 1-22)
- **Problem Set 1:** 
  - Implement locally weighted regression
  - Experiment with bandwidth parameter $\tau$
  - Implement logistic regression with Newton's method
- **Textbook References:**
  - *Pattern Recognition and Machine Learning* by Bishop
  - *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman

---

## Practice Problems

1. **Feature Engineering**: For housing prices, would you use $x$, $x^2$, $\sqrt{x}$, or $\log(x)$ as features? Why?

2. **LWR**: What happens to locally weighted regression when $\tau \to 0$? When $\tau \to \infty$?

3. **Probabilistic**: If errors were uniformly distributed (not Gaussian), what cost function would MLE give us?

4. **Logistic**: Why is $P(y=1|x) + P(y=0|x) = 1$ always true?

5. **Newton's**: For a 100,000-dimensional problem, why might gradient descent be preferred over Newton's method?

---

**End of Lecture 3 Notes**