## ML Project #1: Optimising NN Architechtures

The Problem: Given any arbitrary (continuous) function $f: [0,1]^d \to [0,1]^m$, and a fixed number of $n$ neurons for the  hidden layers, what is the best configuration for the neural network? How do we strike a balance between width and depth?

We're often told ``depth over width" when designing neural network architectures.  But exactly _how_ deep do they need to be? If we take this mantra to the extreme (arranging our $n$ neurons into a long chain of $n$ hidden layers), then we clearly lose much of the potential complexity our model could capture. 

So what's the optimal structure? How does it vary with the ratios $d:m:n$ ? Does the structure vary wildly for different functions $f: [0,1]^d \to [0,1]^m$, or are there general heuristics we can follow?

Neural networks set out to approximate an unknown function $f: [0,1]^d \to [0,1]^m$ from a handful of fixed observations, by tuning their affine maps. These unknown functions tend to be high dimensional and non-linear, but atleast we know they're _continuous_. Take the classical MNIST dataset for instance. Here, the goal is to approximate an underlying function $f: [0,1]^{784} \to [0,1]^{10}$. Though extremely complicated, a small change in one pixel value should not cause a large change in the model's output (a probability distribution over 10 vector components). Even if that change is enough to turn a vague '3' into an '8', we would expect the correct probability distribution to only shift from 51-49 to 49-51. From here on out, we limit ourselves to the set of all continuous functions  $f: [0,1]^d \to [0,1]^m$, denoted $C([0,1]^d, [0,1]^m)$.

### Choosing "Random" Functions
Since we're experimenting on _function approximators_, we need a good way to 'pick out' continuous functions of the form $f: [0,1]^d \to [0,1]^m$ if we hope to fairly test our neural network configurations. But what does it mean to pick out a function "randomly"?

Take the set $\mathbb{R}$ for example. Picking out an element from this set isn't straightforward - there's no such thing as a "uniform" probability distribution function (PDF) over the reals - we're limited to normal distributions. This should be a good insight into what picking out a _function_ could look like. Let's start with the set of quadratic functions, $\mathbb{R}[X]\_{\leq 2}$ . Since quadratics are solely determined by three real numbers, a reasonable approach to picking out a random $q \in \mathbb{R}[X]\_{\leq 2}$ would be to choose three reals from three normal distributions. Fair enough - and this should work for any "space" whose elements can be described by finitely many real parameters - even multivariable polynomials. The issue with $C([0,1]^d, [0,1]^m)$ is that this space is _infinite_ dimensional - it admits no finite basis. 

### Navigating $C([0,1]^d, [0,1]^m)$ with Polynomials


### Notation

- $\mathcal{N}=(\mathcal{W}, \mathcal{B})$ describes a given neural network, with $\mathcal{W} = (W_1, ... W_\ell)$ denoting weight matrices and $\mathcal{B} = (b_1, ... b_\ell)$ the bias vectors. We'll always use ReLU for activation functions, with the exception of sigmoid for the last layer (otherwise our network would be limited to positive outputs). 

- $NN : [0,1]^d \to [0,1]^m$ with $NN(x):= \text{sigmoid}(...(\textit{ReLU}(W_1 x+b_1)...))$ denotes the _induced_ function by this collection of affine maps + activations.


### Measuring NN Performance

Given some target function $f : [0,1]^d \to [0,1]^m$ and a neural network $\mathcal{N}=(\mathcal{W}, \mathcal{B})$, there are several ways of measuring performance. We'll use the following: given an error range $\epsilon>0$, how many _steps_ does it take (of backpropagation + updating parameters) for our neural network to always be within $\epsilon$ of $f$ ? In other words, when is $||f(x)-NN(x)||<\epsilon$ for all inputs $x \in [0,1]^d$ ?. We can generalise further by ranging over several choices of $\epsilon$, measuring how quickly our network 'squeezed' into a given constraint.




