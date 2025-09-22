## ML Project #1: Optimising NN Architechtures

The Problem: Given any arbitrary (continuous) function $f: [0,1]^d \to [0,1]^m$, and a fixed number of $n$ neurons for the  hidden layers, what is the best configuration for the neural network? How do we strike a balance between width and depth?

We're often told ``depth over width" when designing neural network architectures.  But exactly \textit{how} deep do they need to be? If we take this mantra to the extreme (arranging our $n$ neurons into a long chain of $n$ hidden layers), then we clearly lose much of the potential complexity our model could capture. 

So what's the optimal structure? How does it vary with the ratios $d:m:n$ ? Does the structure vary wildly for different functions $f: [0,1]^d \to [0,1]$, or are there general heuristics we can follow?

Neural networks set out to approximate an unknown function $f: [0,1]^d \to [0,1]^m$ from a handful of fixed observations, by tuning their affine map parameteres. These unknown functions tend to be high dimensional and non-linear - but atleast we know they're _continuous_. Take the classical MNIST dataset for instance. Here, the goal is to approximate an underlying function $f: [0,1]^{784} \to [0,1]^{10}$. Though extremely complicated, a small change in one pixel value should not cause a large change in the model's output (a probability distribution over 10 vector components). Even if that change is enough to turn a vague '3' into an '8', we would expect the correct probability distribution to only shift from 51-49 to 51-49. From here on out, we limit ourselves to the set of all continuous functions  $f: [0,1]^d \to [0,1]^m$, denoted $C([0,1]^d, [0,1]^m)$.

### Navigating $C([0,1]^d, [0,1]^m)$ with the Stone-Weierstrass Theorem
Since we're experimenting on _function approximators_, we need a good way to 'pick out' continuous functions of the form $f: [0,1]^d \to [0,1]^m$... but choosing such functions from the ocean $C([0,1]^d, [0,1]^m)$ is difficult - most are formless abstractions with no closed form, unattainable from composing elementary functions. What we _do_ have, however, is a useful theorem from Topology: _The Stone-Weierstrass Theorem_.
