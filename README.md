# baby-bayes

Naïve Bayes is a relatively simple machine learning technique for classification. It's called
"naïve" because it assumes that the features are conditionally independent given the class
label, i.e. that the relationships between features are not important when predicting the class
of a given input.

Although it's not a sophisticated model, it:

- Is fast, both to train and to evaluate
- Is simple
- Can be trained online
- Allows the user to specify a prior
- Captures uncertainty

## Example

In this example, we'll train a classifier to learn the boolean "NOT" function, but with some
unimportant data also included in the training examples.

```rust
use baby_bayes::{Beta, BroadcastBernoulli, BroadcastBeta, NaiveBayes};
use ndarray::array;

// Using improper Beta priors for simplicity of testing
let mut naive_bayes = NaiveBayes::new(3, BroadcastBeta::new(0.0, 0.0), Beta::new(0.0, 0.0));

// Train the model on two data points
naive_bayes.train(array!(true, false, true).view(), false);
naive_bayes.train(array!(true, false, false).view(), true);

// What is the marginal distribution of y?
assert_eq!(0.5, naive_bayes.y().p);

// We can also predict an X given a y
assert_eq!(array!(1.0, 0.0, 1.0), naive_bayes.x_given_y(false).p);
assert_eq!(array!(1.0, 0.0, 0.0), naive_bayes.x_given_y(true).p);
```

