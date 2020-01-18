//! Naïve Bayes is a relatively simple machine learning technique for classification. It's called
//! "naïve" because it assumes that the features are conditionally independent given the class
//! label, i.e. that the relationships between features are not important when predicting the class
//! of a given input.
//!
//! Although it's not a sophisticated model, it:
//!
//! - Is fast, both to train and to evaluate
//! - Is simple
//! - Can be trained online
//! - Allows the user to specify a prior
//! - Captures uncertainty
//!
//! # Example
//!
//! In this example, we'll train a classifier to learn the boolean "NOT" function, but with some
//! unimportant data also included in the training examples.
//!
//! ```
//! use baby_bayes::{Beta, BroadcastBernoulli, BroadcastBeta, NaiveBayes};
//! use ndarray::array;
//!
//! // Using improper Beta priors for simplicity of testing
//! let mut naive_bayes = NaiveBayes::new(3, BroadcastBeta::new(0.0, 0.0), Beta::new(0.0, 0.0));
//!
//! // Train the model on two data points
//! naive_bayes.train(array!(true, false, true).view(), false);
//! naive_bayes.train(array!(true, false, false).view(), true);
//!
//! // What is the marginal distribution of y?
//! assert_eq!(0.5, naive_bayes.y().p);
//!
//! // We can also predict an X given a y
//! assert_eq!(array!(1.0, 0.0, 1.0), naive_bayes.x_given_y(false).p);
//! assert_eq!(array!(1.0, 0.0, 0.0), naive_bayes.x_given_y(true).p);
//! ```
//!
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::Rng;

pub const C: usize = 2;

/// # Math
///
/// I've taken this notation from section 3.5 of *Machine Learning: A Probabilistic Perspective* by
/// Kevin Murphy:
///
/// - $D ∈ ℤ_{≥0}$: the number of binary features in each input vector
/// - $C = 2$: the number of classes in the output. Since this implementation is a binary
///   classifier, $C$ is always 2.
/// - $X = (X_1, X_2, ..., X_D)$ where each $X_j ∈ {0, 1}$: a data vector, where each element is a
///   binary variable
/// - $y ∈ {0, 1}$: the class assigned to the data vector
/// - $N_C = (N_0, N_1)$ where each $N_c ∈ ℤ_{≥0}$: the number of times we've seen $y = c$
/// - $N_{jc}$: the number of times we've seen $(x, y) = (j, c)$
///
#[derive(Debug, PartialEq)]
pub struct NaiveBayes {
    pub prior_x: BroadcastBeta,
    pub prior_y: Beta,
    pub n_c: Array1<f64>,  // Length C
    pub n_jc: Array2<f64>, // D x C
}

impl NaiveBayes {
    pub fn new(d: usize, prior_x: BroadcastBeta, prior_y: Beta) -> Self {
        Self {
            prior_x,
            prior_y,
            n_c: Array1::zeros(C),
            n_jc: Array2::zeros((d, C)),
        }
    }

    pub fn train(&mut self, x: ArrayView1<bool>, y: bool) {
        assert_eq!(x.dim(), self.d());
        let c = if y { 1 } else { 0 };
        self.n_c[c] += 1.0;
        for (j, &x) in x.indexed_iter() {
            if x {
                self.n_jc[(j, c)] += 1.0;
            } // We don't need to record the else clause b/c we already have enough info
        }
    }

    pub fn d(&self) -> usize {
        let (d, _c) = self.n_jc.dim();
        d
    }

    pub fn x(&self) -> BroadcastBernoulli {
        // TODO maybe need to multiply prior by size of x... maybe not?
        BroadcastBernoulli::new(
            (self.n_jc.sum_axis(Axis(1)).to_owned() + self.prior_x.alpha)
                / (self.n_c.sum() + self.prior_x.alpha + self.prior_x.beta),
        )
    }

    pub fn x_given_y(&self, y: bool) -> BroadcastBernoulli {
        let c = if y { 1 } else { 0 };
        BroadcastBernoulli::new(
            (self.n_jc.column(c).to_owned() + self.prior_x.alpha)
                / (self.n_c[c] + self.prior_x.alpha + self.prior_x.beta),
        )
    }

    // The distribution of y, marginalized over x
    pub fn y(&self) -> Bernoulli {
        Bernoulli::new(
            (self.prior_y.alpha + self.n_c[1])
                / (self.prior_y.alpha + self.prior_y.beta + self.n_c.sum()),
        )
    }
}

#[derive(Debug, PartialEq)]
pub struct Bernoulli {
    pub p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

#[derive(Debug, PartialEq)]
pub struct BroadcastBernoulli {
    pub p: Array1<f64>,
}

impl BroadcastBernoulli {
    pub fn new(p: Array1<f64>) -> Self {
        Self { p }
    }

    pub fn sample<R: Rng>(&self, mut rng: R) -> Array1<bool> {
        self.p.map(|&p| rng.gen::<f64>() < p)
    }
}

#[derive(Debug, PartialEq)]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }
}

#[derive(Debug, PartialEq)]
pub struct BroadcastBeta {
    pub alpha: f64,
    pub beta: f64,
}

impl BroadcastBeta {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    pub fn whatever() -> Self {
        BroadcastBeta::new(1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Bernoulli, Beta, BroadcastBernoulli, BroadcastBeta, NaiveBayes};
    use ndarray::array;

    #[test]
    fn y_0_features() {
        let d = 0;

        // Given just the prior on x
        let mut naive_bayes = NaiveBayes::new(d, BroadcastBeta::whatever(), Beta::new(0.5, 1.5));
        assert_eq!(naive_bayes.y(), Bernoulli::new(0.25));

        // Seeing trues will increase E[y]
        naive_bayes.train(array!().view(), true);
        naive_bayes.train(array!().view(), true);
        assert_eq!(naive_bayes.y(), Bernoulli::new(0.625));

        // Seeing falses will decrease E[y]
        naive_bayes.train(array!().view(), false);
        naive_bayes.train(array!().view(), false);
        naive_bayes.train(array!().view(), false);
        naive_bayes.train(array!().view(), false);
        assert_eq!(naive_bayes.y(), Bernoulli::new(0.3125));
    }

    #[test]
    fn x_1_feature() {
        let float_doesnt_matter = 1.0;
        let bool_doesnt_matter = true;

        // Given just the prior on x
        let mut naive_bayes = NaiveBayes::new(
            1,
            BroadcastBeta::new(0.25, 0.75),
            Beta::new(float_doesnt_matter, float_doesnt_matter),
        );
        assert_eq!(
            naive_bayes.x(),
            BroadcastBernoulli::new(array!((0.25 + 0.0) / (0.25 + 0.75 + 0.0 + 0.0)))
        );

        // Seeing true will increase E[x]
        naive_bayes.train(array!(true).view(), bool_doesnt_matter);
        assert_eq!(
            naive_bayes.x(),
            BroadcastBernoulli::new(array!((0.25 + 1.0) / (0.25 + 0.75 + 1.0 + 0.0)))
        );

        // Seeing false will decrease E[x]
        naive_bayes.train(array!(false).view(), bool_doesnt_matter);
        assert_eq!(
            naive_bayes.x(),
            BroadcastBernoulli::new(array!((0.25 + 1.0) / (0.25 + 0.75 + 1.0 + 1.0)))
        );

        // Seeing falses will decrease E[x]
        naive_bayes.train(array!(false).view(), bool_doesnt_matter);
        naive_bayes.train(array!(false).view(), bool_doesnt_matter);
        assert_eq!(
            naive_bayes.x(),
            BroadcastBernoulli::new(array!((0.25 + 1.0) / (0.25 + 0.75 + 1.0 + 3.0)))
        );
    }

    #[test]
    fn x_given_y_1_feature() {
        let float_doesnt_matter = 1.0;

        // Given just the prior on x
        let mut naive_bayes = NaiveBayes::new(
            1,
            BroadcastBeta::new(0.25, 0.75),
            Beta::new(float_doesnt_matter, float_doesnt_matter),
        );
        let expected_no_data =
            BroadcastBernoulli::new(array!((0.25 + 0.0) / (0.25 + 0.75 + 0.0 + 0.0)));
        assert_eq!(expected_no_data, naive_bayes.x_given_y(false),);
        assert_eq!(expected_no_data, naive_bayes.x_given_y(true),);

        // Seeing a point where y = 1 will only change the distribution for y = 1
        naive_bayes.train(array!(true).view(), true);
        assert_eq!(expected_no_data, naive_bayes.x_given_y(false),);
        assert_eq!(
            BroadcastBernoulli::new(array!((0.25 + 1.0) / (0.25 + 0.75 + 1.0 + 0.0))),
            naive_bayes.x_given_y(true),
        );
    }
}
