//! This example uses iterative user feedback to generate simple quilts that can easily be printed
//! to the terminal. This is based on a Naïve Bayes classifier where `x` represents the design of a
//! quilt and `y` represents whether the user liked that quilt. To generate quilts that the user
//! likes more, we sample `x|y=1`.
//!
//!
//!
//! Here are some generated quilts that I liked:
//!
//! ```
//! ██       ██
//! █         █
//!      █
//!    █ █ █
//!     █ █
//!   ██   ██
//!     █ █
//!    █ █ █
//!      █
//! █         █
//! ██       ██
//! ```
//!
//! ```
//! █  █████  █
//!   █     █
//!  █ █████ █
//! █ ███ ███ █
//! █ ██ █ ██ █
//! █ █ █ █ █ █
//! █ ██ █ ██ █
//! █ ███ ███ █
//!  █ █████ █
//!   █     █
//! █  █████  █
//! ```
//!
//! ```
//! █    █    █
//!      █
//!      █
//!    █ █ █
//!     █ █
//! ████ █ ████
//!     █ █
//!    █ █ █
//!      █
//!      █
//! █    █    █
//! ```
//!
//! ```
//! ███████████
//! ███ ███ ███
//! ██   █   ██
//! █   █ █   █
//! ██ █   █ ██
//! ███  █  ███
//! ██ █   █ ██
//! █   █ █   █
//! ██   █   ██
//! ███ ███ ███
//! ███████████
//! ```
use baby_bayes::{Beta, BroadcastBeta, NaiveBayes};
use itertools::Itertools as _;
use ndarray::Array1;
use rand::thread_rng;
use std::fmt::{Display, Error, Formatter, Write};
use std::io::stdin;

const INDEX_MIN: isize = -5;
const INDEX_MAX: isize = 6;
const X_SIZE: usize = 21;

const MS: &[isize] = &[
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5,
];
const NS: &[isize] = &[
    0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5,
];

struct Quilt {
    x: Array1<bool>,
}

impl Display for Quilt {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        for i in INDEX_MIN..INDEX_MAX {
            for j in INDEX_MIN..INDEX_MAX {
                // Map the sampled x's so that the final quilt has 8 axes of reflective symmetry
                let m = i.abs().min(j.abs());
                let n = i.abs().max(j.abs());
                let (x_idx, _) = MS
                    .iter()
                    .zip(NS)
                    .find_position(|(&an_m, &an_n)| (an_m, an_n) == (m, n))
                    .unwrap();
                f.write_char(if self.x[x_idx] { '█' } else { ' ' })?
            }
            f.write_char('\n')?;
        }
        Ok(())
    }
}

fn main() {
    let mut naive_bayes =
        NaiveBayes::new(X_SIZE, BroadcastBeta::new(1.0, 1.0), Beta::new(1.0, 2.0));

    loop {
        let x = naive_bayes.x_given_y(true).sample(thread_rng());
        println!("{}", Quilt { x: x.clone() });
        println!("Is this a promising quilt? (y/n): ");
        let input = {
            let mut buffer = String::new();
            stdin().read_line(&mut buffer).unwrap();
            buffer
        };
        match input.as_str().chars().nth(0).unwrap() {
            'n' => naive_bayes.train(x.view(), false),
            'y' => naive_bayes.train(x.view(), true),
            input => println!("Unrecognized input: {}", input),
        }
        println!("{:#?}", naive_bayes);
    }
}
