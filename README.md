## mlml

This repository contains a reimplementation of the [Text Classification](https://github.com/tracel-ai/burn/tree/main/examples/text-classification)
example from the excellent [Burn Deep Learning Framework](https://burn.dev). Instead of classifying
text, it can evaluate stateful expressions in a subset of propositional logic with ~98% accuracy.

Example expressions:

```
[s, p: false] s ∨ p
[m: true; z, f: false] (z ∧ m) → f
[n, v: true; d: false] (n ↔ v) ↔ (n → d)
```

## Usage

The [mlml-dataset](https://github.com/ljedrz/mlml/tree/master/mlml-dataset) contains a binary that
can generate a dataset as specified in the [config.json](https://github.com/ljedrz/mlml/blob/master/config.json)
file. Just use `cargo run` to generate an SQLite database containing a dataset split into training,
validation, and test sets.

[mlml-model](https://github.com/ljedrz/mlml/tree/master/mlml-model) contains a CPU-backed
implementation of a simple transformer-based model that's designed to be used with the
aforementioned datased. The training is quite quick (~3min) on a reasonably beefy CPU.

```
cargo run --example train --release // train the model using the training and validation splits
cargo run --example infer --release // run inference on the test split
```

[mlml-util](https://github.com/ljedrz/mlml/tree/master/mlml-util) just contains a specification of
the [config.json](https://github.com/ljedrz/mlml/blob/master/config.json) file and miscellaneous
helper functions/objects.

## Status

This is a toy project that was configured to be small, simple, and quick to train. It could easily
be extended to support more operators or to solve more complex expressions, though it would likely
require some configuration effort in order to maintain the current accuracy levels.
