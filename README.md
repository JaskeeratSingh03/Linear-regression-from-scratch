# Linear Regression from Scratch

A small, hand-rolled implementation of linear regression in pure Python — no numpy, no scikit-learn, no external dependencies. Built to be readable enough to explain line-by-line in a viva.

The project lives at the intersection of **math**, **ML**, and **software engineering**: enough math to feel real, enough code to actually run, small enough to hold in your head.

---

## What is this?

A console app that learns a straight line `y = w * x + b` from data using **gradient descent**.

You can either feed it synthetic noisy data (great for demos) or a CSV file. It will:

1. Find the `w` and `b` that minimize the Mean Squared Error.
2. Print the learned equation.
3. Predict values for new `x` on demand.
4. Plot the cost curve (does the model actually learn?) and the data + line together (does the line actually fit?).

---

## The math

Three equations. That's the whole project.

**Prediction** (the forward pass):

```
y_pred[i] = w * x[i] + b
```

**Cost** (how wrong we are — Mean Squared Error):

```
               1          n
MSE   =   --------  *   sum (y_pred[i] - y[i]) ** 2
              2n         i=1
```

The `1/(2n)` is a teaching trick — the `2` cancels when we differentiate, leaving clean gradients.

**Gradients** (which direction to nudge each parameter):

```
               n
dw   =  (1/n) * sum (y_pred[i] - y[i]) * x[i]
               i=1

               n
db   =  (1/n) * sum (y_pred[i] - y[i])
               i=1
```

**Update rule** (the actual learning step):

```
w   :=   w   -   alpha * dw
b   :=   b   -   alpha * db
```

`alpha` is the **learning rate**. The minus sign says: if our prediction was too high (positive error), push `w` *down*.

We repeat that update loop `n_iters` times. Each iteration the cost drops. Eventually it stops dropping and we say the model has **converged**.

> **One implementation note:** the model standardizes features and targets (mean 0, std 1) before training. This keeps gradients well-behaved across datasets with very different scales. The coefficients you see printed (`w`, `b`) are converted back to the original units for display.

---

## How to run

From the project root:

```bash
python3 main.py
```

You'll see a menu. Recommended first run:

1. Pick `1` — generate synthetic data and train.
2. Pick `3` — see the learned equation.
3. Pick `6` — see the data and the line on one ASCII plot.
4. Pick `5` — see how the cost dropped over iterations.
5. Pick `0` — exit.

You can also pass an x value (option 4) to get a prediction:

```
For x = 42.5, predicted y = 115.8334
```

---

## How to run tests

```bash
python3 -m unittest discover -s tests
```

You should see:

```
...............
----------------------------------------------------------------------
Ran 15 tests in 0.05s

OK
```

The tests cover:

- Cost is `0` when predictions match targets exactly.
- Cost is positive when predictions are imperfect.
- Cost decreases monotonically across iterations (gradient descent is actually descending).
- The model recovers a perfect `y = 2x + 5` line with no noise.
- The model approximates a noisy `y = 2x + 5 + noise` within tolerance.
- The synthetic generator is reproducible with a fixed seed and yields different data for different seeds.
- `load_csv` skips blank lines and reads the sample CSV correctly.

---

## Sample output

**Option 3 — learned equation:**

```
Learned equation: y = 2.4442 * x + 11.9051
```

(Generated from synthetic data with true line `y = 2.5*x + 10`. Close enough!)

**Option 5 — cost vs iteration (cost drops sharply, then flattens):**

```
cost: 0.01 -> 0.50   (1000 iterations)
```

**Option 6 — data + regression line:**

```
  y_max = 255.38
|                                                     o oo*o*
|                                                     *o*   o
|                                                 ooo*
|                                              oo*oo
|                                         o ***
|                                     o  ***o
|                             o       o**o
|                               ooo*o*
|                             o**o o
|                       oo  ***
|                       o***
|                    **o*
|               oo*o*
|            o o**
|           *o*
|       **o* o
|    oo* o
| o**
|*o  o
|o
+------------------------------------------------------------
  y_min = 1.94
  learned line: y = 2.4442 * x + 11.9051
```

The `o`s are data points. The row of `*`s is the learned regression line. Notice how the line cuts diagonally through the cloud of points.

---

## How to explain this in a viva

If the examiner asks *"what does your project do?"*, answer in three sentences:

1. **The model.** *"I'm implementing linear regression from scratch. The model is a straight line `y = w*x + b`, and the goal is to find the `w` and `b` that best fit a dataset."*

2. **The learning.** *"I measure how good a guess is using Mean Squared Error. I take the partial derivative of that cost with respect to `w` and `b` to get gradients, then I update each parameter by subtracting the gradient times a small learning rate — that pushes the parameter in the direction that reduces error."*

3. **The loop.** *"I repeat the predict → cost → gradient → update cycle for many iterations. The cost goes down each time and the line gradually fits the data better. The ASCII plot in the demo shows the final line passing through the points."*

If they ask *"why standardize the inputs?"*, the answer is: *"Without it, gradients can have very different magnitudes depending on the scale of `x` and `y`, which makes the same learning rate either too small (slow convergence) or too big (divergence). Standardizing gives all features comparable scale, so a single learning rate works across many datasets."*

---

## File layout

```
linear-regression-from-scratch/
├── linear_regression/
│   ├── __init__.py
│   ├── model.py            ← LinearRegression class: fit, predict, gradient descent
│   ├── dataset.py          ← make_synthetic(), load_csv()
│   └── visualize.py        ← ASCII plots (cost curve, scatter+line)
├── tests/
│   ├── test_model.py       ← math correctness + convergence
│   └── test_dataset.py     ← data loading + reproducibility
├── data/
│   └── sample.csv          ← 15 noisy points
├── main.py                 ← console menu
├── requirements.txt        ← (stdlib only)
└── README.md
```

---

## What's next (extensions)

This code generalizes more easily than it looks. The `weights` list is already ready for N features, not just 1. Some natural follow-ups:

- **Multiple linear regression.** Add a second column to the CSV. The model already handles it — only the menu / plot needs to change.
- **Polynomial features.** Add `x**2`, `x**3` as new columns. Same model, different features. Fits curves, not just lines.
- **Logistic regression.** Swap MSE for cross-entropy, swap the linear activation for sigmoid. The gradient-descent skeleton is identical; only the cost changes.
- **Compare against scikit-learn.** Train both on the same data and print both equations side-by-side. Great "is my answer right?" check for a viva.

---

## License

MIT.
