# strategic-negotiation-games

Code for final project of UdeM's IFT6756: Game Theory and Machine Learning

---

This implementation is based on the great [End to End Negotiator](https://github.com/facebookresearch/end-to-end-negotiator) code project.

We've followed the given instructions to train all the models, which are files `*.th`.

A comprehensive code demo is found at [the project's Jupyter](Strategic.ipynb).
This file walks the user through the steps on how we produced our reward plots (found in [`plots/`](plots/)).

Moreover, all the experiments can be run directly via [`reinforce.py`](reinforce.py). The parameters are specified as `argparse` arguments, while some of the default are also found in [`run.sh`](run.sh).

Finally, the main implemenation of the LOLA algorithm was added to the `RlAgent` class of [`agent.py`](agent.py).
The 3-agent and 2-vs-1 settings were coded as two independent dialog classes of [`dialog.py`](dialog.py).