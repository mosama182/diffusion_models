
import matplotlib.pyplot as plt

from data import generate_annular, generate_spiral

# Implements a Linear point wise flow
class LinearFlow():
    """
    Linear pointwise flow for pair (xo, x1)
    That is v^{x0, x1}
    """

    def __init__(self) -> None:
        pass

    def compute(self, xo, x1, t):
        """
        Evaluate pointwise flow at time t
        That is v_{t}^{xo, x1}
        xo: N x input_dim tensor or numpy of target data points
        x1: N x input_dim tensor or numpy of source data points
        t: N x 1 tenor of time points uniformly sampled [0, 1]
        """
        v = xo - x1 # no dependence of t for LinearFlow

        return v
    
    def runflow(self, xo, x1, t):
        """
        Run pointwise flow corresponding to pair (xo, x1)
        until time t to produce x_{t}
        """
        xt = t * x1 + (1 - t) * xo

        return xt
    

if __name__ == "__main__":
    
    # samples from annular data
    num_samples = 50
    source_data = generate_annular(num_samples, inner_radius=2, outer_radius=2.25)
    target_data = generate_spiral(num_samples)


    # flow
    flow = LinearFlow()
    t = 0
    xt = flow.runflow(target_data, source_data, t)

    # plot
    plt.figure()
    plt.scatter(source_data[:, 0], source_data[:, 1], label=r'source samples')
    plt.scatter(target_data[:, 0], target_data[:, 1], label=r'target samples')
    plt.scatter(xt[:, 0], xt[:, 1], label=f'Samples at $t={t}$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid()
    plt.legend()
    plt.show()


