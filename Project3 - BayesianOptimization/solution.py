import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from warnings import catch_warnings, simplefilter

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        f_kernel = Matern(length_scale=0.5, nu=2.5)*0.5 + WhiteKernel(noise_level=0.15)
        v_kernel = Matern(length_scale=0.5, nu=2.5)*1.414 + WhiteKernel(noise_level=0.0001) + ConstantKernel(1.5)
        self.x_train = []
        self.f_train = []
        self.v_train = []
        self.start = True
        
        self.GP_f = GaussianProcessRegressor(kernel=f_kernel)
        self.GP_v = GaussianProcessRegressor(kernel=v_kernel)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        with catch_warnings():
            simplefilter('ignore')
            if self.start:
                x = 2.222
                self.start = False
            else:
                x = self.optimize_acquisition_function()
        return x


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """
        with catch_warnings():
            simplefilter('ignore')
            def objective(x):
                return -self.acquisition_function(x)

            f_values = []
            x_values = []

            # Restarts the optimization 20 times and pick best solution
            for _ in range(20):
                x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                    np.random.rand(domain.shape[0])
                result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                    approx_grad=True)
                x_values.append(np.clip(result[0], *domain[0]))
                f_values.append(-result[1])

            ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        with catch_warnings():
            simplefilter('ignore')
            x = np.reshape(x, (-1,1))

            mean_pred, std_pred = self.GP_f.predict(x, return_std=True) 
            mean_speed, std_speed = self.GP_v.predict(x, return_std=True)

            pred_samples = self.GP_f.predict(np.reshape(self.x_train, (-1,1)),return_std=False)
            fmax = np.max(pred_samples)

            Z = (mean_pred-fmax-0.01)/std_pred 
            EI = (Z*norm.cdf(Z)+norm.pdf(Z))*std_pred
            c = 1-norm.cdf(x=1.2, loc=mean_speed, scale=std_speed)
            EI_c = c*EI
           
        return EI_c


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        with catch_warnings():
            simplefilter('ignore')
            self.x_train.append(x)
            self.f_train.append(f)
            self.v_train.append(v)
            self.GP_f.fit(X=np.reshape(self.x_train, (-1,1)), y=np.reshape(self.f_train, (-1,1)))
            self.GP_v.fit(X=np.reshape(self.x_train, (-1,1)), y=self.v_train)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        idx = 0
        f_star = -np.inf
        for i in range(len(self.x_train)):
            if self.f_train[i] > f_star and self.v_train[i] > 1.2:
                f_star = self.f_train[i]
                idx = i
        return self.x_train[idx]


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        print(f'j = {j}')
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    print(f'solution shape = {solution.shape}')
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()