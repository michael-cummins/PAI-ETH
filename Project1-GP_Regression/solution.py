import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, WhiteKernel
import pandas as pd

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0

class RFF_RBF:
    def __init__(self, sigma_f: float = 1, length: float = 1):
        self.sigma_f = sigma_f
        self.l = length

    def __call__(self, x1: np.array, x2: np.array) -> float:
        return float(((self.sigma_f**2)/2*np.pi)*np.exp(-(np.linalg.norm(x1 - x2)**2)/2*(self.l**2)))

# Helper function to calculate the respective covariance matrices
def cov_matrix(x1, x2, cov_function) -> np.array:
    return np.array([[cov_function(a, b) for a in x1] for b in x2])

class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        self.covariance_function = RFF_RBF()
        # print(self.rng)
        # TODO: Add custom initialization for your model here if necessary

    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        # at_values = test_features
        # k_lower_left = cov_matrix(self.data_x, at_values,
        #                           self.covariance_function)
        # k_lower_right = cov_matrix(at_values, at_values,
        #                            self.covariance_function)

        # # Mean.
        # mean_at_values = np.dot(
        #     k_lower_left,
        #     np.dot(self.data_y,
        #            self._inverse_of_covariance_matrix_of_input.T).T).flatten()

        # # Covariance.
        # cov_at_values = k_lower_right - \
        #     np.dot(k_lower_left, np.dot(
        #         self._inverse_of_covariance_matrix_of_input, k_lower_left.T))

        # # Adding value larger than machine epsilon to ensure positive semi definite
        # cov_at_values = cov_at_values + 3e-7 * np.ones(
        #     np.shape(cov_at_values)[0])

        # var_at_values = np.diag(cov_at_values)

        # # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean = np.zeros(test_features.shape[0], dtype=float)
        gp_std = np.zeros(test_features.shape[0], dtype=float)
        # # print('Infering')
        # gp_mean, gp_std = self.model.predict(test_features, return_std=True)        
        # test_features = (test_features - self.x_mean)/self.x_std
        gp_mean, gp_std = self.model.predict(test_features, return_std=True)
        # # TODO: Use the GP posterior to form your predictions here
        # predictions = gp_mean
        predictions = gp_mean

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray, train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        data_x = pd.DataFrame(train_features)
        labels = pd.DataFrame(train_GT)
        data = pd.concat([data_x, labels], axis=1)
        data = data.sample(frac=0.1, replace=True, random_state=1)
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:, -1])

        # self.y_mean = y.mean()
        # self.x_mean = x.mean()
        # self.x_std = x.std()
        # y = y - self.y_mean
        # x = x - self.x_mean

        kernel  = RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1)
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-09, normalize_y = True, n_restarts_optimizer=2, random_state=0)
        
        # self._inverse_of_covariance_matrix_of_input = np.linalg.inv(cov_matrix(X, X, self.covariance_function))
        # TODO: Fit your model here

        self.model.fit(X=X, y=y)
        

    def cluster(self, train_x, train_y):
        N = int(len(train_x)*0.2)
        cls_model = KMeans(n_clusters=N)
        cls_model.fit(train_x)
        print(cls_model.cluster_centers_)
        closest, _ = pairwise_distances_argmin_min(X=cls_model.cluster_centers_, Y=train_x, metric='euclidean',axis=1)
        clustered_x = np.array([train_x[i] for i in closest])
        print(clustered_x)
        clustered_y = np.array([train_y[i] for i in closest])
        return clustered_x, clustered_y


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)



def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT,train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
