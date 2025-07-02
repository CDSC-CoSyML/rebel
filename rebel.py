import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, tpe, fmin, Trials
from sympy import symarray, Mul, Add, lambdify
from itertools import combinations_with_replacement, chain
from scipy.stats import wasserstein_distance

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
from sympy2jax import SymbolicModule

from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem

from jax import numpy as jnp
import jax

# --- The rebel class definition (as provided) ---
class rebel(object):
    def __init__(self, **kwargs):
        self.data = kwargs.get("data", None)
        self.data_dot = kwargs.get("data_dot", None)
        self.h = kwargs.get("stepsize", 1)
        self.lb = kwargs.get("lb", np.log(0.001))
        self.ub = kwargs.get("ub", np.log(1))
        self.IC = kwargs.get("IC", "BIC")
        self.evals = kwargs.get("evals", 200)
        self.error = kwargs.get("error", "integrate")
        self.order = kwargs.get("order", 3)
        self.alpha_lb = kwargs.get("lb", 0.1)
        self.alpha_ub = kwargs.get("ub", 0.9)
        self.multi = True

    ###### methods for initialization #######
    def preprocess(self):
        print("calculating derivative with splines")
        self.data_dot = np.zeros(self.data.shape)
        # Using np.gradient to approximate the derivative
        self.data_dot = np.gradient(self.data, self.h, axis=1)
        print("Set up rhs-library")
        self.x = symarray("x", self.data.shape[0])
        self.poly_liste = list(chain.from_iterable(
            [Mul(*[l0 for l0 in l]) for l in combinations_with_replacement(self.x, i)]
            for i in range(0, self.order + 1)))
        self.sigma = symarray("s", len(self.poly_liste))
        self.poly_summe = Add(*list(Mul(*(s, y)) for s, y in zip(self.sigma, self.poly_liste)))
        self.f_poly = SymbolicModule(self.poly_summe)
        
        self.lib = np.ones(self.data.shape[1])
        for i in range(1, len(self.poly_liste)):
            self.lib = np.append(self.lib, [lambdify(self.x, self.poly_liste[i])(*self.data)])
        self.lib = self.lib.reshape(-1, self.data.shape[1]).T
        self.init_mask = np.ones((self.data.shape[0], self.sigma.size), dtype=bool).T
        
        self.term = ODETerm(self.rhs)
        self.ode_solver = Dopri5()
        self.ts = np.arange(self.data.shape[1]) * self.h
        self.t1 = self.ts[-1]
        self.saveat = SaveAt(ts=np.arange(self.data.shape[1]) * self.h)

        if self.error == "mixed": 
            self.alpha_lb = 0.1
            self.alpha_ub = 0.9
        elif self.error == "L2":
            self.alpha_lb = 0.99999
            self.alpha_ub = 1
        elif self.error == "W":
            self.alpha_lb = 0
            self.alpha_ub = 0.00001

        self.space = [hp.uniform('alpha', self.alpha_lb, self.alpha_ub)]
        if self.multi:
            for i in range(self.data.shape[0]):
                self.space += [hp.loguniform('l' + str(i), self.lb, self.ub)]
        else:
            self.space += [hp.loguniform('l', self.lb, self.ub)]

        self.solver = sinkhorn.Sinkhorn()

    def create_opt_space(self):
        self.space = [hp.uniform('alpha', self.alpha_lb, self.alpha_ub)]
        if self.multi:
            for i in range(self.data.shape[0]):
                self.space += [hp.loguniform('l' + str(i), self.lb, self.ub)]
        else:
            self.space += [hp.loguniform('l', self.lb, self.ub)]

    ###### methods for model estimation #######
    # sparsify with SINDy algorithm and sparsity knob lamb
    def sparsifyDynamics(self, lamb):
        xi = np.zeros(self.init_mask.shape)
        for ix in range(self.data_dot.shape[0]):
            xi[self.init_mask[:, ix], ix] = np.linalg.lstsq(
                self.lib[:, self.init_mask[:, ix]], self.data_dot.T[:, ix], rcond=None)[0]

        binds = np.copy(self.init_mask)
        if len([lamb]) > 1:
            for i in range(25):
                for j in range(xi.shape[1]):
                    binds[:, j] = np.abs(xi[:, j]) > lamb[j]
                    xi[np.abs(xi[:, j]) < lamb[j], j] = 0

                for ix in range(self.data_dot.shape[0]):
                    xi[binds[:, ix], ix] = np.linalg.lstsq(
                        self.lib[:, binds[:, ix]], self.data_dot.T[:, ix], rcond=None)[0]
        else: 
            for i in range(25):
                binds = np.abs(xi) > lamb
                xi[np.abs(xi) < lamb] = 0
                for ix in range(self.data_dot.shape[0]):
                    xi[binds[:, ix], ix] = np.linalg.lstsq(
                        self.lib[:, binds[:, ix]], self.data_dot.T[:, ix], rcond=None)[0]
        return xi

    # estimate sparse model and integrate it
    def calc_y_model(self, l):
        sigma = self.sparsifyDynamics(l)
        print(self.data_dot.shape, sigma.shape, self.lib.T.shape)
        diff_sindy = ((self.data_dot - np.dot(sigma.T, self.lib.T)).flatten()) ** 2
        dict_s = [{f"s_{i}": value for i, value in enumerate(sigma[:, j])} for j in range(self.data.shape[0])]
        solution = diffeqsolve(self.term, self.ode_solver, t0=0, t1=self.t1, dt0=self.h,
                                y0=self.data[:, 0], args=dict_s, saveat=self.saveat)
        estimate = solution.ys.T
        
        return estimate, sigma, diff_sindy

    def rhs(self, t, y, args):
        x_dict = {f"x_{i}": value for i, value in enumerate(y)}
        return jnp.array([self.f_poly(**{**x_dict, **args[i]}) for i in range(self.data.shape[0])])
    
    def wasserstein(self, x, y):
        geom = pointcloud.PointCloud(x, y)
        ot_prob = linear_problem.LinearProblem(geom)
        ot = self.solver(ot_prob)
        return jnp.sum(ot.matrix * ot.geom.cost_matrix)

    def BIC(self, space):
        alpha = space[0]
        l = space[1:]
        sigma = self.sparsifyDynamics(l)
        dict_s = [{f"s_{i}": value for i, value in enumerate(sigma[:, j])} for j in range(self.data.shape[0])]
        solution = diffeqsolve(self.term, self.ode_solver, t0=0.0, t1=float(self.t1), dt0=float(self.h),
                                y0=jnp.array(self.data[:, 0]), args=dict_s, saveat=self.saveat)
        yest = solution.ys.T
        diff_int = ((yest - self.data).flatten()) ** 2
        diff_int = diff_int.mean() / self.norm_L2
        diff_w = self.wasserstein(yest, self.data)
        diff_w = (diff_w - self.set_off_wasserstein) / self.norm_wasserstein

        diff = alpha * diff_int + (1 - alpha) * diff_w

        N_t = self.data.shape[1]
        logL = -self.data.shape[1] * np.log(diff) / 2
        p = np.count_nonzero(sigma.flatten())
        return -2 * logL + p * np.log(N_t)

    def BIC_sindy(self, space):
        alpha = space[0]
        l = space[1:]
        sigma = self.sparsifyDynamics(l)
        diff = ((self.data_dot - np.dot(sigma.T, self.lib.T)).flatten()) ** 2  # SINDy
        diff = diff.mean()
        N_t = self.data.shape[1]
        logL = -self.data.shape[1] * np.log(diff) / 2
        p = np.count_nonzero(sigma.flatten())
        return -2 * logL + p * np.log(N_t)

    def rms(self, space):
        alpha = space[0]
        l = space[1:]
        sigma = self.sparsifyDynamics(l)
        dict_s = [{f"s_{i}": value for i, value in enumerate(sigma[:, j])} for j in range(self.data.shape[0])]
        solution = diffeqsolve(self.term, self.ode_solver, t0=0, t1=float(self.t1), dt0=float(self.h),
                                y0=jnp.array(self.data[:, 0]), args=dict_s, saveat=self.saveat)
        yest = solution.ys.T
        diff_int = ((yest - self.data).flatten()) ** 2
        diff_int = diff_int.mean() / self.norm_L2
        diff_w = self.wasserstein(yest, self.data)
        diff_w = (diff_w - self.set_off_wasserstein) / self.norm_wasserstein

        diff = alpha * diff_int + (1 - alpha) * diff_w
        return diff.mean()

    def rms_sindy(self, space):
        l = space[1:]
        sigma = self.sparsifyDynamics(l)
        diff = ((self.data_dot - np.dot(sigma.T, self.lib.T)).flatten()) ** 2  # SINDy
        return diff.mean()

    def calc_norms(self):
        sigma = self.sparsifyDynamics(0) * 0
        dict_s = [{f"s_{i}": value for i, value in enumerate(sigma[:, j])} for j in range(self.data.shape[0])]
        solution = diffeqsolve(self.term, self.ode_solver, t0=0.0, t1=float(self.t1), dt0=float(self.h),
                                y0=jnp.array(self.data[:, 0]), args=dict_s, saveat=self.saveat)
        estimate = solution.ys.T
        norm_L2 = (((estimate - self.data).flatten()) ** 2).mean()
        norm_wasserstein = self.wasserstein(self.data, estimate)
        set_off_wasserstein = self.wasserstein(self.data, self.data)
        self.norm_L2 = norm_L2
        self.norm_wasserstein = norm_wasserstein
        self.set_off_wasserstein = set_off_wasserstein
        print("norms", self.norm_L2, self.norm_wasserstein)
    
    ######## Optimization #######
    # Optimize Information Criterion for optimal sparsity
    def optimize(self):
        self.calc_norms()
        print("working with " + self.error + " error")
        print("searching for optimal lambda")
        space = self.space
        trials = Trials()

        if self.IC == "BIC":
            print("->working with Bayesian Information Criterion")
            if self.error in ["mixed", "L2", "W"]:
                x_best = fmin(fn=self.BIC, space=space, algo=tpe.suggest, max_evals=self.evals, trials=trials)
            else:
                x_best = fmin(fn=self.BIC_sindy, space=space, algo=tpe.suggest, max_evals=self.evals, trials=trials)
        elif self.IC == "RMS":
            print("->working only with r.m.s. error")
            if self.error in ["mixed", "L2", "W"]:
                x_best = fmin(fn=self.rms, space=space, algo=tpe.suggest, max_evals=self.evals)
            else:
                x_best = fmin(fn=self.rms_sindy, space=space, algo=tpe.suggest, max_evals=self.evals)
        optimum = x_best

        print("->Best value observed at:")
        if self.multi:
            liste = [x_best["alpha"]]
            for i in range(self.data.shape[0]):
                print("lambda" + str(i) + "=", x_best['l' + str(i)], ", ")
                liste.append(x_best["l" + str(i)])
        else:
            print("lambda=", x_best['l'], ", ")
            liste = [x_best["alpha"]]
            liste.append(x_best['l'])
        print("alpha = ", x_best["alpha"])

        x_best = liste

        self.estimate, self.sigma_est, self.diff_sindy = self.calc_y_model(x_best[1:])
        self.p = np.count_nonzero(self.sigma_est.flatten())
        print("->number of parameters: p=", self.p)
        
        print("->saving parameters and estimated attractor")
        np.save("estimated_parameter", self.sigma)
        np.save("estimated_timeseries", self.estimate)
        resultat = list(self.poly_summe.subs(np.array([self.sigma, np.around(self.sigma_est[:, i], 3)]).T)
                        for i in range(self.data.shape[0]))
        for i in range(len(resultat)):
            print(resultat[i])

        errors = []
        self.error_init = np.copy(self.error)

        self.error = "sindy"
        print("rms_ls:", self.rms_sindy(x_best))
        errors.append(self.rms_sindy(x_best))

        self.error = "L2"
        print("rms_int:", self.rms(x_best) / self.norm_L2)
        errors.append(self.rms(x_best) / self.norm_L2)

        self.error = "W"
        print("rms_wasserstein:", (self.rms(x_best) - self.set_off_wasserstein) / self.norm_wasserstein)
        errors.append((self.rms(x_best) - self.set_off_wasserstein) / self.norm_wasserstein)

        self.error = self.error_init

        return optimum, trials.losses(), errors

    def plot_comparison(self):
        print("plotting comparison to ./plot.pdf")
        for i in range(self.data.shape[0]):
            plt.subplot(self.data.shape[0], 1, i + 1)
            plt.plot(self.estimate[i], label="estimate-" + str(self.error))
            plt.plot(self.data[i], label="original")
            plt.ylabel("$y_%01d$" % i)
            if i == 0:
                plt.legend(prop={'size': 8})
        plt.xlabel("Time t")
        plt.savefig("plot.pdf")

# --- Testing the rebel class with lorenz.npy ---
if __name__ == "__main__":
    # Load the data (ensure lorenz.npy is in the working directory)
    data = np.load("data/lorenz_data.npy")
    
    # Instantiate the rebel class with the data.
    # Adjust stepsize and evals if needed for faster testing.
    model = rebel(data=data, stepsize=0.01, evals=20, error="L2", order=3)
    
    # Preprocess the data (calculate derivative and set up the library)
    model.preprocess()
    
    # (Optional) Calculate norms prior to optimization if you wish to inspect them separately.
    model.calc_norms()
    
    # Run the optimization procedure to estimate model parameters.
    optimum, losses, errors = model.optimize()
    
    # Print the optimization results.
    print("Optimization optimum:", optimum)
    print("Losses during optimization:", losses)
    print("Errors:", errors)
    
    # Plot the comparison of estimated vs. original trajectories.
    model.plot_comparison()
    
    # Optionally, display the plot.
    plt.show()
