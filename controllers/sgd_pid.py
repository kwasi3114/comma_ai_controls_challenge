from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    SGD-based adaptive PID controller
    """
    def __init__(self):
        self.cost_history = []
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.theta = np.array([self.p, self.i, self.d])
        self.learning_rate = 0.001
        self.error_integral = 0
        self.prev_error = 0
        self.del_t = 0.1
        self.accel_cost_multipler = 50.0
        self.target_lataccel_history = []
        self.current_lataccel_history = []
        self.rollout_data = []  # buffer for full trajectory data

    def reset(self):
        """Reset state and history for fresh rollout"""
        self.target_lataccel_history = []
        self.current_lataccel_history = []
        self.error_integral = 0
        self.prev_error = 0

    def get_control(self, theta, error, error_integral, error_diff):
        """Compute PID control signal"""
        p, i, d = theta
        return p * error + i * error_integral + d * error_diff

    def calculate_cost(self):
        target = self.target_lataccel_history
        current = self.current_lataccel_history
        diff = np.subtract(target, current)

        lat_accel_cost = np.mean(diff**2) * 100

        if len(target) < 2:
            return lat_accel_cost * self.accel_cost_multipler

        jerk_cost = np.mean((np.diff(current) / self.del_t) ** 2) * 100
        total_cost = (lat_accel_cost * self.accel_cost_multipler) + jerk_cost
        return total_cost

    def compute_grad(self, data):
        grad = np.zeros_like(self.theta)
        epsilon = 1e-4

        # Save original parameters
        original_theta = np.copy(self.theta)

        # Compute baseline cost using current theta
        self.reset()
        for target_lataccel, current_lataccel, state, future_plan in data:
            self.update_internal(target_lataccel, current_lataccel, state)
        baseline_cost = self.calculate_cost()

        # Compute gradient via finite differences
        for i in range(len(self.theta)):
            theta_perturbed = np.copy(original_theta)
            theta_perturbed[i] += epsilon
            self.theta = theta_perturbed

            self.reset()
            for target_lataccel, current_lataccel, state, future_plan in data:
                self.update_internal(target_lataccel, current_lataccel, state)

            perturbed_cost = self.calculate_cost()
            grad[i] = (perturbed_cost - baseline_cost) / epsilon

        self.theta = original_theta
        return grad

    def update_internal(self, target_lataccel, current_lataccel, state):
        """Run a single update step without gradient descent (for simulation)"""
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # Simulate control effect — just log the histories
        self.target_lataccel_history.append(target_lataccel)
        self.current_lataccel_history.append(current_lataccel)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """Main controller update loop"""
        # Normal PID control step
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # Compute control
        u = self.get_control(self.theta, error, self.error_integral, error_diff)

        # Record data for full rollout
        self.target_lataccel_history.append(target_lataccel)
        self.current_lataccel_history.append(current_lataccel)
        self.rollout_data.append((target_lataccel, current_lataccel, state, future_plan))

        # Update PID gains using gradient descent once we have enough data
        if len(self.rollout_data) >= 10:
            grad = self.compute_grad(self.rollout_data)
            self.theta -= self.learning_rate * grad
            self.p, self.i, self.d = self.theta
            self.rollout_data.clear()
            self.reset()

        return u
