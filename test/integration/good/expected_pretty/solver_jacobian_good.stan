/* $ stanc --auto-format ../solver_jacobian_good.stan : [exit 0] */
functions {
  // ODE: dy/dt = -k .* y + u, with the Jacobian [df/dy | df/dk | df/du]
  vector decay(real t, vector y, vector k, real u) {
    return -k .* y + u;
  }
  matrix decay_jac(real t, vector y, vector k, real u) {
    int N = num_elements(y);
    return append_col(append_col(diag_matrix(-k), diag_matrix(-y)),
                      rep_vector(1, N));
  }
  // data-only and integer arguments get no Jacobian columns that are read
  vector decay_data(real t, vector y, data array[] real x, int n) {
    return -x[1] * y * n;
  }
  matrix decay_data_jac(real t, vector y, data array[] real x, int n) {
    return -x[1] * n * identity_matrix(num_elements(y));
  }
  // jac may accept autodiff arguments where f takes data-only ones
  matrix decay_data_jac_autodiff(real t, vector y, array[] real x, int n) {
    return -x[1] * n * identity_matrix(num_elements(y));
  }
  // overloaded Jacobian: the overload matching f's arguments is used
  matrix decay_data_jac(real t, vector y, vector k, real u) {
    return decay_jac(t, y, k, u);
  }
  // DAE residual F(t, yy, yp, k) = yp + k .* yy
  vector residual(real t, vector yy, vector yp, vector k) {
    return yp + k .* yy;
  }
  matrix residual_jac(real t, vector yy, vector yp, vector k) {
    int N = num_elements(yy);
    return append_col(append_col(diag_matrix(k), identity_matrix(N)),
                      diag_matrix(yy));
  }
  // DDE: dy/dt = -k .* y_lag[:, 1]
  vector delayed(real t, vector y, matrix y_lag, vector k) {
    return -k .* y_lag[ : , 1];
  }
  matrix delayed_jac(real t, vector y, matrix y_lag, vector k) {
    int N = num_elements(y);
    return append_col(append_col(rep_matrix(0, N, N), diag_matrix(-k)),
                      diag_matrix(-y_lag[ : , 1]));
  }
}
data {
  int<lower=1> T;
  int<lower=1> N;
  vector[N] y0;
  real t0;
  array[T] real ts;
  vector<lower=0>[1] tau_data;
  array[1] real x;
  int n;
  vector[N] k_data;
}
transformed data {
  array[T] vector[N] y_td = ode_rk45(decay, decay_jac, y0, t0, ts, k_data,
                                     1.0);
  array[T] vector[N] y_td_data = ode_bdf(decay_data, decay_data_jac, y0, t0,
                                         ts, x, n);
  real err_td = ode_jacobian_error(decay, decay_jac, t0, y0, k_data, 1);
  real err_td_data = ode_jacobian_error(decay_data, decay_data_jac, t0, y0,
                       x, n);
}
parameters {
  vector<lower=0>[N] k;
  real u;
  vector[N] y0_var;
  vector<lower=0>[1] tau;
  real<lower=0> sigma;
}
transformed parameters {
  array[T] vector[N] y_hat;
  y_hat = ode_rk45(decay, decay_jac, y0_var, t0, ts, k, u);
  y_hat = ode_ckrk(decay, decay_data_jac, y0, t0, ts, k, u);
  y_hat = ode_adams(decay, decay_jac, y0_var, t0, ts, k, 1);
  y_hat = ode_bdf(decay, decay_jac, y0, t0, ts, k_data, u);
  y_hat = ode_bdf(decay_data, decay_data_jac_autodiff, y0_var, t0, ts, x, n);
  y_hat = ode_rk45_tol(decay, decay_jac, y0_var, t0, ts, 1e-6, 1e-6, 1000, k,
                       u);
  y_hat = ode_ckrk_tol(decay, decay_jac, y0_var, t0, ts, 1e-6, 1e-6, 1000, k,
                       u);
  y_hat = ode_adams_tol(decay, decay_jac, y0_var, t0, ts, 1e-6, 1e-6, 1000,
                        k, u);
  y_hat = ode_bdf_tol(decay, decay_jac, y0_var, t0, ts, 1e-6, 1e-6, 1000, k,
                      u);
  y_hat = ode_adjoint_tol_ctl(decay, decay_jac, y0_var, t0, ts, 1e-6,
            rep_vector(1e-6, N), 1e-6, rep_vector(1e-6, N), 1e-6, 1e-6, 1000,
            150, 1, 2, 2, k, u);
  y_hat = dae(residual, residual_jac, y0_var, -k .* y0_var, t0, ts, k);
  y_hat = dae_tol(residual, residual_jac, y0_var, -k .* y0_var, t0, ts, 1e-6,
                  1e-6, 1000, k);
  y_hat = dde_rk45(delayed, delayed_jac, y0_var, t0, ts, tau, k);
  y_hat = dde_rk45_tol(delayed, delayed_jac, y0_var, t0, ts, tau_data, 1e-6,
                       1e-6, 1000, k);
  // the solvers without a Jacobian are unchanged
  y_hat = ode_rk45(decay, y0_var, t0, ts, k, u);
  y_hat = dae(residual, y0_var, -k .* y0_var, t0, ts, k);
  y_hat = dde_rk45(delayed, y0_var, t0, ts, tau, k);
  real err = ode_jacobian_error(decay, decay_jac, t0, y0_var, k, u);
  real err_dae = dae_jacobian_error(residual, residual_jac, t0, y0_var,
                   -k .* y0_var, k);
  real err_dde = dde_jacobian_error(delayed, delayed_jac, t0, y0_var,
                   rep_matrix(y0_var, 1), k);
}
model {
  for (t in 1 : T) {
    y0 ~ normal(y_hat[t], sigma);
  }
}
generated quantities {
  array[T] vector[N] y_gq = ode_rk45(decay, decay_jac, y0_var, t0, ts, k, u);
  real err_gq = ode_jacobian_error(decay, decay_jac, ts[1], y_gq[1], k, u);
}

