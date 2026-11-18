functions {
  // ODE: two states, variadic arguments (vector k, data array[] real x, int n)
  vector rhs(real t, vector y, vector k, data array[] real x, int n) {
    vector[2] dydt;
    dydt[1] = -k[1] * y[1] + x[1] * n;
    dydt[2] = k[1] * y[1] - k[2] * square(y[2]);
    return dydt;
  }
  // [df/dy | df/dk | df/dx]: 2 x (2 + 3); the int n gets no column
  matrix rhs_jac(real t, vector y, vector k, data array[] real x, int n) {
    matrix[2, 5] jac = rep_matrix(0, 2, 5);
    jac[1, 1] = -k[1];
    jac[1, 3] = -y[1];
    jac[1, 5] = n;
    jac[2, 1] = k[1];
    jac[2, 2] = -2 * k[2] * y[2];
    jac[2, 3] = y[1];
    jac[2, 4] = -square(y[2]);
    return jac;
  }
  matrix rhs_jac_wrong(real t, vector y, vector k, data array[] real x,
                       int n) {
    matrix[2, 5] jac = rhs_jac(t, y, k, x, n);
    jac[2, 4] = square(y[2]);
    return jac;
  }
  // DAE residual
  vector residual(real t, vector yy, vector yp, vector k) {
    vector[2] res;
    res[1] = yp[1] + k[1] * yy[1] - 1;
    res[2] = yp[2] - k[1] * yy[1] + k[2] * square(yy[2]);
    return res;
  }
  // [dF/dyy | dF/dyp | dF/dk]: 2 x (2 * 2 + 2)
  matrix residual_jac(real t, vector yy, vector yp, vector k) {
    matrix[2, 6] jac = rep_matrix(0, 2, 6);
    jac[1, 1] = k[1];
    jac[1, 3] = 1;
    jac[1, 5] = yy[1];
    jac[2, 1] = -k[1];
    jac[2, 2] = 2 * k[2] * yy[2];
    jac[2, 4] = 1;
    jac[2, 5] = -yy[1];
    jac[2, 6] = square(yy[2]);
    return jac;
  }
  // DDE with one delay: y_lag[:, 1] is y(t - tau[1])
  vector delayed(real t, vector y, matrix y_lag, vector k) {
    vector[2] dydt;
    dydt[1] = -k[1] * y_lag[1, 1] + 1;
    dydt[2] = k[1] * y_lag[1, 1] - k[2] * y[2];
    return dydt;
  }
  // [df/dy | df/dy_lag[:, 1] | df/dk]: 2 x (2 + 2 + 2)
  matrix delayed_jac(real t, vector y, matrix y_lag, vector k) {
    matrix[2, 6] jac = rep_matrix(0, 2, 6);
    jac[1, 3] = -k[1];
    jac[1, 5] = -y_lag[1, 1];
    jac[2, 2] = -k[2];
    jac[2, 3] = k[1];
    jac[2, 5] = y_lag[1, 1];
    jac[2, 6] = -y[2];
    return jac;
  }
}
data {
  int<lower=1, upper=13> solver;
  int<lower=0, upper=1> use_jac;
  int<lower=1> T;
  real t0;
  array[T] real ts;
  array[1] real x;
  int n;
  array[T] vector[2] y_obs;
}
parameters {
  vector<lower=0>[2] k;
  vector<lower=0>[2] y0;
  vector<lower=0>[1] tau;
}
transformed parameters {
  vector[2] yp0 = [1 - k[1] * y0[1], k[1] * y0[1] - k[2] * square(y0[2])]';
  array[T] vector[2] y_hat;
  if (solver == 1) {
    if (use_jac) {
      y_hat = ode_rk45(rhs, rhs_jac, y0, t0, ts, k, x, n);
    } else {
      y_hat = ode_rk45(rhs, y0, t0, ts, k, x, n);
    }
  } else if (solver == 2) {
    if (use_jac) {
      y_hat = ode_ckrk(rhs, rhs_jac, y0, t0, ts, k, x, n);
    } else {
      y_hat = ode_ckrk(rhs, y0, t0, ts, k, x, n);
    }
  } else if (solver == 3) {
    if (use_jac) {
      y_hat = ode_adams(rhs, rhs_jac, y0, t0, ts, k, x, n);
    } else {
      y_hat = ode_adams(rhs, y0, t0, ts, k, x, n);
    }
  } else if (solver == 4) {
    if (use_jac) {
      y_hat = ode_bdf(rhs, rhs_jac, y0, t0, ts, k, x, n);
    } else {
      y_hat = ode_bdf(rhs, y0, t0, ts, k, x, n);
    }
  } else if (solver == 5) {
    if (use_jac) {
      y_hat = ode_rk45_tol(rhs, rhs_jac, y0, t0, ts, 1e-8, 1e-8, 100000, k,
                           x, n);
    } else {
      y_hat = ode_rk45_tol(rhs, y0, t0, ts, 1e-8, 1e-8, 100000, k, x, n);
    }
  } else if (solver == 6) {
    if (use_jac) {
      y_hat = ode_ckrk_tol(rhs, rhs_jac, y0, t0, ts, 1e-8, 1e-8, 100000, k,
                           x, n);
    } else {
      y_hat = ode_ckrk_tol(rhs, y0, t0, ts, 1e-8, 1e-8, 100000, k, x, n);
    }
  } else if (solver == 7) {
    if (use_jac) {
      y_hat = ode_adams_tol(rhs, rhs_jac, y0, t0, ts, 1e-8, 1e-8, 100000, k,
                            x, n);
    } else {
      y_hat = ode_adams_tol(rhs, y0, t0, ts, 1e-8, 1e-8, 100000, k, x, n);
    }
  } else if (solver == 8) {
    if (use_jac) {
      y_hat = ode_bdf_tol(rhs, rhs_jac, y0, t0, ts, 1e-8, 1e-8, 100000, k, x,
                          n);
    } else {
      y_hat = ode_bdf_tol(rhs, y0, t0, ts, 1e-8, 1e-8, 100000, k, x, n);
    }
  } else if (solver == 9) {
    if (use_jac) {
      y_hat = ode_adjoint_tol_ctl(rhs, rhs_jac, y0, t0, ts, 1e-8,
                                  rep_vector(1e-8, 2), 1e-8,
                                  rep_vector(1e-8, 2), 1e-8, 1e-8, 100000,
                                  150, 1, 2, 2, k, x, n);
    } else {
      y_hat = ode_adjoint_tol_ctl(rhs, y0, t0, ts, 1e-8, rep_vector(1e-8, 2),
                                  1e-8, rep_vector(1e-8, 2), 1e-8, 1e-8,
                                  100000, 150, 1, 2, 2, k, x, n);
    }
  } else if (solver == 10) {
    if (use_jac) {
      y_hat = dae(residual, residual_jac, y0, yp0, t0, ts, k);
    } else {
      y_hat = dae(residual, y0, yp0, t0, ts, k);
    }
  } else if (solver == 11) {
    if (use_jac) {
      y_hat = dae_tol(residual, residual_jac, y0, yp0, t0, ts, 1e-8, 1e-8,
                      100000, k);
    } else {
      y_hat = dae_tol(residual, y0, yp0, t0, ts, 1e-8, 1e-8, 100000, k);
    }
  } else if (solver == 12) {
    if (use_jac) {
      y_hat = dde_rk45(delayed, delayed_jac, y0, t0, ts, tau, k);
    } else {
      y_hat = dde_rk45(delayed, y0, t0, ts, tau, k);
    }
  } else {
    if (use_jac) {
      y_hat = dde_rk45_tol(delayed, delayed_jac, y0, t0, ts, tau, 1e-8, 1e-8,
                           100000, k);
    } else {
      y_hat = dde_rk45_tol(delayed, y0, t0, ts, tau, 1e-8, 1e-8, 100000, k);
    }
  }
}
model {
  k ~ lognormal(0, 1);
  y0 ~ lognormal(0, 1);
  tau ~ lognormal(-1, 0.5);
  for (t in 1 : T) {
    y_obs[t] ~ normal(y_hat[t], 0.1);
  }
}
generated quantities {
  // largest difference from the autodiff Jacobian, relative to max(1, |entry|)
  real ode_error = ode_jacobian_error(rhs, rhs_jac, ts[1], y0, k, x, n);
  real ode_error_wrong = ode_jacobian_error(rhs, rhs_jac_wrong, ts[1], y0, k,
                                            x, n);
  real dae_error = dae_jacobian_error(residual, residual_jac, ts[1], y0, yp0,
                                      k);
  real dde_error = dde_jacobian_error(delayed, delayed_jac, ts[1], y0,
                                      rep_matrix(y0, 1), k);
}
