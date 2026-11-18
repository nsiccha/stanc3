functions {
  // one compartment, first-order absorption, Michaelis-Menten elimination:
  // y = (gut, central), theta = (ka, V, Vmax, Km)
  vector oral(real t, vector y, array[] real theta, data array[] real x_r,
              data array[] int x_i) {
    real d = theta[4] * theta[2] + y[2];
    return [-theta[1] * y[1], theta[1] * y[1] - theta[3] * y[2] / d]';
  }
  // [df/dy | df/dtheta | df/dx_r]: 2 x (2 + 4 + n_x_r)
  matrix oral_jac_entries(vector y, array[] real theta, int n_x_r) {
    real d = theta[4] * theta[2] + y[2];
    matrix[2, 6 + n_x_r] jac = rep_matrix(0, 2, 6 + n_x_r);
    jac[1, 1] = -theta[1];
    jac[1, 3] = -y[1];
    jac[2, 1] = theta[1];
    jac[2, 2] = -theta[3] * theta[4] * theta[2] / square(d);
    jac[2, 3] = y[1];
    jac[2, 4] = theta[3] * y[2] * theta[4] / square(d);
    jac[2, 5] = -y[2] / d;
    jac[2, 6] = theta[3] * y[2] * theta[2] / square(d);
    return jac;
  }
  matrix oral_jac(real t, vector y, array[] real theta,
                  data array[] real x_r, data array[] int x_i) {
    return oral_jac_entries(y, theta, size(x_r));
  }
  // the same model with delayed elimination: y_lag[:, 1] is y(t - tau[1])
  vector delayed(real t, vector y, matrix y_lag, array[] real theta,
                 data array[] real x_r, data array[] int x_i) {
    real d = theta[4] * theta[2] + y_lag[2, 1];
    return [-theta[1] * y[1], theta[1] * y[1] - theta[3] * y_lag[2, 1] / d]';
  }
  // [df/dy | df/dy_lag[:, 1] | df/dtheta | df/dx_r]: 2 x (2 + 2 + 4 + size(x_r))
  matrix delayed_jac(real t, vector y, matrix y_lag, array[] real theta,
                     data array[] real x_r, data array[] int x_i) {
    real d = theta[4] * theta[2] + y_lag[2, 1];
    matrix[2, 8 + size(x_r)] jac = rep_matrix(0, 2, 8 + size(x_r));
    jac[1, 1] = -theta[1];
    jac[1, 5] = -y[1];
    jac[2, 1] = theta[1];
    jac[2, 4] = -theta[3] * theta[4] * theta[2] / square(d);
    jac[2, 5] = y[1];
    jac[2, 6] = theta[3] * y_lag[2, 1] * theta[4] / square(d);
    jac[2, 7] = -y_lag[2, 1] / d;
    jac[2, 8] = theta[3] * y_lag[2, 1] * theta[2] / square(d);
    return jac;
  }
}
data {
  int<lower=1, upper=4> solver; // pmx_solve_rk45, _bdf, _adams, _dde_rk45
  int<lower=0, upper=1> use_jac;
  // 0: theta only; 1: per-event theta and every optional argument
  int<lower=0, upper=1> full;
  int<lower=1> nt;
  array[nt] int<lower=1> cmt;
  array[nt] int evid;
  array[nt] int addl;
  array[nt] int ss;
  array[nt] real amt;
  array[nt] real time;
  array[nt] real rate;
  array[nt] real ii;
  array[nt, 1] real x_r;
  array[nt, 1] int x_i;
  vector[nt] c_obs;
}
parameters {
  array[4] real<lower=0> theta;
  array[2] real<lower=0> biovar;
  array[2] real<lower=0> tlag;
  vector<lower=0>[1] tau;
  vector<lower=0>[2] y_hist;
}
transformed parameters {
  // the same theta at every event
  array[nt, 4] real theta_t;
  for (i in 1 : nt) {
    theta_t[i] = theta;
  }
  matrix[2, nt] x;
  if (solver == 1) {
    if (full) {
      if (use_jac) {
        x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                           addl, ss, theta_t, biovar, tlag, x_r, x_i, 1e-10,
                           1e-10, 100000, 1e-8, 1e-8, 100);
      } else {
        x = pmx_solve_rk45(oral, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                           theta_t, biovar, tlag, x_r, x_i, 1e-10, 1e-10,
                           100000, 1e-8, 1e-8, 100);
      }
    } else {
      if (use_jac) {
        x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                           addl, ss, theta);
      } else {
        x = pmx_solve_rk45(oral, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                           theta);
      }
    }
  } else if (solver == 2) {
    if (full) {
      if (use_jac) {
        x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                          addl, ss, theta_t, biovar, tlag, x_r, x_i, 1e-10,
                          1e-10, 100000, 1e-8, 1e-8, 100);
      } else {
        x = pmx_solve_bdf(oral, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                          theta_t, biovar, tlag, x_r, x_i, 1e-10, 1e-10,
                          100000, 1e-8, 1e-8, 100);
      }
    } else {
      if (use_jac) {
        x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                          addl, ss, theta);
      } else {
        x = pmx_solve_bdf(oral, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                          theta);
      }
    }
  } else if (solver == 3) {
    if (full) {
      if (use_jac) {
        x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid,
                            cmt, addl, ss, theta_t, biovar, tlag, x_r, x_i,
                            1e-10, 1e-10, 100000, 1e-8, 1e-8, 100);
      } else {
        x = pmx_solve_adams(oral, 2, time, amt, rate, ii, evid, cmt, addl,
                            ss, theta_t, biovar, tlag, x_r, x_i, 1e-10,
                            1e-10, 100000, 1e-8, 1e-8, 100);
      }
    } else {
      if (use_jac) {
        x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid,
                            cmt, addl, ss, theta);
      } else {
        x = pmx_solve_adams(oral, 2, time, amt, rate, ii, evid, cmt, addl,
                            ss, theta);
      }
    }
  } else if (solver == 4) {
    if (full) {
      if (use_jac) {
        x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii,
              evid, cmt, addl, ss, theta_t, tau, y_hist, biovar, tlag, x_r,
              x_i, 1e-10, 1e-10, 100000, 1e-8, 1e-8, 100);
      } else {
        x = pmx_solve_dde_rk45(delayed, 2, time, amt, rate, ii, evid, cmt,
              addl, ss, theta_t, tau, y_hist, biovar, tlag, x_r, x_i, 1e-10,
              1e-10, 100000, 1e-8, 1e-8, 100);
      }
    } else {
      if (use_jac) {
        x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii,
              evid, cmt, addl, ss, theta, tau, y_hist);
      } else {
        x = pmx_solve_dde_rk45(delayed, 2, time, amt, rate, ii, evid, cmt,
              addl, ss, theta, tau, y_hist);
      }
    }
  }
}
model {
  theta ~ lognormal(0, 1);
  biovar ~ lognormal(-0.5, 0.5);
  tlag ~ lognormal(-1, 0.5);
  tau ~ lognormal(-1, 0.5);
  y_hist ~ lognormal(0, 1);
  for (i in 1 : nt) {
    if (evid[i] == 0) {
      c_obs[i] ~ normal(x[2, i] / theta[2], 0.5);
    }
  }
}

