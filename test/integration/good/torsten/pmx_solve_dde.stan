functions {
  // y = (A_gut, A_c, R); column j of y_lag is y(t - tau[j])
  vector turnover(real t, vector y, matrix y_lag, array[] real theta,
                  data array[] real x_r, data array[] int x_i) {
    real C = y[2] / theta[3];
    vector[3] dydt;
    dydt[1] = -theta[1] * y[1];
    dydt[2] = theta[1] * y[1] - theta[2] / theta[3] * y[2];
    dydt[3] = theta[4] * theta[5] * pow(theta[5] / y_lag[3, 1], theta[6])
              * (1 - theta[7] * C / (theta[8] + C)) - theta[4] * y[3];
    return dydt;
  }
}
data {
  int<lower=1> nt;
  array[nt] int<lower=1> cmt;
  array[nt] int evid;
  array[nt] int addl;
  array[nt] int ss;
  array[nt] real amt;
  array[nt] real time;
  array[nt] real rate;
  array[nt] real ii;
  array[nt, 2] real x_r;
  array[nt, 2] int x_i;
  vector[1] tau_data;
  vector[3] y_hist_data;
  array[8] real theta_data;
  real rel_tol;
  real abs_tol;
  int max_num_steps;
  real as_rel_tol;
  real as_abs_tol;
  int as_max_num_steps;
}
transformed data {
  matrix[3, nt] x_data = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii,
                                            evid, cmt, addl, ss, theta_data,
                                            tau_data, y_hist_data);
}
parameters {
  array[8] real<lower=0> theta;
  array[nt, 8] real<lower=0> theta_t;
  array[3] real<lower=0> biovar;
  array[nt, 3] real<lower=0> biovar_t;
  array[3] real<lower=0> tlag;
  array[nt, 3] real<lower=0> tlag_t;
  vector<lower=0>[1] tau;
  real<lower=0> R0;
}
transformed parameters {
  vector[3] y_hist = [0, 0, R0]';
  matrix[3, nt] x;
  // theta only
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta, tau, y_hist);
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta_t, tau, y_hist);
  // biovar
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta, tau, y_hist, biovar);
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta_t, tau, y_hist, biovar_t);
  // biovar, tlag
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta, tau, y_hist, biovar, tlag);
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta_t, tau, y_hist, biovar_t, tlag_t);
  // biovar, tlag, x_r
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta, tau, y_hist, biovar, tlag, x_r);
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta_t, tau, y_hist, biovar, tlag_t, x_r);
  // biovar, tlag, x_r, x_i
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta, tau, y_hist, biovar, tlag, x_r, x_i);
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta_t, tau, y_hist, biovar_t, tlag_t, x_r, x_i);
  // ODE tolerances
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta, tau, y_hist, rel_tol, abs_tol,
                         max_num_steps);
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta_t, tau, y_hist, biovar_t, tlag, x_r, x_i,
                         rel_tol, abs_tol, max_num_steps);
  // ODE and algebra-solver tolerances
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta, tau, y_hist, biovar, rel_tol, abs_tol,
                         max_num_steps, as_rel_tol, as_abs_tol,
                         as_max_num_steps);
  x = pmx_solve_dde_rk45(turnover, 3, time, amt, rate, ii, evid, cmt, addl,
                         ss, theta_t, tau, y_hist, biovar, tlag_t, x_r, x_i,
                         rel_tol, abs_tol, max_num_steps, as_rel_tol,
                         as_abs_tol, as_max_num_steps);
}
model {
}
