functions {
  // column j of y_lag is y(t - tau[j])
  vector feedback(real t, vector y, matrix y_lag, array[] real theta,
                  data array[] real x_r, data array[] int x_i) {
    return [-theta[1] * y[1], theta[1] * y[1] - theta[2] * y_lag[2, 1]]';
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
  array[nt, 1] real x_r;
  array[nt, 1] int x_i;
}
parameters {
  array[2] real<lower=0> theta;
  array[nt, 2] real<lower=0> theta_t;
  array[2] real<lower=0> biovar;
  array[2] real<lower=0> tlag;
  vector<lower=0>[1] tau;
  vector<lower=0>[2] y_hist;
}
transformed parameters {
  matrix[2, nt] x = pmx_solve_dde_rk45(feedback, 2, time, amt, rate, ii, evid,
                                       cmt, addl, ss, theta, tau, y_hist);
  matrix[2, nt] x_full = pmx_solve_dde_rk45(feedback, 2, time, amt, rate, ii,
                                            evid, cmt, addl, ss, theta_t, tau,
                                            y_hist, biovar, tlag, x_r, x_i,
                                            1e-8, 1e-8, 10000, 1e-6, 1e-6,
                                            100);
}
