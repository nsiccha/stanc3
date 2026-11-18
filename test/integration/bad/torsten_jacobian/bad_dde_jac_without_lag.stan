functions {
  vector delayed(real t, vector y, matrix y_lag, array[] real theta,
                 data array[] real x_r, data array[] int x_i) {
    return -theta[1] * y_lag[ : , 1];
  }
  matrix delayed_jac(real t, vector y, array[] real theta,
                     data array[] real x_r, data array[] int x_i) {
    return [[0, 0, -theta[1]]];
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
}
parameters {
  array[2] real<lower=0> theta;
  array[nt, 2] real<lower=0> theta_t;
  vector<lower=0>[1] tau;
  vector<lower=0>[1] y_hist;
}
transformed parameters {
  matrix[1, nt] x = pmx_solve_dde_rk45(delayed, delayed_jac, 1, time, amt, rate, ii, evid, cmt, addl, ss, theta, tau, y_hist);
}
