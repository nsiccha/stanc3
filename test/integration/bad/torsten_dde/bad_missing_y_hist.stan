// refused: the history y_hist is required after tau
functions {
  vector rhs(real t, vector y, matrix y_lag, array[] real theta,
             data array[] real x_r, data array[] int x_i) {
    return -theta[1] * y_lag[ : , 1];
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
  array[1] real theta;
  vector[1] tau;
  vector[1] y_hist;
}
transformed parameters {
  matrix[1, nt] x = pmx_solve_dde_rk45(rhs, 1, time, amt, rate, ii, evid, cmt,
                                        addl, ss, theta, tau);
}
