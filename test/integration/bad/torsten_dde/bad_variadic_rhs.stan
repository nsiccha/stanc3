// refused: the right-hand side has the fixed pmx_solve form (t, y, y_lag, theta, x_r, x_i), not the variadic form of dde_rk45
functions {
  vector rhs(real t, vector y, matrix y_lag, real k) {
    return -k * y_lag[ : , 1];
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
                                        addl, ss, tau, y_hist, theta[1]);
}
