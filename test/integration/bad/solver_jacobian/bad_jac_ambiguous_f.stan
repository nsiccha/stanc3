functions {
  // two overloads of f are equally good for the call below; jac fits both
  vector decay(real t, vector y, vector k, int a, real b) {
    return -k .* y;
  }
  vector decay(real t, vector y, vector k, real a, int b) {
    return -k .* y;
  }
  matrix decay_jac(real t, vector y, vector k, int a, real b) {
    return rep_matrix(0, 2, 5);
  }
  matrix decay_jac(real t, vector y, vector k, real a, int b) {
    return rep_matrix(0, 2, 5);
  }
}
data {
  vector[2] y0;
  real t0;
  array[4] real ts;
}
parameters {
  vector<lower=0>[2] k;
}
transformed parameters {
  array[4] vector[2] y_hat = ode_rk45(decay, decay_jac, y0, t0, ts, k, 0, 0);
}
