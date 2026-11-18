functions {
  vector decay(real t, vector y, vector k) {
    return -k .* y;
  }
  matrix decay_jac(real t, vector y, vector k) {
    return rep_matrix(0, 2, 4);
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
  vector[2] err = ode_jacobian_error(decay, decay_jac, t0, y0, k);
}
