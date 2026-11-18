functions {
  vector decay(real t, vector y, vector k) {
    return -k .* y;
  }
  matrix decay_jac_rng(real t, vector y, vector k) {
    return rep_matrix(normal_rng(0, 1), 2, 4);
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
  array[4] vector[2] y_hat = ode_ckrk(decay, decay_jac_rng, y0, t0, ts, k);
}
