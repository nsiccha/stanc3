functions {
  vector decay(real t, vector y, vector k) {
    return -k .* y;
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
  real err = ode_jacobian_error(decay, t0, y0, k);
}
