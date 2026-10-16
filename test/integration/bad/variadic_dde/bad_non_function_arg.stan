data {
  vector[2] y0;
  real t0;
  array[4] real ts;
  vector[2] tau;
  real g;
  array[4] vector[2] y;
}
parameters {
  vector<lower=0>[3] k;
  real<lower=0> sigma;
}
transformed parameters {
  array[4] vector[2] y_hat = dde_rk45(g, y0, t0, ts, tau, k);
}
model {
  for (t in 1 : 4) {
    y[t] ~ normal(y_hat[t], sigma);
  }
}
