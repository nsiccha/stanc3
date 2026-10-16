functions {
  vector delayed_decay(real t, vector y, matrix y_lag) {
    return -y_lag[ : , 1];
  }
}
data {
  vector[2] y0;
  real t0;
  array[4] real ts;
  vector[2] tau;
  array[4] vector[2] y;
}
parameters {
  vector<lower=0>[3] k;
  real<lower=0> sigma;
}
transformed parameters {
  array[4] vector[2] y_hat = dde_rk45(delayed_decay, y0, t0, ts);
}
model {
  for (t in 1 : 4) {
    y[t] ~ normal(y_hat[t], sigma);
  }
}
