functions {
  vector residual(real t, vector yy, vector yp, vector k) {
    return yp + k .* yy;
  }
  // missing yp: a DAE Jacobian takes exactly the residual's arguments
  matrix residual_jac(real t, vector yy, vector k) {
    return rep_matrix(0, 2, 6);
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
  array[4] vector[2] y_hat = dae(residual, residual_jac, y0, -k .* y0, t0, ts, k);
}
