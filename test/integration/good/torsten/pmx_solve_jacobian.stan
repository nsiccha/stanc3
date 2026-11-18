functions {
  // one compartment, first-order absorption, Michaelis-Menten elimination:
  // y = (gut, central), theta = (ka, V, Vmax, Km)
  vector oral(real t, vector y, array[] real theta, data array[] real x_r,
              data array[] int x_i) {
    real d = theta[4] * theta[2] + y[2];
    return [-theta[1] * y[1], theta[1] * y[1] - theta[3] * y[2] / d]';
  }
  // [df/dy | df/dtheta | df/dx_r]: 2 x (2 + 4 + n_x_r)
  matrix oral_jac_entries(vector y, array[] real theta, int n_x_r) {
    real d = theta[4] * theta[2] + y[2];
    matrix[2, 6 + n_x_r] jac = rep_matrix(0, 2, 6 + n_x_r);
    jac[1, 1] = -theta[1];
    jac[1, 3] = -y[1];
    jac[2, 1] = theta[1];
    jac[2, 2] = -theta[3] * theta[4] * theta[2] / square(d);
    jac[2, 3] = y[1];
    jac[2, 4] = theta[3] * y[2] * theta[4] / square(d);
    jac[2, 5] = -y[2] / d;
    jac[2, 6] = theta[3] * y[2] * theta[2] / square(d);
    return jac;
  }
  matrix oral_jac(real t, vector y, array[] real theta,
                  data array[] real x_r, data array[] int x_i) {
    return oral_jac_entries(y, theta, size(x_r));
  }
  // jac may accept autodiff arguments where f takes data-only ones
  matrix oral_jac_autodiff(real t, vector y, array[] real theta,
                           array[] real x_r, array[] int x_i) {
    return oral_jac_entries(y, theta, size(x_r));
  }
  // the same model with delayed elimination: y_lag[:, 1] is y(t - tau[1])
  vector delayed(real t, vector y, matrix y_lag, array[] real theta,
                 data array[] real x_r, data array[] int x_i) {
    real d = theta[4] * theta[2] + y_lag[2, 1];
    return [-theta[1] * y[1], theta[1] * y[1] - theta[3] * y_lag[2, 1] / d]';
  }
  // [df/dy | df/dy_lag[:, 1] | df/dtheta | df/dx_r]: 2 x (2 + 2 + 4 + size(x_r))
  matrix delayed_jac(real t, vector y, matrix y_lag, array[] real theta,
                     data array[] real x_r, data array[] int x_i) {
    real d = theta[4] * theta[2] + y_lag[2, 1];
    matrix[2, 8 + size(x_r)] jac = rep_matrix(0, 2, 8 + size(x_r));
    jac[1, 1] = -theta[1];
    jac[1, 5] = -y[1];
    jac[2, 1] = theta[1];
    jac[2, 4] = -theta[3] * theta[4] * theta[2] / square(d);
    jac[2, 5] = y[1];
    jac[2, 6] = theta[3] * y_lag[2, 1] * theta[4] / square(d);
    jac[2, 7] = -y_lag[2, 1] / d;
    jac[2, 8] = theta[3] * y_lag[2, 1] * theta[2] / square(d);
    return jac;
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
  array[nt, 1] real x_r;
  array[nt, 1] int x_i;
  array[4] real theta_data;
  vector[1] tau_data;
  vector[2] y_hist_data;
  real rel_tol;
  real abs_tol;
  int max_num_steps;
  real as_rel_tol;
  real as_abs_tol;
  int as_max_num_steps;
}
transformed data {
  matrix[2, nt] x_data = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate,
                                        ii, evid, cmt, addl, ss, theta_data);
  x_data = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii,
             evid, cmt, addl, ss, theta_data, tau_data, y_hist_data, x_r,
             x_i);
}
parameters {
  array[4] real<lower=0> theta;
  array[nt, 4] real<lower=0> theta_t;
  array[2] real<lower=0> biovar;
  array[nt, 2] real<lower=0> biovar_t;
  array[2] real<lower=0> tlag;
  array[nt, 2] real<lower=0> tlag_t;
  vector<lower=0>[1] tau;
  vector<lower=0>[2] y_hist;
}
transformed parameters {
  matrix[2, nt] x;
  // pmx_solve_rk45(f, jac, ...)
  // theta
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta);
  // theta, biovar
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta_t, biovar);
  // theta, biovar, tlag
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta, biovar_t, tlag);
  // theta, biovar, tlag, x_r
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta_t, biovar, tlag_t, x_r);
  // theta, biovar, tlag, x_r, x_i
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta, biovar_t, tlag, x_r, x_i);
  // theta + ODE tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta_t, rel_tol, abs_tol, max_num_steps);
  // theta, biovar + ODE tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta, biovar_t, rel_tol, abs_tol, max_num_steps);
  // theta, biovar, tlag + ODE tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta_t, biovar, tlag_t, rel_tol, abs_tol,
                     max_num_steps);
  // theta, biovar, tlag, x_r + ODE tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta, biovar_t, tlag, x_r, rel_tol, abs_tol,
                     max_num_steps);
  // theta, biovar, tlag, x_r, x_i + ODE tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta_t, biovar, tlag_t, x_r, x_i, rel_tol, abs_tol,
                     max_num_steps);
  // theta + ODE and algebra-solver tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta, rel_tol, abs_tol, max_num_steps, as_rel_tol,
                     as_abs_tol, as_max_num_steps);
  // theta, biovar + ODE and algebra-solver tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta_t, biovar, rel_tol, abs_tol, max_num_steps,
                     as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag + ODE and algebra-solver tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta, biovar_t, tlag, rel_tol, abs_tol,
                     max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag, x_r + ODE and algebra-solver tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta_t, biovar, tlag_t, x_r, rel_tol, abs_tol,
                     max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag, x_r, x_i + ODE and algebra-solver tolerances
  x = pmx_solve_rk45(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                     ss, theta, biovar_t, tlag, x_r, x_i, rel_tol, abs_tol,
                     max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // pmx_solve_bdf(f, jac, ...)
  // theta
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta);
  // theta, biovar
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta_t, biovar);
  // theta, biovar, tlag
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta, biovar_t, tlag);
  // theta, biovar, tlag, x_r
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta_t, biovar, tlag_t, x_r);
  // theta, biovar, tlag, x_r, x_i
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta, biovar_t, tlag, x_r, x_i);
  // theta + ODE tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta_t, rel_tol, abs_tol, max_num_steps);
  // theta, biovar + ODE tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta, biovar_t, rel_tol, abs_tol, max_num_steps);
  // theta, biovar, tlag + ODE tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta_t, biovar, tlag_t, rel_tol, abs_tol,
                    max_num_steps);
  // theta, biovar, tlag, x_r + ODE tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta, biovar_t, tlag, x_r, rel_tol, abs_tol,
                    max_num_steps);
  // theta, biovar, tlag, x_r, x_i + ODE tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta_t, biovar, tlag_t, x_r, x_i, rel_tol, abs_tol,
                    max_num_steps);
  // theta + ODE and algebra-solver tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta, rel_tol, abs_tol, max_num_steps, as_rel_tol,
                    as_abs_tol, as_max_num_steps);
  // theta, biovar + ODE and algebra-solver tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta_t, biovar, rel_tol, abs_tol, max_num_steps,
                    as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag + ODE and algebra-solver tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta, biovar_t, tlag, rel_tol, abs_tol,
                    max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag, x_r + ODE and algebra-solver tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta_t, biovar, tlag_t, x_r, rel_tol, abs_tol,
                    max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag, x_r, x_i + ODE and algebra-solver tolerances
  x = pmx_solve_bdf(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt, addl,
                    ss, theta, biovar_t, tlag, x_r, x_i, rel_tol, abs_tol,
                    max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // pmx_solve_adams(f, jac, ...)
  // theta
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta);
  // theta, biovar
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta_t, biovar);
  // theta, biovar, tlag
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta, biovar_t, tlag);
  // theta, biovar, tlag, x_r
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta_t, biovar, tlag_t, x_r);
  // theta, biovar, tlag, x_r, x_i
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta, biovar_t, tlag, x_r, x_i);
  // theta + ODE tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta_t, rel_tol, abs_tol, max_num_steps);
  // theta, biovar + ODE tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta, biovar_t, rel_tol, abs_tol,
                      max_num_steps);
  // theta, biovar, tlag + ODE tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta_t, biovar, tlag_t, rel_tol, abs_tol,
                      max_num_steps);
  // theta, biovar, tlag, x_r + ODE tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta, biovar_t, tlag, x_r, rel_tol, abs_tol,
                      max_num_steps);
  // theta, biovar, tlag, x_r, x_i + ODE tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta_t, biovar, tlag_t, x_r, x_i, rel_tol,
                      abs_tol, max_num_steps);
  // theta + ODE and algebra-solver tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta, rel_tol, abs_tol, max_num_steps,
                      as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar + ODE and algebra-solver tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta_t, biovar, rel_tol, abs_tol,
                      max_num_steps, as_rel_tol, as_abs_tol,
                      as_max_num_steps);
  // theta, biovar, tlag + ODE and algebra-solver tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta, biovar_t, tlag, rel_tol, abs_tol,
                      max_num_steps, as_rel_tol, as_abs_tol,
                      as_max_num_steps);
  // theta, biovar, tlag, x_r + ODE and algebra-solver tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta_t, biovar, tlag_t, x_r, rel_tol,
                      abs_tol, max_num_steps, as_rel_tol, as_abs_tol,
                      as_max_num_steps);
  // theta, biovar, tlag, x_r, x_i + ODE and algebra-solver tolerances
  x = pmx_solve_adams(oral, oral_jac, 2, time, amt, rate, ii, evid, cmt,
                      addl, ss, theta, biovar_t, tlag, x_r, x_i, rel_tol,
                      abs_tol, max_num_steps, as_rel_tol, as_abs_tol,
                      as_max_num_steps);
  // pmx_solve_dde_rk45(f, jac, ...)
  // theta
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta, tau, y_hist);
  // theta, biovar
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta_t, tau, y_hist, biovar);
  // theta, biovar, tlag
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta, tau, y_hist, biovar_t, tlag);
  // theta, biovar, tlag, x_r
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta_t, tau, y_hist, biovar, tlag_t, x_r);
  // theta, biovar, tlag, x_r, x_i
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta, tau, y_hist, biovar_t, tlag, x_r, x_i);
  // theta + ODE tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta_t, tau, y_hist, rel_tol, abs_tol, max_num_steps);
  // theta, biovar + ODE tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta, tau, y_hist, biovar_t, rel_tol, abs_tol,
        max_num_steps);
  // theta, biovar, tlag + ODE tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta_t, tau, y_hist, biovar, tlag_t, rel_tol,
        abs_tol, max_num_steps);
  // theta, biovar, tlag, x_r + ODE tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta, tau, y_hist, biovar_t, tlag, x_r, rel_tol,
        abs_tol, max_num_steps);
  // theta, biovar, tlag, x_r, x_i + ODE tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta_t, tau, y_hist, biovar, tlag_t, x_r, x_i,
        rel_tol, abs_tol, max_num_steps);
  // theta + ODE and algebra-solver tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta, tau, y_hist, rel_tol, abs_tol, max_num_steps,
        as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar + ODE and algebra-solver tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta_t, tau, y_hist, biovar, rel_tol, abs_tol,
        max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag + ODE and algebra-solver tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta, tau, y_hist, biovar_t, tlag, rel_tol, abs_tol,
        max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag, x_r + ODE and algebra-solver tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta_t, tau, y_hist, biovar, tlag_t, x_r, rel_tol,
        abs_tol, max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // theta, biovar, tlag, x_r, x_i + ODE and algebra-solver tolerances
  x = pmx_solve_dde_rk45(delayed, delayed_jac, 2, time, amt, rate, ii, evid,
        cmt, addl, ss, theta, tau, y_hist, biovar_t, tlag, x_r, x_i, rel_tol,
        abs_tol, max_num_steps, as_rel_tol, as_abs_tol, as_max_num_steps);
  // jac accepting autodiff x_r and x_i
  x = pmx_solve_bdf(oral, oral_jac_autodiff, 2, time, amt, rate, ii, evid,
                    cmt, addl, ss, theta_t, biovar, tlag, x_r, x_i);
  // the overloads without a Jacobian are unchanged
  x = pmx_solve_rk45(oral, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                     theta);
  x = pmx_solve_bdf(oral, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                    theta_t, biovar, tlag, x_r, x_i, rel_tol, abs_tol,
                    max_num_steps);
  x = pmx_solve_adams(oral, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                      theta, biovar);
  x = pmx_solve_dde_rk45(delayed, 2, time, amt, rate, ii, evid, cmt, addl,
        ss, theta, tau, y_hist);
}
model {
  
}

