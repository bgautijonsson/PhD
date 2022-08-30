
data {
    int N_obs;
    int N_stations;
    
    vector[N_obs] y;
    vector[N_obs] year;
    array[N_obs] int station;
    vector[N_stations] log_min_y;
}


parameters {
    vector[N_stations] psi;
    vector[N_stations] tau;
    vector[N_stations] phi;
    vector[N_stations] gamma;
    
}

transformed parameters {
    vector[N_stations] mu = exp(psi);
    vector[N_stations] sigma = exp(tau) ./ mu;
    vector[N_stations] xi;
    vector[N_stations] delta;
    
    for (i in 1:N_stations) {
        xi[i] = 1 / (1 + exp(-phi[i])) - 0.5;
        delta[i] = 0.008 * 1 / (1 + exp(-gamma[i])) - 0.004;
    }
    
}


model {
    vector[N_obs] mu_i = mu_i[station] .* (1 + delta[station] .* year);
    vector[N_obs] z = (y - mu_i) ./ sigma[station];
    vector[N_stations] xi_inv = inv(xi);
    vector[N_stations] log_sigma = tau - psi;
    // vector[N_obs] log_z = log(1 + xi[station] .* z);
    
    target += normal_lpdf(psi | 1.4, 1);
    target += normal_lpdf(tau | -1.2, 1);
    target += std_normal_lpdf(phi);
    target += std_normal_lpdf(gamma);
    
    for (i in 1:N_obs) {
        target += -log(sigma[station[i]]) - (1 + xi_inv[station[i]]) * log(1 + xi[station[i]] * z[i]) - pow(1 + xi[station[i]] * z[i], -xi_inv[station[i]]);
    }
    
    // target += - N_obs * log_sigma - (1 + xi_inv) * sum(log_z) - sum(pow(1 + xi[station] .* z, -xi_inv));
    

}

