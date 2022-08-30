
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
    
    real psi_mean;
    real log_psi_sd;
    
    real tau_mean;
    real log_tau_sd;
    
    
    real phi_mean;
    real log_phi_sd;
    
}

transformed parameters {
    vector[N_stations] mu = exp(psi);
    vector[N_stations] sigma = exp(tau) ./ mu;
    vector[N_stations] xi;
    real psi_sd = exp(log_psi_sd);
    real tau_sd = exp(log_tau_sd);
    real phi_sd = exp(log_phi_sd);
    
    for (i in 1:N_stations) {
        xi[i] = 1 / (1 + exp(-phi[i])) - 0.49;
    }
    
}


model {
    vector[N_obs] z = (y - mu[station]) ./ sigma[station];
    vector[N_stations] xi_inv = inv(xi);
    vector[N_stations] log_sigma = tau - psi;
    
    target += normal_lpdf(psi | psi_mean, psi_sd);
    target += normal_lpdf(tau | tau_mean, tau_sd);
    target += normal_lpdf(phi | phi_mean, phi_sd);
    
    target += std_normal_lpdf(psi_mean);
    target += std_normal_lpdf(log_psi_sd);
    
    target += std_normal_lpdf(tau_mean);
    target += std_normal_lpdf(log_tau_sd);
    
    target += std_normal_lpdf(phi_mean);
    target += std_normal_lpdf(log_phi_sd);
    
    for (i in 1:N_obs) {
        target += -log_sigma[station[i]] - (1 + xi_inv[station[i]]) * log(1 + xi[station[i]] * z[i]) - pow(1 + xi[station[i]] * z[i], -xi_inv[station[i]]);
    }
    
    

}

