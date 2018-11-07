function [mu,cov,x,y] = ekf(xfunc,yfunc,x0,mu0,cov0,Afunc,Cfunc,Q,R,u,tspan,m,flag)
% EKF: Extended Kalman Filter Algorithm.
%
% Inputs:
%	xfunc - Function handle for state equation
%   yfunc - Function handle for sensor equation
%   x0 - Initial State Conditions
%   mu0 - Initial Guess Estimate
%   cov0 - Initial Guess Covariance
%   Afunc - Linearized State Matrix (linearized about updated mean)
%   Cfunc - Linearized Measurement Matrix (linearized about predicted mean)
%   Q - Process Noise Covariance
%   R - Sensor Noise Covariance
%   u - Command Input
%   tspan - Time Span
%   m - Map Features
%   flag - Indicates Localization vs. Mapping Algorithm
%
% Outputs:
%	mu - Time history of estimated states
%	cov - Time history of estimated covariance
%   x - Time history of propagated noisy state
%   y - Time history of propagated noisy measurements

n = length(tspan);
dt = tspan(2)-tspan(1);
mu(:,1) = mu0;
cov = cov0;
x(:,1) = x0;

for ii = 2:n      
    if flag == 1 % Localization
        % Simulate Noisy State and Sensor Dynamics
        W = chol(Q)*randn(length(mu0),1);
        V = chol(R)*randn(length(yfunc),1);
        for mm = 1:length(xfunc)
            x(mm,ii) = xfunc{mm}(m,x(:,ii-1),u(ii-1),dt) + W(mm);
        end
        for mm = 1:length(yfunc)
            y(mm,ii) = yfunc{mm}(m,x(:,ii),u(ii-1),dt) + V(mm);
        end
        % Linearized System A-Jacobian
        for jj = 1:size(Afunc,1)
            for kk = 1:size(Afunc,2)
                A(jj,kk) = Afunc{jj,kk}(m,mu(:,ii-1),u(ii-1),dt);
            end
        end
        % Prediction Step
        for mm = 1:length(xfunc)
            mu_pred(mm,:) = xfunc{mm}(m,mu(:,ii-1),u(ii-1),dt);
        end
        for jj = 1:size(Cfunc,1)
            for kk = 1:size(Cfunc,2)
                C(jj,kk) = Cfunc{jj,kk}(m,mu_pred,u(ii-1),dt);
            end
        end
        cov_pred = A*cov*A' + Q;  
        % Compute Gain and Innovation
        Kt = cov_pred*C'/(C*cov_pred*C' + R*eye(size(C*cov_pred*C')));    
        for mm = 1:length(yfunc)
            y_hat(mm,:) = yfunc{mm}(m,mu_pred,u(ii-1),dt);
        end
        % Update Step and Store Values
        mu(:,ii) = mu_pred + Kt*(y(:,ii) - y_hat);
        cov = cov_pred - Kt*C*cov_pred;
    elseif flag == 2 % Mapping
        % Simulate Known State and Noisy Sensor Dynamics
        V = chol(R)*randn(length(yfunc),1);
        for mm = 1:length(xfunc)
            x(mm,ii) = xfunc{mm}(m,x(:,ii-1),u(ii-1),dt);
        end
        for mm = 1:length(yfunc)
            y(mm,ii) = yfunc{mm}(mu(2*mm-1:2*mm,ii-1),x(:,ii),u(ii-1),dt) + V(mm);
        end
        % Linearized System A-Jacobian
        for jj = 1:size(Afunc,1)
            for kk = 1:size(Afunc,2)
                A(jj,kk) = Afunc{jj,kk}(mu(:,ii-1),x(:,ii-1),u(ii-1),dt);
            end
        end
        % Prediction Step
        mu_pred = mu(:,ii-1);       
        for jj = 1:size(Cfunc,1)
            for kk = 1:size(Cfunc,2)
                C(jj,kk) = Cfunc{jj,kk}(mu_pred(2*jj-1:2*jj),x(:,ii-1),u(ii-1),dt);
            end
        end
        cov_pred = A*cov*A' + Q;        
        % Compute Gain and Innovation
        Kt = cov_pred*C'/(C*cov_pred*C' + R*eye(size(C*cov_pred*C')));    
        for mm = 1:length(yfunc)
            y_hat(mm,:) = yfunc{mm}(mu_pred(2*mm-1:2*mm),x(:,ii),u(ii-1),dt);
        end
        % Update Step and Store Values
        mu(:,ii) = mu_pred + Kt*(y(:,ii) - y_hat);
        cov = cov_pred - Kt*C*cov_pred;
    elseif flag == 3 % SLAM
        [mu_p,cov_p(:,:,ii),p,y,mu_p_pred(:,ii),cov_p_pred(:,:,ii)] = ekf_loc(xfunc,yfunc,p,mu_p,cov_p(:,:,ii-1),Afunc1,Cfunc1,Q1,R,u(ii-1),dt,m_set);   
        [mu_m,cov_m,x,y] = ekf_map(xfunc,yfunc2,mu_p,mu_m,cov_m,Afunc2,Cfunc2,Q2,R,u(ii-1),dt,[]);
    
        m(:,ii) = mu_m;
        mu_set = reshape(m(:,ii),2,4);
        mu_p_vec = [mu_p_vec mu_p];
        mu_m_vec = [mu_m_vec mu_m];
        p_vec = [p_vec p];
    end
end
end