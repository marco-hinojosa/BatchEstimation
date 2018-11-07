function [J,g,H] = batch_cost(z,xfunc,yfunc,Afunc,Cfunc,x0,Q,R,u,tspan,m)
%% Cost Function for Batch Estimator of Nonlinear Systems
%
% Inputs:
%	xfunc - Function handle for state equation
%   yfunc - Function handle for sensor equation
%   Afunc - State Dynamics Jacobian
%   Cfunc - Measurement Equation Jacobian
%   x0 - Initial State Conditions
%   Q - Process Noise Covariance
%   R - Sensor Noise Covariance
%   u - Command Input
%   tspan - Time Span
%   m - Map Features
%
% Outputs:
%   J - Cost Function Scalar Value
%   g - Cost Function Gradient
%   H - Cost Function Hessian

n = length(tspan);
dt = tspan(2)-tspan(1);
xbar0 = [1;1;0]; % Initial Estimate Mean
P0 = 0.01*eye(3); % Initial Estimate Covariance
xp = reshape(z(:),length(x0),n); % Optimizing Variable, Estimated State

%% Create Initial Conditions
V = chol(R)*randn(length(yfunc),1); % Random Sensor Noise
for mm = 1:length(yfunc) % This for-loop creates the first observation
    y(mm,1) = yfunc{mm}(m,xp(:,1),0,dt); % Observations from Model y = h(x)
    yp(mm,1) = y(mm,1) + V(mm); % Noisy Sensor Data
end
%J = (xbar0 - x0)'*inv(P0)*(xbar0 - x0); % Non-Sliding Horizon (Batch)
J = 0;

%% Simulate Running Cost to Optimize
for kk = 1:(n-1)      
    % Simulate State and Noisy Sensor Dynamics
    V = chol(R)*randn(length(yfunc),1); % Sensor Noise
    for mm = 1:length(xfunc)
        fxp(mm,kk) = xfunc{mm}(m,xp(:,kk),u(kk),dt); % Propagate Each Decision Variable w/ Model x = f(x)
    end
    for mm = 1:length(yfunc)
        y(mm,kk) = yfunc{mm}(m,xp(:,kk),u(kk),dt); % Observations from Model y = h(x)
        yp(mm,kk) = y(mm,kk) + V(mm); % Noisy Sensor Data
    end
    
    % Create Cost Function for Nonlinear Dynamics and Observations
    J = J + 0.5*((xp(:,kk+1)-fxp(:,kk))'*inv(Q)*(xp(:,kk+1)-fxp(:,kk)) + (yp(:,kk)-y(:,kk))'*inv(R)*(yp(:,kk)-y(:,kk)));
    
    % Compute Jacobians
    for ii = 1:size(Afunc,1)
        for jj = 1:size(Afunc,2)
            A(ii,jj) = Afunc{ii,jj}(m,xp(:,kk),u(kk),dt);
        end
    end
    for ii = 1:size(Cfunc,1)
        for jj = 1:size(Cfunc,2)
            C(ii,jj) = Cfunc{ii,jj}(m,xp(:,kk),u(kk),dt);
        end
    end
    if kk == 1
        dJdx(kk,:) = -(xp(:,kk+1)-fxp(:,kk))'*inv(Q)*A - (yp(:,kk) - y(:,kk))'*inv(R)*C;
        ddJddx(:,:,kk) = A'*inv(Q)*A + C'*inv(R)*C;
    else
        dJdx(kk,:) = -(xp(:,kk+1)-fxp(:,kk))'*inv(Q)*A - (yp(:,kk) - y(:,kk))'*inv(R)*C + (xp(:,kk)-fxp(:,kk-1))'*inv(Q);
        ddJddx(:,:,kk) = A'*inv(Q)*A + C'*inv(R)*C + inv(Q);
    end
    uppdiag(:,:,kk) = -(A')*inv(Q); % Off-Diagonal Hessian Terms
    lowdiag(:,:,kk) = -inv(Q)*A; % Off-Diagonal Hessian Terms
end
dJdx(n,:) = (xp(:,n)-fxp(:,n-1))'*inv(Q);
ddJddx(:,:,n) = inv(Q);
q = num2cell(ddJddx,[1,2]); H = blkdiag(q{:});
g = reshape(dJdx,1,3*n);
for kk = 1:(n-1)
    H(3*kk-2:3*kk,3*kk+1:3*kk+3) = uppdiag(:,:,kk); % Upper Diagonal Terms
    H(3*kk+1:3*kk+3,3*kk-2:3*kk) = lowdiag(:,:,kk); % Lower Diagonal Terms
end  
end