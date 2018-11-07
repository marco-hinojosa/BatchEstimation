%% Batch Estimator for AA273: HW 6, Problem 3
% Marco Hinojosa
% 06181747

clearvars, clc

% Initial Conditions and Time Span
dt = .1; n = 50; t = 0:dt:(n-1)*dt; % Time Step, seconds
m1 = [0 0]'; m2 = [10 0]'; m3 = [10 10]'; m4 = [0 10]';
m = [m1 m2 m3 m4];
Q = 0.01*eye(3); R = .1*eye(8);
v = 1; u = sin(t);
x0 = [1;1;0];

% Define Dynamics and Observation Functions
xfunc = {@(m,x,u,dt)x(1) + v*cos(x(3))*dt; @(m,x,u,dt)x(2) + v*sin(x(3))*dt; @(m,x,u,dt)x(3) + u*dt};
yfunc = {@(m,x,u,dt)norm(m(:,1) - x(1:2),2);
         @(m,x,u,dt)norm(m(:,2) - x(1:2),2);
         @(m,x,u,dt)norm(m(:,3) - x(1:2),2);
         @(m,x,u,dt)norm(m(:,4) - x(1:2),2);
         @(m,x,u,dt)atan2(m(2,1)-x(2),m(1,1)-x(1)) - x(3);
         @(m,x,u,dt)atan2(m(2,2)-x(2),m(1,2)-x(1)) - x(3);
         @(m,x,u,dt)atan2(m(2,3)-x(2),m(1,3)-x(1)) - x(3);
         @(m,x,u,dt)atan2(m(2,4)-x(2),m(1,4)-x(1)) - x(3)};
     
% Define Gradients of f(x) and h(x)
Afunc = {@(m,mu,u,dt)1, @(m,mu,u,dt)0, @(m,mu,u,dt)-v*sin(mu(3))*dt; 
         @(m,mu,u,dt)0, @(m,mu,u,dt)1, @(m,mu,u,dt)v*cos(mu(3))*dt; 
         @(m,mu,u,dt)0, @(m,mu,u,dt)0, @(m,mu,u,dt)1};
Cfunc = {@(m,mu_pred,u,dt)(mu_pred(1)-m(1,1))/norm(mu_pred(1:2)-m(:,1),2), @(m,mu_pred,u,dt)(mu_pred(2)-m(2,1))/norm(mu_pred(1:2)-m(:,1),2), @(m,mu_pred,u,dt)0;
         @(m,mu_pred,u,dt)(mu_pred(1)-m(1,2))/norm(mu_pred(1:2)-m(:,2),2), @(m,mu_pred,u,dt)(mu_pred(2)-m(2,2))/norm(mu_pred(1:2)-m(:,2),2), @(m,mu_pred,u,dt)0;
         @(m,mu_pred,u,dt)(mu_pred(1)-m(1,3))/norm(mu_pred(1:2)-m(:,3),2), @(m,mu_pred,u,dt)(mu_pred(2)-m(2,3))/norm(mu_pred(1:2)-m(:,3),2), @(m,mu_pred,u,dt)0;
         @(m,mu_pred,u,dt)(mu_pred(1)-m(1,4))/norm(mu_pred(1:2)-m(:,4),2), @(m,mu_pred,u,dt)(mu_pred(2)-m(2,3))/norm(mu_pred(1:2)-m(:,4),2), @(m,mu_pred,u,dt)0;
         @(m,mu_pred,u,dt)(m(2,1)-mu_pred(2))/(norm(m(:,1)-mu_pred(1:2),2)^2), @(m,mu_pred,u,dt)-(m(1,1)-mu_pred(1))/(norm(m(:,1)-mu_pred(1:2),2)^2), @(m,mu_pred,u,dt)-1;
         @(m,mu_pred,u,dt)(m(2,2)-mu_pred(2))/(norm(m(:,2)-mu_pred(1:2),2)^2), @(m,mu_pred,u,dt)-(m(1,2)-mu_pred(1))/(norm(m(:,2)-mu_pred(1:2),2)^2), @(m,mu_pred,u,dt)-1;
         @(m,mu_pred,u,dt)(m(2,3)-mu_pred(2))/(norm(m(:,3)-mu_pred(1:2),2)^2), @(m,mu_pred,u,dt)-(m(1,3)-mu_pred(1))/(norm(m(:,3)-mu_pred(1:2),2)^2), @(m,mu_pred,u,dt)-1;
         @(m,mu_pred,u,dt)(m(2,4)-mu_pred(2))/(norm(m(:,4)-mu_pred(1:2),2)^2), @(m,mu_pred,u,dt)-(m(1,3)-mu_pred(1))/(norm(m(:,4)-mu_pred(1:2),2)^2), @(m,mu_pred,u,dt)-1};

     
%% Simulate Dynamics (Neglecting Process Noise)
xtrue(:,1) = x0;
for ii = 2:n      
    % Simulate State and Noisy Sensor Dynamics
    V = chol(R)*randn(length(yfunc),1); % Sensor Noise
    for mm = 1:length(xfunc)
        xtrue(mm,ii) = xfunc{mm}(m,xtrue(:,ii-1),u(ii-1),dt); % Propagate Dynamics (No Noise)
    end
    for mm = 1:length(yfunc)
        ytrue(mm,ii) = yfunc{mm}(m,xtrue(:,ii),u(ii-1),dt) + V(mm); % Make Observations (With Noise)
    end
end

%% Find Optimal Solution
z0 = reshape(xtrue(:),length(x0)*n,1); %z0(1:3) = [2; 2; 1];
func = @(z)batch_cost(z,xfunc,yfunc,Afunc,Cfunc,x0,Q,R,u,t,m);

% Using fminunc
%options = optimoptions('fminunc','Display','iter','StepTolerance',1.0000e-10);
options = optimoptions('fminunc','Display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','StepTolerance',1.0000e-18,'MaxIterations',2000);
[solution,Jval] = fminunc(func,z0,options);

% Using lsqnonlin
%options = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt');
%[solution,Jval] = lsqnonlin(func,z0,options);

x = reshape(solution(:),length(x0),n); % Independent (Optimizing) Variable
figure(1),hold on
subplot(3,1,1),plot(t,xtrue(1,:),t,x(1,:)),xlabel('Time (s)'),ylabel('p_x'), title('Batch Estimation Results')
subplot(3,1,2),plot(t,xtrue(2,:),t,x(2,:)),xlabel('Time (s)'),ylabel('p_y')
subplot(3,1,3),plot(t,xtrue(3,:),t,x(3,:)),xlabel('Time (s)'),ylabel('\theta')
legend('Truth','Estimate','Location','NorthWest')

