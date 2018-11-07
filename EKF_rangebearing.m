%% EKF with Range and Bearing Measurements for AA273: HW 6, Problem 3
% Marco Hinojosa
% 06181747

clearvars, clc

dt = 0.1; n = 50; t = 0:dt:(n-1)*dt; % Time Step, seconds
m1 = [0 0]'; m2 = [10 0]'; m3 = [10 10]'; m4 = [0 10]';
m = [m1 m2 m3 m4];

Q = 0.00001*dt*eye(3); R = 0.1;
Q_m = zeros(8);
v = 1; u = sin(t);

mu0 = [1;1;0]; cov0 = 0.01*eye(3); x0 = [1;1;0];

xfunc = {@(m,x,u,dt)x(1) + v*cos(x(3))*dt; @(m,x,u,dt)x(2) + v*sin(x(3))*dt; @(m,x,u,dt)x(3) + u*dt};
yfunc = {@(m,x,u,dt)norm(m(:,1) - x(1:2),2);
         @(m,x,u,dt)norm(m(:,2) - x(1:2),2);
         @(m,x,u,dt)norm(m(:,3) - x(1:2),2);
         @(m,x,u,dt)norm(m(:,4) - x(1:2),2);
         @(m,x,u,dt)atan2(m(2,1)-x(2),m(1,1)-x(1)) - x(3);
         @(m,x,u,dt)atan2(m(2,2)-x(2),m(1,2)-x(1)) - x(3);
         @(m,x,u,dt)atan2(m(2,3)-x(2),m(1,3)-x(1)) - x(3);
         @(m,x,u,dt)atan2(m(2,4)-x(2),m(1,4)-x(1)) - x(3)};

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
     
[mu,cov,p,y] = ekf(xfunc,yfunc,x0,mu0,cov0,Afunc,Cfunc,Q,R,u,t,m,1);

figure,
subplot(3,1,1),plot(t,p(1,:),t,mu(1,:),'r--'),xlabel('Time (s)'),ylabel('p_x'), title('Extended Kalman Filter')
subplot(3,1,2),plot(t,p(2,:),t,mu(2,:),'r--'),xlabel('Time (s)'),ylabel('p_y')
subplot(3,1,3),plot(t,p(3,:),t,mu(3,:),'r--'),xlabel('Time (s)'),ylabel('\theta')
legend('Truth','Estimate','Location','NorthWest')
figure,plot(p(1,:),p(2,:),mu(1,:),mu(2,:),'r--')
title('EKF Trajectory'),xlabel('p_x'),ylabel('p_y'),legend('Truth','Estimate','Location','NorthWest')