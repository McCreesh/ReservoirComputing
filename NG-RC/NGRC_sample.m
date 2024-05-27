%% Sample Code for Tracking NG-RC Controller

close all

%% Basic Parameters

m = 10; % LTN Threshold
dt = 0.02; % time step

%% Generate Network for Tracking
% Using a single E-I pair for example.
% This is W1 in Selective Inhibition Example in "Control of
% Linear-Threshold Brain Networks Through Reservoir Computing"
W = [0.0112 -0.9903;
    0.4101 -0.5115];

tau_net = 1/4;


%% NG-RC Parameters

k = 2; % Quadratic polynomial nonlinear vector (Class file needs adjusted if changed)
K = -5; % NG-RC Proportional Control Value - determined experimentally
c= 0.5; % NG-RC feature vector scalar term
beta = 0.5; % Tikhonov regularization parameter
%% Initialize Reservoir Controller

R = NGRC_Tracking(W,m,k,K,dt,tau_net,c,beta);

%% Train NG-RC Controller

n_samples = 500;
 % v_normal = normrnd(0,0.2,[n,n_samples]); % This has impact on performance,
% may need to run multiple times to find a "good" one.

[X_driven, y_driven] = R.system_training(NGRC_top.training(:,1:n_samples));

% To see error - comment out if not desired
% hold_val = [R.J_u R.J_X]*X_driven;
% training_error = rmse(y_driven,hold_val) 

%% Use the NG-RC Controller to Track the Reference Signal

T = 500;
[t,sys_traject, control_vals] = R.controlled_system(T);
%% Plot
% Illustration of success of RC in tracking the reference signal.
figure
plot(t,sys_traject(:,1),'-b','LineWidth',1.5);
hold on
plot(t,sys_traject(:,2),'g','LineWidth',1.5);
plot(t,sin(2*pi*1/200*t)+2,'--r','LineWidth',1);
plot(t,0*t,'--k','LineWidth',1)
legend('Excitatory Node','Inhibitory Node','Excitatory Reference','Inhibitory Reference')
xlabel('Time (s)')
ylabel('Firing Rate (Hz)')
title('NG-RC Reference Tracking')

