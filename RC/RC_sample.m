%% Sample Code for Tracking RC Controller

close all

%% Base Parameters
% Basic parameters for Reservoir Computer

m = 10; % Threshold
dt = .02; % time step
Tau_res = 1.4; % Reservoir responsiveness (time constant)
delta = 1; % Number of steps to look ahead in reference tracking

% Times and samples for training and control
T_l = 1000;                         % learning phase length
T_t = 100;                          % transient phase length
T_p = 1000;                          % prediction phase length
n_train = T_l/dt;                   % Number of training samples
n_trans = T_t/dt;                   % Number of transient samples
n_pred = T_p/dt;                    % Number of prediction samples
t_train = (1:n_train) + n_trans;    % Index of training samples
n_samples = n_trans + n_train + n_pred;     % Total number of samples



%% Generate Network for Tracking
% Using a single E-I pair for example.
% This is W1 in Selective Inhibition Example in "Control of
% Linear-Threshold Brain Networks Through Reservoir Computing"
W = [0.0112 -0.9903;
    0.4101 -0.5115];

tau_net = 1/4*eye(2);


%% Generate Reservoir 
% This is done randomly with specified size.
N = 50; % Reservoir scaling
n = 2; % Network size
num_layers = 1; % For if using a deep RC controller

% [J,Jy,Jr,Jrdot] = reservoir_generation(N,n,num_layers);

% Note that performance does depend on the reservoir, and not all generated
% ones will give satisfactory performance.

d = 0.1; % Training constant

%% Initialize Reservoir Controller

R = ReservoirControllerTracking(W,m,J,Jy,Jr,Jrdot,d,dt,tau_net,Tau_res,delta);

%% Train Reservoir Controller
% Use an arbitrarily constructed oscillating training signal, but could try
% almost anything, but performance can differ.

index = 1:n_samples+delta;
constant = sin(2*index/1000);
v = constant.*[3*ones(1,n_samples+delta); -2*ones(1,n_samples+delta)]; 

[X_driven,x_driven] = R.system_training(v); % The reservoir and system states driven by training input
X_drivenT = X_driven(:,t_train); % discard initial T_t seconds of driven reservoir states
vT = v(:,t_train,1); % discard initial T_t seconds of training data

% Learn the RC output vector
J_out = R.learn_output_vector(X_drivenT,vT);


%% Use the Reservoir Controller to Track the Reference Signal

% Initialize the network input - Randomize as desired
c = m/2*ones(n,1); 

[t,system_trajectory, control_trajectory] = R.controlled_system(J_out,c,750);

%% Plot
% Illustration of success of RC in tracking the reference signal.
plot(t,system_trajectory(:,1),'-b','LineWidth',1.5);
hold on
plot(t,system_trajectory(:,2),'g','LineWidth',1.5);
plot(t,sin(2*pi*1/200*t)+2,'--r','LineWidth',1);
plot(t,0*t,'--k','LineWidth',1)
legend('Excitatory Node','Inhibitory Node','Excitatory Reference','Inhibitory Reference')
xlabel('Time (s)')
ylabel('Firing Rate (Hz)')
title('Reference Tracking for Middle Layer')
