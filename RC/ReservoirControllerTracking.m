classdef ReservoirControllerTracking < handle
    properties
        % Parameters
        % Network
           W               % n x n Network matrix
           m               % Network saturation bound
           Tau_net         % Network timescale 
           c               % Network input
           
        % Reservoir
           J               % N x N matrix of internal reservoir connections
           Jy              % N x n matrix of input connections from network
           Jr              % N X n matrix of input connections from reference trajectory
           Jrdot           % N x n matrix of input connections from reference trajectory derivative
           d               % additive constant for training
           dt              % Time step for numerical integration
           Tau_res         % Reservoir timescale

        % States
            x              % n x 1 vector of current network state 
            X              % N x 1 vector of current reservoir state

        % Other
            delta           % Number of steps ahead to look at reference signal
    end


    methods
        %% Constructor
        function obj = ReservoirControllerTracking(W,m,J,Jy,Jr,Jrdot,d,dt,Tau_net,Tau_res,delta)
            % Input Parameters
            obj.W = W;
            obj.m = m;
            obj.J = J;
            obj.Jy = Jy;
            obj.Jr = Jr;
            obj.Jrdot = Jrdot;
            obj.d = d;
            obj.dt = dt;
            obj.Tau_net = Tau_net;
            obj.Tau_res = Tau_res;
            obj.delta = delta;

            % Initialize Network States - Randomize if desired
            obj.x = blkdiag(ones(size(W/3,1),1))*m;
            obj.X = ones(size(J,1),1)*m;

            % Initialize Network Input Value - Randomize if desired
            obj.c = 0; 
        end


%% Learning Phase
        
   % Train the System  
function [X_driven,x_driven] = system_training(o,v)
    x_driven = 0*v; % Initialize network trajectory to get correct variable size

    % Propogate the Network
    for i=1:size(v,2)
        o.propogate_system_learning(v(:,i));
        x_driven(:,i) = o.x;
    end

    % Use propogated network to propogate the reservoir
    X_driven = zeros(length(o.J),size(v,2));
    for i=1:size(v,2)- o.delta
        o.propagate_reservoir_learning(x_driven(:,i),x_driven(:,i+o.delta),(x_driven(:,i+o.delta)-x_driven(:,i))/(o.delta*o.dt));
        X_driven(:,i) = o.X;
    end
    % Returns both system and reservoir trajectories from training input
end

%% Learning Auxiliary Functions
% RK Integrator and Linear-Threshold Activation Functions
        function propogate_system_learning(o,v)
            k1 = o.dt * o.del_l_s(o.x,v);
            k2 = o.dt * o.del_l_s(o.x+k1/2,v);
            k3 = o.dt * o.del_l_s(o.x+k2/2,v);
            k4 = o.dt * o.del_l_s(o.x+k3,v);
            o.x = o.x + (k1 + 2*k2 + 2*k3 + k4)/6;
        end
      
         % Learning System LTN
        function ds = del_l_s(o,x,v)
            threshold_term = o.W*x + v;
            idx_0 = threshold_term<=0;
            idx_m = threshold_term>=o.m;
            threshold_term(idx_0) = 0;
            threshold_term(idx_m) = o.m;
            ds = o.Tau_net * (-x + threshold_term);
        end

        % Learning Reservoir Runge-Kutta
        function propagate_reservoir_learning(o,y,ydelta,ydot)
            k1 = o.dt * o.del_l_r(o.X,y,ydelta,ydot);
            k2 = o.dt * o.del_l_r(o.X + k1/2,y,ydelta,ydot);
            k3 = o.dt * o.del_l_r(o.X + k2/2,y,ydelta,ydot);
            k4 = o.dt * o.del_l_r(o.X + k3, y,ydelta,ydot);
            o.X = o.X + (k1 + 2*k2 + 2*k3 + k4)/6;
        end

        % Learning Reservoir LTN
        function dr = del_l_r(o,X,y,ydelta,ydot)
            threshold_term = o.J*X + o.Jy*y + o.Jr*ydelta + o.Jrdot*ydot + o.d;
            idx_0 = threshold_term<=0;
            idx_m = threshold_term>=o.m;
            threshold_term(idx_0) = 0;
            threshold_term(idx_m) = o.m;
            dr = o.Tau_res * (-X + threshold_term);
        end

% Output Learning Functions
    function J_out = learn_output_vector(o,X_driven,vT)
        J_out_transpose = o.tikhonov_reg(X_driven',vT',0.5);
        J_out = J_out_transpose';
    end

% Tikhonov Regularization
    function X = tikhonov_reg(o,A,b,beta)
        % SVD of Matrix
        [U,S,V] = svd(A,'econ');
        btilde = U'*b;
        % Compute Solution
        Xtilde = (S'*S + beta^2*eye(length(S'*S)))\(S'*btilde);
        % Final Solution
        X = V*Xtilde;
    end

%% Reference Signal
% Varies depending on signal to track and size of network. Below is to
% track for single network with two nodes, and inhibiting one to zero
function r = reference(o,t)
   freq = 1/200;
    r1 =  sin(2*pi*freq*t)+2;
    r2 = 0;

    r = [r1; r2];
end

function rdot = reference_deriv(o,t)
    freq = 1/200;
    r1_deriv = 2*pi*freq*cos(2*pi*freq*t);
    r2_deriv = 0;
    rdot = [r1_deriv; r2_deriv];
end

%% Control Phase

% Controlling the system to the desired trajectory above, with initial
% control, output matrix and time horizon input.
function [t,sys_traject, control_vals] = controlled_system(o,J_out,c0,T)

    o.c = c0; 
    sys_traject = zeros(length(0:o.dt:T),length(o.x));
    control_vals = sys_traject;
    counter = 1;
    for i=0:o.dt:T
        o.propagate_system;
        o.propagate_reservoir(i);
        if i < 25
            o.c = 0.5*c0;
        else
        o.c = J_out*o.X;
        end
        sys_traject(counter,:) = o.x;
        control_vals(counter,:) = o.c;
        counter = counter+1;
    end
    t = 0:o.dt:T; % Time variable to return for plotting data
end

  %% Auxiliary Functions for Control

  % RK Integrators and LTN Activation Functions
        % Control Phase System RK Integrator
        function propagate_system(o)
            k1 = o.dt * o.del_s(o.x,o.c);
            k2 = o.dt * o.del_s(o.x+k1/2,o.c);
            k3 = o.dt * o.del_s(o.x+k2/2,o.c);
            k4 = o.dt * o.del_s(o.x+k3,o.c);
            o.x = o.x + (k1 + 2*k2 + 2*k3 + k4)/6;
        end

      
         % Control Phase System LTN
        function ds = del_s(o,x,v)
            threshold_term = o.W*x + v;
            idx_0 = threshold_term<=0;
            idx_m = threshold_term>=o.m;
            threshold_term(idx_0) = 0;
            threshold_term(idx_m) = o.m;
            ds = o.Tau_net * (-x + threshold_term);
        end

        % Control Phase Reservoir RK Integrator
        function propagate_reservoir(o,t)
            k1 = o.dt * o.del_r(o.X,t);
            k2 = o.dt * o.del_r(o.X+k1/2,t);
            k3 = o.dt * o.del_r(o.X+k2/2,t);
            k4 = o.dt * o.del_r(o.X+k3,t);
            o.X = o.X + (k1 + 2*k2 + 2*k3 + k4)/6; 
        end

        % Control Phase Reservoir LTN
        function dr = del_r(o,X,t)
            threshold_term = o.J*X + o.Jy*o.x + o.Jr*o.reference(t+o.delta*o.dt) + o.Jrdot*o.reference_deriv(t) + o.d;
            idx_0 = threshold_term<=0;
            idx_m = threshold_term>=o.m;
            threshold_term(idx_0) = 0;
            threshold_term(idx_m) = o.m;
            dr = o.Tau_res * (-o.X + threshold_term);
        end

    end

end