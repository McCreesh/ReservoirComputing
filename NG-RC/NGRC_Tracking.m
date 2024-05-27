classdef NGRC_Tracking < handle
    properties
        % Network parameters
           W              % n x n Network matrix
           m              % Network saturation bound
           Tau_net        % Network timescale 
           u              % Network input
           

        % Next-gen reservoir computer (NG-RC) parameters
          J_X             % NG-RC state output vector
          J_u             % NG-RC control output vector
          k               % Polynomial degree for nonlinear NG-RC vector
          c               % NG-RC scalar term

        % States
        x                 % n x 1 network state
        X_l               % N x 1 NG-RC linear components
        X_nl              % NG-RC nonlinear components - size depends on vector definition

        % Other parameters
        dt                % Numerical integration scalar
        K                 % Proportional error parameter
        beta              % Tikhonov regularization parameter
    end


    methods

      %% Constructor
      function obj = NGRC_Tracking(W,m,k,K,dt,Tau_net,c,beta)
            % Parameters
            obj.W = W;
            obj.m = m;
            obj.k = k;
            obj.dt = dt;
            obj.Tau_net = Tau_net;

            % Initialize Network States
            obj.x = ones(size(W,1),1)*m/2;
            obj.X_l = ones(size(W,1),1)*m;
            obj.X_nl = [];
            
            obj.J_X = [];
            obj.J_u = [];
            obj.c = c;
            obj.K = K;
            obj.u = [];
            obj.beta = beta;
        end
   
%% Learning Phase

function [X_total,y_driven] = system_training(o,v)
    y_driven = 0*v; % initialize system trajectory
    X_total = zeros(size(v,1)+1+size(o.W,1)+size(o.W,1)*(size(o.W,1)+1)/2,size(v,2));
    for i=1:size(v,2)
        o.X_l = o.x;
        o.X_nl = o.nonlinear_feature_vector(o.X_l);
        X_total(:,i) = [v(:,i); o.c; o.X_l; o.X_nl];
        % Move network forward
        o.propagate_system_learning(v(:,i));
        y_driven(:,i) = o.x;
    end
    J_total = o.learn_output_vector(X_total,y_driven);
    o.J_u = J_total(:,1:size(v,1));
    o.J_X = J_total(:,size(v,1)+1:end);

end

%% Reference Signal
% Varies depending on network size and signal to track. Below is to track
% for a network with two nodes, inhibiting one to zero and the other to r1.
function r = reference(o,t)
   freq = 1/200;
    r1 =  sin(2*pi*freq*t)+2;
    r = [r1;0]; % r2; 0; 0; r3];
end


%% Control Phase

function [t,sys_traject, control_vals] = controlled_system(o,T)

    sys_traject = zeros(length(0:o.dt:T),length(o.x));
    control_vals = sys_traject;
    counter = 1;
    o.x = ones(size(o.W,1),1);
    o.u = zeros(size(o.W,1),1);
    for i=0:o.dt:T
        o.propagate_system;
        o.X_l = o.x;
        o.X_nl = o.nonlinear_feature_vector(o.X_l);
        X = [o.c; o.X_l; o.X_nl];
        if i < 25 % Period without using learned control
            o.u = zeros(size(o.W,1),1);
        else % Start to apply control
        o.u = o.J_u\(o.reference(i+o.dt)-o.J_X*X + o.K*(o.x-o.reference(i)));
        end
        sys_traject(counter,:) = o.x;
        control_vals(counter,:) = o.u;
        counter = counter+1;
    end
    t = 0:o.dt:T; % Timepoints to return for plotting
end
  
    
%% Auxiliary Functions
% Construct feature vector
        function X_nl = nonlinear_feature_vector(o,X_l)
            X_nl = X_l;
            for i=1:o.k-1
                X_nl = o.unique_outer_product(X_nl,X_l,i);
            end
        end

% Outer product computation for feature vector
        function Y = unique_outer_product(o,X1,X2,index)
            Y = zeros((size(o.W,1)*(index))*(size(o.W,1)*(index)+1)/2,1);
            for i=1:length(X1)
                for j=i:length(X2)
                    Y(i+j-1) = X1(i)*X2(j);
                end
            end
        end
% Output Learning Functions
        function J_out = learn_output_vector(o,X_driven,vT)
            J_out_transpose = o.tikhonov_reg(X_driven',vT');
            J_out = J_out_transpose';
        end

% Tikhonov Regularization
        function X = tikhonov_reg(o,A,b)
            % SVD of Matrix
            [U,S,V] = svd(A,'econ');
            btilde = U'*b;
            % Compute Solution
            Xtilde = (S'*S + o.beta^2*eye(length(S'*S)))\(S'*btilde);
            % Final Solution
            X = V*Xtilde;
        end

%ODEs

        % Learning System Runge-Kutta
        function propagate_system_learning(o,v)
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

        % Control System Runge-Kutta
        function propagate_system(o)
            k1 = o.dt * o.del_s(o.x,o.u);
            k2 = o.dt * o.del_s(o.x+k1/2,o.u);
            k3 = o.dt * o.del_s(o.x+k2/2,o.u);
            k4 = o.dt * o.del_s(o.x+k3,o.u);
            o.x = o.x + (k1 + 2*k2 + 2*k3 + k4)/6;
        end

      
         % Control System LTN
        function ds = del_s(o,x,v)
            threshold_term = o.W*x + v;
            idx_0 = threshold_term<=0;
            idx_m = threshold_term>=o.m;
            threshold_term(idx_0) = 0;
            threshold_term(idx_m) = o.m;
            ds = o.Tau_net * (-x + threshold_term);
        end

 
    end % end methods
end % end class