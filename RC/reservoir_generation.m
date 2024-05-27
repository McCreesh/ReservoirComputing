function [J,J_y,J_r,J_rdot] = reservoir_generation(N,n,num_layers)
% Constructs all reservoirs in the network. If num_layers is more than 1 we
% are constructing a deep RC network with multiple reservoirs.

% Take the desired size and construct the reservoir along with the
% randomized input matrices for the network output and reference trajectory
% for each 

%% Initialize
J = zeros(N*n,N*n,num_layers); % Reservoir
J_y = zeros(N*n,n,num_layers); % Connection from system to reservoir
J_r = zeros(N*n,n,num_layers); % Connection from reference signal to reservoir
J_rdot = zeros(N*n,n,num_layers); % Connection from reference signal derivative to reservoir

%% Generation
 for i=1:num_layers

    % Reservoir
    J_hold = sprand(N*n,N*n,0.4);
    J_hold(J_hold~=0) = 4*nonzeros(J_hold) - 2; % Shift to have positive and negative values
    J_hold = full(J_hold);
    J_hold = (J_hold / max(real(eig(J_hold))))*0.95;

    J(:,:,i) = J_hold;

    % Input Vectors
    J_y(:,:,i) = 2*(rand(N*n,n)-0.5);
    J_r(:,:,i) = 2*(rand(N*n,n)-0.5);
    J_rdot(:,:,i) = 2*(rand(N*n,n)-0.5);
 end
end