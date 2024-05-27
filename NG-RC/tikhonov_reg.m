    function X = tikhonov_reg(A,b,beta)
        % SVD of Matrix
        [U,S,V] = svd(A,'econ');
        btilde = U'*b;
        % Compute Solution
        Xtilde = (S'*S + beta^2*eye(length(S'*S)))\(S'*btilde);
        % Final Solution
        X = V*Xtilde;
    end