(* ::Package:: *)

% Incomplete Markets Model - Partial Equilibrium 
%     The interest rate and endowment are given. 
%
clear('all')

% basic parameters
beta = .95; % discount factor
R = 1.04;   % interest rate
bR = beta * R;
rho = 2;    % risk aversion : u(c)= c^(1-rho)/(1-rho)
phi = 0;    % borrowing limit (must be positive)

% Structure of the shocks
nshocks = 2;   % two shocks
unos = ones(nshocks, 1); % useful vector
P = [.5  .5; .2  .8]; % transition matrix
Y = [0.5; 1];  % endowments at the two shocks (changing the wage changes this)
   
Yhat = Y - (R - 1) * phi * unos;  % modified Y 

% grid for the cash in hand today
zN = 2000;   % number of grid points
ahatmax = 15;  % maximum amount of savings.. this is just a guess
ahatmin = 0;
ahatgrid = linspace(ahatmin, ahatmax, zN);

% cash in hand tomorrow given ahatgrid
zprime = unos * R * ahatgrid + Yhat * ones(1, zN);

% an initial guess for the policy function tomorrow
% (one policy function per state)
ahatpolicy = unos * ahatgrid;

difference = 1; % initializing the distance
zold = 0;

% The following loop iterates the policy function
% until convergence. It does so by iterating on tomorrow's 
% policy function. This avoids solving the root of the euler equation
% which is a computationally intensive procedure. As a result, this is
% really fast.
while difference > 1e-7
    z = (P * bR * (zprime - ahatpolicy).^(-rho)).^(-1/rho) ...
        + unos * ahatgrid;
                            % this computes the cash in hand today
                            % consistent with the the policy tomorrow being
                            % ahatpolicy
    difference = max(max(abs(zold - z)))   % check the distance
    zold = z;
    for i=1:nshocks
        % given the implicit optimal policy today z -> ahatgrid
        % now we interpolate zprime -> ahatpolicy!
        % (and we check the borrowing limit)
        ahatpolicy(i,:) = max(interp1(z(i,:), ahatgrid, zprime(i,:), ...
        'linear', 'extrap'), 0);    % recall that borrowing contrainst
                                    % in the zhat problem is ahat >= 0
    end
end

% We have converged

disp('-----------------')
disp('Parameters ')
disp(' ')
disp('beta: ')
disp(beta)
disp('R: ')
disp(R)
disp('beta * R:')
disp(bR)
disp('rho: ')
disp(R)
disp('phi: ')
disp(phi)
disp('Y:')
disp(Y)
disp('Transition probability:')
disp(P)
disp('---------------- ')

% Getting the grid in terms of cash in hands:
xgrid = zprime - phi * ones(nshocks, zN);

% Getting the asset policy in terms of actual asset position
apolicy =  ahatpolicy -  phi * ones(nshocks, zN);
% Transition plots : one per state
for i=1:nshocks
    figure;
    plot(xgrid (i,:)', xgrid (i,:)', '-');
    title({strcat('Transition graph for cash in hands: state ', num2str(i));
           strcat('Probabilities high and low: ', mat2str(P(i,:)))});
    xlabel('z');
    ylabel('zprime');
    hold on;
    plot(xgrid (i,:)', (R * unos * apolicy(i,:) + Y * ones(1, zN))', '.-');
    hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we compute the stationary distribution
disp('computing stationary distribution')

% given that we are interested in assets, we translate
% from cash in hands to asses:
agrid = (xgrid - Y * ones(1,zN))/R;  

simplegrid = 1:zN;   % a simple index
for i=1:nshocks
    apolicyindex(i,:) = interp1(agrid(i,:), simplegrid, ...
                                     apolicy(i,:), 'nearest');
    % given the policy, this tell us the asset "position" in the grid 
    % that generated it. 
end

if max(max(isnan(apolicyindex))) > 0
    disp('!!asset grid is narrower than asset policy: increase ahatmax!!')
    break
end

% creating big transition matrix
T = zeros(nshocks*zN);
for i=1:nshocks
    for j=1:zN
        T( (i - 1)* zN + j, (0:nshocks-1) * zN + ...
            apolicyindex (:, j)' ) = P(i,:);
    end
end
% 
% % constructing an initial distribution
% for i=1:nshocks
%    initial_dist((i-1)*zN+1:i*zN) = P (i,1)/(zN * nshocks);
% end
% 
% % iterating towards the stationary distribution
% f = initial_dist;
% distance2 = 1;
% while distance2 > 1e-6
%     final_dist = f * T;
%     distance2 = max(abs(f - final_dist))
%     f = final_dist;
% end

% another way of finding the stationary distribution
% 
% disp('getting the eigenvector')
% % computing the stationary distribution by finding the eigenvector
% % associated with the eigenvalue = 1 of the transition matrix T'
[final_dist eigenvalue] = eigs(T', 1, 1);
final_dist = final_dist / sum(final_dist);  % the dist sums to 1

% plotting the distribution per employment state
for i=1:nshocks
    figure;
    plot(agrid(i,:),final_dist((i-1)*zN + 1: i*zN));
    title(strcat('Stationary distribution by employment state:', ...
        num2str(i) ));
end
                                                                      
disp('Done')                                    

