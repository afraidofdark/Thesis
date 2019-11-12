function [center, U, obj_fcn] = ffcmw(data, cluster_n, options)
%FFCMW Fastest Fuzzy C-Means of the West!
%
%---------------------------------------------------------------------------------   
%   This is a fast implementation of the classical Fuzzy C-Means clustering
%   algorithm. In particular, it is faster than the implementation
%   provided withing the official Matlab Fuzzy Logic Toolbox (fcm.m).
%
%   If you can program it even faster, please share with us your code:
%   after all I'm looking for the fastest Fuzzy C-Means implementation of the West!
%   The obvious constraint is that it must be in pure Matlab code (no MEX
%   functions allowed). I'm also interested in faster versions obtained 
%   by exploiting GP-GPUs and/or and multicores (parfor).
%---------------------------------------------------------------------------------   
%
%   [CENTER, U, OBJ_FCN] = FFCMW(DATA, N_CLUSTER) finds N_CLUSTER number of
%   clusters in the data set DATA. DATA is size M-by-N, where M is the number of
%   data points and N is the number of coordinates for each data point. The
%   coordinates for each cluster center are returned in the rows of the matrix
%   CENTER. The membership function matrix U contains the grade of membership of
%   each DATA point in each cluster. The values 0 and 1 indicate no membership
%   and full membership respectively. Grades between 0 and 1 indicate that the
%   data point has partial membership in a cluster. At each iteration, an
%   objective function is minimized to find the best location for the clusters
%   and its values are returned in OBJ_FCN.
%
%   [CENTER, ...] = FFCMW(DATA,N_CLUSTER,OPTIONS) specifies a vector of options
%   for the clustering process:
%       OPTIONS(1): exponent for the matrix U             (default: 2.0)
%       OPTIONS(2): maximum number of iterations          (default: 100)
%       OPTIONS(3): minimum amount of improvement         (default: 1e-5)
%       OPTIONS(4): info display during iteration         (default: 1)
%   The clustering process stops when the maximum number of iterations
%   is reached, or when the objective function improvement between two
%   consecutive iterations is less than the minimum amount of improvement
%   specified. Use NaN to select the default value.
%
%   See also FCM, FCMDEMO, INITFCM, IRISFCM, DISTFCM, STEPFCM.

%   Marco Cococcioni
%   $ Revision: 1.0.4 $ 03-07-2019
%   $ Revision: 1.0.0 $ 16-09-2015

if nargin ~= 2 && nargin ~= 3
	error('Too many or too few input arguments!');
end

data_n = size(data, 1);

% Change the following to set default options
default_options = [   2;  % exponent for the partition matrix U
		            100;  % max. number of iteration
		           1e-5;  % min. amount of improvement
		             1];  % info display during iteration 

if nargin == 2
    options = default_options;
else
    % If "options" is not fully specified, pad it with default values.
    if length(options) < 4
        tmp = default_options;
        tmp(1:length(options)) = options;
        options = tmp;
    end
    % If some entries of "options" are nan's, replace them with defaults.
    nan_index = find(isnan(options)==1);
    options(nan_index) = default_options(nan_index);
    if options(1) <= 1
        error('The exponent should be greater than 1!');
    end
end

expo = options(1);		       % Exponent for U
max_iter = options(2);         % Max. iteration
min_impro = options(3);        % Min. improvement
display = options(4);          % Display info or not

obj_fcn = zeros(max_iter, 1);  % Array for objective function

U = rand(cluster_n, data_n);
col_sum = sum(U);
U = U./col_sum(ones(cluster_n, 1), :);
sum_data_dot_data = sum(data.*data,2)';

% main loop
for i = 1:max_iter
    
    % apply the exponential modification to the membership matrix
    mf = U.^expo;       

    % computing the euclidean distance
    center = bsxfun(@rdivide,mf*data,sum(mf,2));
    dist = bsxfun(@plus, sum_data_dot_data, sum(center.*center,2)) - 2*(center*data');
           
    % compute the objective function
    obj_fcn(i) = sum(sum(dist.*mf));
        
    % compute the new membership matrix U
    tmp = dist.^(-1/(expo-1));
    U = bsxfun(@rdivide, tmp, sum(tmp,1));

    if display 
        fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
    end
	
    % check terminating condition
    if ( i > 1 ) &&  ( abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro )
        break;        
    end
end

iter_n = i;	% actual number of iterations 
obj_fcn(iter_n+1:max_iter) = [];

