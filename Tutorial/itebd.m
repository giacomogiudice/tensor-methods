% This script runs Time-Evolving Block Decimation (TEBD) for a 
% periodic lattice with unit cell
%     |                    |    
% -Gamma{A}--Lambda{A}--Gamma{B}--Lambda{B}--
%
% Where Lambda are diagonal, containing the Schmidt coefficents
% Indexing convention for Gamma is (left, up, right)
% Parameters    
d = 2;			% Physical dimension
D = 4;			% Bond dimension
g = 0.5;		% Coupling
dt = 0.01;		% Time step
iter_max = 1000;% Number of time steps
% The good ol' Pauli matrices
sx = [0,1; 1,0];
sy = [0,1i; -1i,0];
sz = [1,0; 0,-1];
si = [1,0; 0,1];
% The Gamma tensors
G = cell(1,2);
G{1} = rand(D,d,D);
G{2} = rand(D,d,D);
% The Lambda tensors
lambda = cell(1,2);
lambda{1} = diag(rand(D,1));
lambda{2} = diag(rand(D,1));
% Create 2-site Hamiltonian and evolution operator
h = -ncon({sz,sz},{[-1,-3],[-2,-4]}) - g/2*(ncon({sx,si},{[-1,-3],[-2,-4]}) + ncon({si,sx},{[-1,-3],[-2,-4]}));
U = expm(-dt*reshape(h,[d^2,d^2]));
U = reshape(U,[d,d,d,d]);
% Start loop
E_exact = integral(@(k) (-1/pi)*sqrt(1+g^2-2*g*cos(k)),0,pi);
fprintf('Running iTEBD for D=%d...\n',D)
for iter = 1:iter_max
	% Alternate A,B
	A = mod(iter,2)+1;
	B = mod(A,2)+1;
	% Get time step
	block_twosite = ncon({lambda{B},G{A},lambda{A},G{B},lambda{B}},{[-1,1],[1,-2,2],[2,3],[3,-3,4],[4,-4]});
	theta = ncon({block_twosite,U},{[-1,1,2,-4],[-2,-3,1,2]});
	% Perform SVD and truncate
	[X,S,Y] = svd(reshape(theta,[d*D,d*D]),0);
	X = X(:,1:D);
	S = S(1:D,1:D);
	S = S/norm(diag(S));
	Y = Y(:,1:D);
	% Reshape into rank-3 tensors
	X = reshape(X,[D,d,D]);
	Y = reshape(Y',[D,d,D]);
	% Compute inverse
	B_diag = diag(lambda{B});
	lambda_inv = zeros(D,1);
	indx = find(B_diag > 1e-8);
	lambda_inv(indx) = 1./B_diag(indx);
	lambda_inv = diag(lambda_inv);
	% Apply inverse
	G{A} = ncon({lambda_inv,X},{[-1,1],[1,-2,-3]});
	G{B} = ncon({Y,lambda_inv},{[-1,-2,1],[1,-3]});
	lambda{A} = S;
	% Display convergence
	if mod(iter,100) == 0 || iter == iter_max
		block_twosite = ncon({lambda{B},G{A},lambda{A},G{B},lambda{B}},{[-1,1],[1,-2,2],[2,3],[3,-3,4],[4,-4]});
		E = ncon({conj(block_twosite),h,block_twosite},{[5,1,2,6],[1,2,3,4],[5,3,4,6]});
		fprintf('  E=%.7g,\t|E-E_exact|=%.4g,\tdiff%c=%.4g,\n',E,abs(E - E_exact),char(955),norm(lambda{A} - lambda{B}));
	end
end
fprintf('...Done\n');
% Define 2-site magnetization operators
magn_x_twosite = 1/2*(ncon({sx,si},{[-1,-3],[-2,-4]}) + ncon({si,sx},{[-1,-3],[-2,-4]}));
magn_z_twosite = 1/2*(ncon({sz,si},{[-1,-3],[-2,-4]}) + ncon({si,sz},{[-1,-3],[-2,-4]}));
% Compute magnetization
magn_x = ncon({conj(block_twosite),magn_x_twosite,block_twosite},{[5,1,2,6],[1,2,3,4],[5,3,4,6]});
magn_z = ncon({conj(block_twosite),magn_z_twosite,block_twosite},{[5,1,2,6],[1,2,3,4],[5,3,4,6]});
fprintf('Magnetization along [x,z]: [%g, %g]\n',magn_x,magn_z);
% Compute entanglement entropy
S = -sum(diag(lambda{A}.^2).*log2(diag(lambda{A}.^2)));
fprintf('Entanglement entropy: %g\n',S);
