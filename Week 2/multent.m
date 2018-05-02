% Define functions
schmidt = @(m) svd(m,'econ').^2;
reduce = @(s) s(s > sqrt(eps));	
calc_ent = @(s) sum(s.*log2(1./s));
entanglement = @(m) calc_ent(reduce(schmidt(m)));
kron4 = @(a,b,c,d) kron(kron(kron(d,c),b),a);

% Define the state and put it in a rank-4 tensor
d = 2;
up = [1;0];
down = [0; 1];
state = 
    kron4(down,down,down,down) + kron4(down,down,up,up) + ...
    kron4(up,up,down,down) + kron4(up,up,up,up) + ...
    kron4(down,up,down,up) + kron4(down,up,up,down) + ...
    kron4(up,down,down,up) + kron4(up,down,up,down);
state = state/norm(state);
ctensor = reshape(state,[d,d,d,d]);

% Form the bipartition matrices and compute entanglement
bipart = reshape(ctensor,[d,d^3]);
fprintf('Entanglement (A)(BCD): %g\n',entanglement(bipart)); 
bipart = reshape(ctensor,[d^2,d^2]);
fprintf('Entanglement (AB)(CD): %g\n',entanglement(bipart));
bipart = reshape(ctensor,[d^3,d]);
fprintf('Entanglement (ABC)(D): %g\n',entanglement(bipart));