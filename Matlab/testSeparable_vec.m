% Copyright (c) 2020 Paul Irofti <paul@irofti.net>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
p = 8;                  % patch size
s = 4;                  % sparsity
N = 4000;               % total number of patches
%n = 256;                % dictionary size
%nfactor = [4 6 8];      % 2D atoms: nfactor(i)*sqrt(n)
nfactor = 1;      % 2D atoms: nfactor(i)*sqrt(n)
iters = 500;            % DL iterations
replatom = 'worst';     % replace unused atoms
rounds = 1;             % rounds

%%-------------------------------------------------------------------------
updates = {'sedil', 'pair_dl_vec', 'aksvd'};
spfuncs = {'NOP', 'pair_omp_vec', 'omp'};
methods = [
  % Name	Function         Dictionary index
  {'ak2D', @denoise_2D,      1};
  {'ak1D', @denoise_omp,     2};
];

%addpath(genpath('DL')); % Set to your local copy of dl-box
ts = datestr(now, 'yyyymmddHHMMss');
%%-------------------------------------------------------------------------

%% size of the patch, nxn
% images = {'images/lena.bmp', 'images/peppers.bmp', 'images/boat.bmp'};
images = {'images/lena.bmp'};
Ys = readImages(images, p*p);
% Ys = Ys./255;

%% how many patches
[n, M] = size(Ys);

%% these are the M representations, each nxn (keep full, not sparse)
Xs = sparse([], [], [], n, M, s*M);
D = kron(dctmtx(sqrt(n))', dctmtx(sqrt(n))'); % the initial dictionary is the DCT matrix
aux = D*Ys;
clear D
[~, ind_sort] = sort(abs(aux), 'descend');
for m = 1:M
    Xs(ind_sort(1:s, m), m) = aux(ind_sort(1:s, m), m);
end
clear aux
clear ind_sort

%%% RUN THE ALGORITHMS
min_iter = 10; % run update_A_B_X for a minimum of min_iter steps

%% STARK
addpath(genpath('STARK'));
addpath(genpath('TeFDiL'));
addpath(genpath('PARAFAC'));

tic
N_freq = 2;
M = [sqrt(n), sqrt(n)];
P =[sqrt(n) ,sqrt(n)];

Dictionary_sizes{1}=fliplr(M);
Dictionary_sizes{2}=fliplr(P);
[Permutation_vector, Permutation_vectorT]=permutation_vec(Dictionary_sizes);
Permutation_vectors=[Permutation_vector, Permutation_vectorT];

%% Algorithm Parameters
K = 2; %tensor order
% Sparse Coding Parameters. Have to select sparse coding method here:
% 'OMP', 'SPAMS', 'FISTA'
% paramSC is input to all DL algorithms
paramSC.s = s;
paramSC.SparseCodingMethod= 'OMP';

% Dictionary Learning Parameters
Max_Iter_DL = 50;
tol_DL = 10^(-4);

%initializing subdictionaries
D = kron(dctmtx(sqrt(n)),dctmtx(sqrt(n)));

% STARK Parameters
% ParamSTARK.TolADMM = 1e-4; %tolerance in ADMM update
% ParamSTARK.MaxIterADMM = 10;
% ParamSTARK.DicSizes=Dictionary_sizes;
% ParamSTARK.Sparsity=s;
% ParamSTARK.MaxIterDL=Max_Iter_DL;
% ParamSTARK.TolDL=10^(-4);
% 
% lambdaADMM=norm(Ys,'fro')^(1.5)/10;
% gammaADMM = lambdaADMM/20;
% ParamSTARK.lambdaADMM=lambdaADMM;
% ParamSTARK.gammaADMM=gammaADMM;
% [D_STARK, X_STARK, errs_stark] = STARK(Ys, ...
%     Permutation_vectors, D, ParamSTARK, paramSC);
% errs_stark=errs_stark./sqrt(numel(Ys));

% TeFDiL Parameters
ParamTeFDiL.MaxIterCP=50;
ParamTeFDiL.DicSizes=Dictionary_sizes;
ParamTeFDiL.Sparsity=s;
ParamTeFDiL.MaxIterDL=Max_Iter_DL;
ParamTeFDiL.TolDL=tol_DL;
ParamTeFDiL.epsilon=0.01; %to improve the condition number of XX^T. multiplied by its frobenious norm.
ParamTeFDiL.TensorRank=1;
[D_TeFDiL, X_TeFDiL, errs_tefdil] = TeFDiL(Ys, ...
    Permutation_vectors, D, ParamTeFDiL, paramSC);

errs_tefdil=errs_tefdil./sqrt(numel(Ys));
toc

%% SuKro
tic
D = kron(dctmtx(sqrt(n)),dctmtx(sqrt(n)));
params.iternum = 30;        % Number of iterations
params.memusage = 'high';   % Memory usage
params.initdict = D;% Initial dictionary
params.data = Ys./255;           % Data to approximate
params.u = 500;             % ADMM coefficient
params.alpha = 2.15;        % Regularization parameter
% Sparse coding parameters
params.codemode = 'sparsity';   % 'error' or 'sparsity'
params.Edata = 1e-1;            % when using codemode = 'error'
params.Tdata = ceil(size(D,1)/10);     % when using codemode = 'sparsity'
% Dimensions of the subdictionaries
params.kro_dims.N1 = p; params.kro_dims.N2 = p;
params.kro_dims.M1 = sqrt(nfactor*n); params.kro_dims.M2 = sqrt(nfactor*n);

% Checking subdictionary dimensions
assert( params.kro_dims.N1*params.kro_dims.N2==p^2 && ...
        params.kro_dims.M1*params.kro_dims.M2==n, ...
        'N (resp. M) should be equal to N1*N2 (resp. M1*M2)')
assert( round(params.kro_dims.N1)==params.kro_dims.N1 && ...
        round(params.kro_dims.N2)==params.kro_dims.N2 && ...
        round(params.kro_dims.M1)==params.kro_dims.M1 && ...
        round(params.kro_dims.M2)==params.kro_dims.M2,   ...
        'N1,N2,M1 and M2 should all be integers')

% Running SuKro algorithm
flops_dense = 2*size(Ys,1)*size(Ys,2);

[D, D_not_normalized, X, errs_sukro] = sum_separable_dict_learn(params);
clear D; 
time_sukro = toc

%% 2D general
% tic;
% [A, B, Xs_gen, errs_general] = ...
%     update_A_B_X(D, D, Xs, Ys, epsilon_to_stop, s, min_iter);
% tus_g = toc;

%% 2D ortho
tic;
epsilon_to_stop = 0.00001;
[Q1, Q2, Xs_ortho, errs_ortho_faster_vec] = ...
    update_Q1_Q2_faster_vec(dctmtx(sqrt(n)), dctmtx(sqrt(n)), Xs, Ys, epsilon_to_stop, s);
tus_q_faster_vec = toc

% tic;
% Xs_sparse = sparse(Xs);
% [Q1, Q2, Xs_ortho_sparse, errs_ortho_faster_sparse] = ...
%     update_Q1_Q2_faster_sparse(D, D, Xs_sparse, Ys, epsilon_to_stop, s);
% tus_q_sparse_faster = toc;

%% a combination
tic;
epsilon_to_stop = 0.000000000000001;
A = odct(sqrt(n), sqrt(n));
B = odct(sqrt(n), sqrt(n));
Q = kron(B, A);
Xs = omp_sparse(Q'*Ys, kron(B'*B, A'*A), s);
min_iter = 50;
[A2, B2, ~, errs_ortho_general_vec] = ...
    update_A_B_X_vec(Q1, Q2, Xs_ortho, Ys, epsilon_to_stop, s, min_iter);
%    update_A_B_X_vec(A, B, Xs, Ys, epsilon_to_stop, s, min_iter);
% 
tus_q_g = toc

%%-------------------------------------------------------------------------
% EXPERIMENTS
%%-------------------------------------------------------------------------
for nf = nfactor
    %fprintf('nf=%d: ', nf);
    [n1, n2] = deal(nf*sqrt(n));
%%-------------------------------------------------------------------------
% INITIALIZATION
%%-------------------------------------------------------------------------
    % Glue
    N = M;
    iters = 50;    
    
    D = dctmtx(sqrt(n));
    ups = length(updates);
    Xall = cell(ups,1);
    D0 = cell(ups,1);
    %D0{2} = odctdict(p^2,n);
    %D0{1} = {odctdict(p,n1) odctdict(n2,p)};
    D0{3} = kron(D,D);
    D0{2} = {D D};
    D0{1} = {D D};
%%-------------------------------------------------------------------------
% LEARNING
%%-------------------------------------------------------------------------
    Dall = cell(rounds,ups,1);
    Xtrainall = cell(rounds,ups,1);
    errsall = zeros(rounds,ups,iters);
    Y = [];
    Ytrain = cell(rounds,1);
%% (PAIR-)AKSVD
    for r = 1:rounds
        Ytrain{r} = Ys;
        %fprintf('%d', mod(r,rounds));
        for i = 1:ups
            %fprintf('%s', updates{i}(1));
            fprintf('%s\n', updates{i});
            tic;
            [Dall{r}{i}, Xtrainall{r}{i}, errsall(r,i,:)] = ...
                DL(Ytrain{r}, D0{i}, s, iters, ...
                str2func(updates{i}), 'spfunc', str2func(spfuncs{i}), ...
                'replatom', replatom);
                time = toc
        end
    end
end

%% PLOT
plot(errs_ortho_faster_vec, 'r');
%hold on; plot(errs_sukro, 'c');
hold on; plot(errs_ortho_general_vec, 'y');
hold on; plot(errs_tefdil, 'g');
hold on; plot(squeeze(errsall(1,1,:)), 'm');
hold on; plot(squeeze(errsall(1,2,:)), 'b');
hold on; plot(squeeze(errsall(1,3,:)), 'k');
legend('ortho', 'general', 'TeFDiL', ...
    updates{1}, updates{2}, updates{3});
xlabel('iteration');
ylabel('error');
