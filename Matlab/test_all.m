% Copyright (c) 2018-2020 Paul Irofti <paul@irofti.net>
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
%% all images to consider
images = {'images/lena.bmp', 'images/peppers.bmp', 'images/boat.bmp'};

%% all patche sizes
%ps = [8 16 32 64];
ps = [8 16];

%% percentages of data to use
%percs = [0.1 0.25 0.5 0.75 1];
%repeats = [5 5 5 5 5 1];
percs = 0.75;
repeats = 5;

%% empty all variables
support = [];
errs_ortho = []; errs_general = []; errs_sukro = []; errsall = [];
errs_tefdil = [];
tus_q_faster_vec = []; tus_q_g = []; time_sukro = []; times = [];
time_tefdil = [];

addpath(genpath('TeFDiL'));
addpath(genpath('PARAFAC'));
addpath(genpath('STARK'));

%% for each patch size
for p = ps
    %% get full dataaset
    Ys = readImages(images, p*p);
    N = size(Ys, 2);
    
    index = 0;
    %% for each size of the dataset
    for perc = percs
        index = index + 1;
        for repeat = 1:repeats(index)
            %% sample the full dataset
            support = randsample(N, round(perc*N));
            Y = Ys(:, support);
            
            %% dictionary sizes
            ns = (1:3)*p;
            for n = ns
                %% sparsities
                %ss = round([(1:3)*log2(p*p) (1:3)*sqrt(p*p)]);
                ss = [round(log2(p*p)) round(sqrt(p*p))];
                ss = unique(ss);
                
                %% initial dictionaries for all methods
                A = odct(p, n);
                B = odct(p, n);
                for s = ss
                    %filename = ['results ' num2str(p) ' ' num2str(perc) ' ' num2str(repeat) ' ' num2str(n) ' ' num2str(s) '.mat'];
                    filename = ['tefdil results ' num2str(p) ' ' num2str(perc) ' ' num2str(repeat) ' ' num2str(n) ' ' num2str(s) '.mat'];
                    if exist(filename, 'file') == 2
                        continue;
                    end
                    
                    X = omp_sparse(kron(B', A')*Y, kron(B'*B, A'*A), s);
                    
%{
                    if p == n
                        tic;
                        epsilon_to_stop = 0.0000000001;
                        [Q1, Q2, X_ortho, errs_ortho] = ...
                            update_Q1_Q2_faster_vec(A, B, X, Y, epsilon_to_stop, s);
                        tus_q_faster_vec = toc;
                        
                        min_iter = 100;
                        tic;
                        epsilon_to_stop = 0.0000000001;
                        [A_new, B_new, ~, errs_general] = ...
                           update_A_B_X_vec(Q1, Q2, X_ortho, Y, epsilon_to_stop, s, min_iter);
                        tus_q_g = toc;
                    else
                        min_iter = 100;
                        epsilon_to_stop = 0.0000000001;
                        tic;
                        [A_new, B_new, ~, errs_general] = ...
                           update_A_B_X_vec(A, B, X, Y, epsilon_to_stop, s, min_iter);
                        tus_q_g = toc;
                    end
                    
                    %% setup to call the toolbox
                    updates = {'pair_dl_vec', 'aksvd'};
                    spfuncs = {'pair_omp_vec', 'omp'};
                    D = A;
                    ups = length(updates);
                    Xall = cell(ups,1);
                    D0 = cell(ups,1);
%                     D0{2} = kron(D,D);
                    % for AKSVD get an overcomplete initial dictionary
                    D0{2} = odct(p*p, n*p);
                    D0{1} = {D D'};
                    Ytrain = cell(1,1);
                    Ytrain{1} = Y;
                    iters = 100;
                    
                    times = zeros(ups, 1);
                    Dall = cell(1, ups, 1);
                    Xtrainall = cell(1, ups, 1);
                    errsall = zeros(1, ups, iters);
                    replatom = 'worst';
                    for i = 1:ups
                        tic;
                        [Dall{1}{i}, Xtrainall{1}{i}, errsall(1,i,:)] = ...
                            DL(Ytrain{1}, D0{i}, s, iters, ...
                            str2func(updates{i}), 'spfunc', str2func(spfuncs{i}), ...
                            'replatom', replatom);
                        times(i) = toc;
                    end
                    
                    %% SuKro
                    tic
                    params.iternum = 200;        % Number of iterations
                    params.memusage = 'high';   % Memory usage
                    params.initdict = kron(A, B);% Initial dictionary
                    params.data = Y./255;           % Data to approximate
                    params.u = 500;             % ADMM coefficient
                    params.alpha = 2.15;        % Regularization parameter
                    % Sparse coding parameters
                    params.codemode = 'sparsity';   % 'error' or 'sparsity'
                    params.Tdata = s;     % when using codemode = 'sparsity'
                    % Dimensions of the subdictionaries
                    params.kro_dims.N1 = p; params.kro_dims.N2 = p;
                    params.kro_dims.M1 = n; params.kro_dims.M2 = n;

                    tic;
                    [D_sukro, D_not_normalized, X_sukro, errs_sukro] = sum_separable_dict_learn(params);
                    time_sukro = toc;
%}
                    %% TeFDiL
                    tic;
                    N_freq = 2;
                    M = [p, p];
                    P =[n, n];

                    Dictionary_sizes{1}=fliplr(M);
                    Dictionary_sizes{2}=fliplr(P);
                    [Permutation_vector, Permutation_vectorT]=permutation_vec(Dictionary_sizes);
                    Permutation_vectors=[Permutation_vector, Permutation_vectorT];

                    paramSC.s = s;
                    paramSC.SparseCodingMethod= 'OMP';
                    Max_Iter_DL = 50;
                    tol_DL = 10^(-4);

                    D = kron(A, B);

                    ParamTeFDiL.MaxIterCP=50;
                    ParamTeFDiL.DicSizes=Dictionary_sizes;
                    ParamTeFDiL.Sparsity=s;
                    ParamTeFDiL.MaxIterDL=Max_Iter_DL;
                    ParamTeFDiL.TolDL=tol_DL;
                    ParamTeFDiL.epsilon=0.01;
                    ParamTeFDiL.TensorRank=1;
                    [D_TeFDiL, X_TeFDiL, errs_tefdil] = TeFDiL(Ys, ...
                        Permutation_vectors, D, ParamTeFDiL, paramSC);

                    errs_tefdil=errs_tefdil./sqrt(numel(Ys));
                    time_tefdil = toc;
                    
                    %% SAVE DATA
                    save(['tefdil results ' num2str(p) ' ' num2str(perc) ' ' num2str(repeat) ' ' num2str(n) ' ' num2str(s) '.mat'], ...
                        'support', ...
                        'errs_tefdil', 'time_tefdil');
                        %'errs_ortho', 'errs_general', 'errs_sukro', 'errsall', ...
                        %'tus_q_faster_vec', 'tus_q_g', 'time_sukro', 'times');
                    stop = 1;
                end
            end
        end
    end
end
