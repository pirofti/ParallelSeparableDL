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

% Cite as:
% P. Irofti and B. Dumitrescu, �Pairwise Approximate K-SVD,� 
% in Acoustics Speech and Signal Processing (ICASSP), 
% 2019 IEEE International Conference on, 2019, pp. 1--5.

%% Separable: 2D DL denoising with training on noisy image
function test_denoising_sepsum()
clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
p = 8;                  % patch size
s = 6;                  % sparsity
N = 4000;              % total number of patches
n = 64;                % dictionary size
[n1, n2] = deal(sqrt(n));
%n1 = n1*sqrt(p);
%n2 = n2*sqrt(p);
iters = 600;             % DL iterations
%iters = 1000;          % DL iterations
replatom = 'worst';     % replace unused atoms
stds = [5 10 20 30 50]; % noise standard deviation
%%-------------------------------------------------------------------------
updates = {'pair_dl_vec', 'aksvd'};
spfuncs = {'pair_omp_vec', 'omp'};
methods = [
  % Name	Function         Dictionary index
  {'pair', @denoise_omp,     1};
  {'ortho', @denoise_omp,  2};
  {'general', @denoise_omp,   3};
];
%%-------------------------------------------------------------------------
datadir = 'data\';
dataprefix = 'pair_denoise';

imdir = 'denoising\';
img_test = {'barbara.png', 'boat.png'};
% img_test = {'lena.png', 'barbara.png', 'boat.png', 'peppers.png' 'house.png',};
%%-------------------------------------------------------------------------
addpath(genpath('DL'));     % Set to your local copy of dl-box
ts = datestr(now, 'yyyymmddHHMMss');
%%-------------------------------------------------------------------------
% INITIALIZATION
%%-------------------------------------------------------------------------
ups = length(updates);
Dall = cell(ups,1);
Xall = cell(ups,1);
D0 = cell(ups,1);
D0{2} = odctdict(p^2,n);
%D0{2} = {odctdict(p,n1) odctdict(n2,p)};
D0{1} = {odctdict(p,n1) odctdict(n2,p)};
%%-------------------------------------------------------------------------
% DENOISING
%%-------------------------------------------------------------------------
f = {'name', 'func', 'dict'};
m = cell2struct(methods, f, 2);
%-------------------------------------------------------------------------
results = sprintf('%s\n', dataprefix);
Yall = {};
Xall = {};
Iall = {};
psnrall = {};
ssimall = {};
%-------------------------------------------------------------------------
for iimg = 1:length(img_test)
    img = img_test{iimg};
    for sigma = stds
        do_img(img, sigma);        
        clc; disp(results);
    end
end
%-------------------------------------------------------------------------
function do_img(img, sigma)
    fname = [datadir dataprefix '-' img '-std' num2str(sigma) ...
        '-n' num2str(n) '-' ts '.mat'];

    %% Initial Data
    [I, Inoisy, Ynoisy, Ynmean] = ...
        denoise_init_data([imdir,char(img)], sigma, p, p);    
    %save(fname, 'Inoisy', 'Ynoisy', 'Ynmean');
    
    results = sprintf('%s%s sigma=%2d psnr=%f:\n', results, ...
        img, sigma, psnr(Inoisy,I,255));
    clc;disp(results);
  
    function do_denoise(name, dfunc, D, i)
        [Iall{i}, Yall{i}, Xall{i}] = ...
            denoise(dfunc, Inoisy, D, {'sigma', sigma, 's', s}, p, p);
        psnrall{i} = psnr(Iall{i},I,255);
        ssimall{i} = ssim(Iall{i}, I, 'DynamicRange', 255);
        results = sprintf('%s %s psnr=%f ssim=%f\n', ...
            results, name, psnrall{i}, ssimall{i});
        clc; disp(results);
    end

    Dall = cell(ups,1);
    Xtrainall = cell(ups,1);
    errsall = zeros(ups,iters);
    Ytrain = Ynoisy(:,randperm(size(Ynoisy,2), N));
    
    [Dpak, ~, ~] = ...
        DL(Ytrain, D0{1}, s, iters, ...
        str2func(updates{1}), 'spfunc', str2func(spfuncs{1}), ...
        'replatom', replatom);
    Dall{1} = kron(Dpak{2}',Dpak{1});
    
    %A = D0{1}{1};
    %B = D0{1}{2}';
    A = odct(p, n1);
    B = odct(p, n2);
    Y = Ytrain;
    X = omp_sparse(kron(B', A')*Y, kron(B'*B, A'*A), s);

    if p*p == n
        tic;
        epsilon_to_stop = 0.0000000001;
        [Q1, Q2, X_ortho, errs_ortho] = ...
            update_Q1_Q2_faster_vec(A, B, X, Y, epsilon_to_stop, s);
        tus_q_faster_vec = toc;
        Dall{2} = kron(Q1 , Q2);

        min_iter = iters;
        tic;
        epsilon_to_stop = 0.0000000001;
        [A_new, B_new, ~, errs_general] = ...
           update_A_B_X_vec(Q1, Q2, X_ortho, Y, epsilon_to_stop, s, min_iter);
        tus_q_g = toc
        Dall{3} = kron(A_new, B_new);
    else       
        min_iter = iters;
        epsilon_to_stop = 0.0000000001;
        tic;
        [A_new, B_new, ~, errs_general] = ...
           update_A_B_X_vec(A, B, X, Y, epsilon_to_stop, s, min_iter);
        tus_q_g = toc
        Dall{2} = kron(A_new, B_new);
        
%         figure(1);
%         plot(1:length(errs_general), errs_general);
%         hold on;
    end

    %{
    for i = 1:ups
        fprintf('%s', updates{i}(1));
        [Dall{i}, Xtrainall{i}, errsall(i,:)] = ...
            DL(Ytrain, D0{i}, s, iters, ...
            str2func(updates{i}), 'spfunc', str2func(spfuncs{i}), ...
            'replatom', replatom);
    end
    %}
    for i = 1:size(methods,1)
        do_denoise(m(i).name, m(i).func, Dall{m(i).dict}, i);
    end
    
    %% Save DL-based results
%     save(fname, 'Yall', 'Xall', 'Iall', ...
%         'Dall', 'Xtrainall', 'errsall', ...
%         'psnrall', 'ssimall', '-append');

    clc; disp(results);
end       
end