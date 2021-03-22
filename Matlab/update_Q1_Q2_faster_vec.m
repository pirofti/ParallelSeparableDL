% Copyright (c) 2020 Cristi Rusu <cristi.rusu.tgm@gmail.com>
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

function [Q1, Q2, Xs, errs] = update_Q1_Q2_faster_vec(Q1, Q2, Xs, Ys, epsilon_to_stop, s)
%% update Q1 and Q2 to minimize || Q1*X1*Q2' - Y1 || + ... + || Q1*XM*Q2' - YM ||
%% Q1 and Q2 are orthonormal

[n, M] = size(Xs);
sqrtn = sqrt(n);
[p, N] = size(Ys);

err_old = inf;
errs = [];
Q = kron(Q2, Q1);
err_new = norm(Ys - Q*Xs, 'fro')/sqrt(p*N);

errs = [errs err_new];

aux = zeros(sqrtn, sqrtn);
while (err_old - err_new >= epsilon_to_stop)
    %% update useful matrices
    XXT = zeros(sqrtn);
    XTX = zeros(sqrtn);
    XYT = zeros(sqrtn);
    for m = 1:M
        aux = reshape(Xs(:, m), sqrtn, sqrtn);
        XXT = XXT + aux*aux';
        XTX = XTX + aux'*aux;
        XYT = XYT + aux*Q2'*reshape(Ys(:, m), sqrtn, sqrtn)'; 
    end
    
    %% update A
    Z = XYT'*XXT;
    [U, ~, V] = svd(Z);
    Q1 = (U*V');
    
    
    %% update B
    XTY = zeros(sqrtn);
    for m = 1:M
        XTY = XTY + reshape(Xs(:, m), sqrtn, sqrtn)'*Q1'*reshape(Ys(:, m), sqrtn, sqrtn);
    end
    
    Z = XTY'*XTX;
    [U, ~, V] = svd(Z);
    Q2 = (U*V');
    
    
    %% update representations
%     Xs = sparse([], [], [], n, M, s*M);
    Q = kron(Q2, Q1);
    Xs = omp_sparse(Q'*Ys, eye(n), s);
%     [~, ind_sort] = sort(abs(aux), 'descend');
%     for m = 1:M
%         Xs(:, m) = 0;
%         Xs(ind_sort(1:s, m), m) = aux(ind_sort(1:s, m), m);
%     end
    
    %% update error
    err_old = errs(end);
    err_new = norm(Ys - Q*Xs, 'fro')/sqrt(p*N);
    
    %% to be removed
    errs = [errs err_new];
    stop = 1;
end
