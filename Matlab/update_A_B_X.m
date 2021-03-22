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

function [A, B, Xs, errs] = update_A_B_X(A, B, Xs, Ys, epsilon_to_stop, s, min_iter)
%% update A, B and X to minimize || A*X1*B' - Y1 || + ... + || A*XM*B' - YM ||
%% A and B are general and have normalized columns

[n, ~, M] = size(Xs);
[p1, p2, N] = size(Ys);

Nerr = zeros(N,1);
err_old = inf;
for k = 1:N
    Nerr(k) = norm(Ys(:,:,k) - A*Xs(:,:,k)*B', 'fro')^2;
end
err_new = sqrt(sum(Nerr))/sqrt(p1*p2*N);

errs = [err_new];
iter = 0;
while (err_old - err_new >= epsilon_to_stop) || (iter <= min_iter)
    iter = iter +1;
    
    %% update A
    XXT = zeros(n);
    XYT = zeros(n);
    for m = 1:M
        aux = Xs(:, :, m)*B';
        XXT = XXT + aux*aux';
        XYT = XYT + aux*Ys(:, :, m)';
    end
    
    A = (XXT\XYT)';
    %% update X because of A
    for j = 1:n
        aux = norm(A(:, j));
        A(:, j) = A(:, j)/aux;
        Xs(j, :, :) = aux*Xs(j, :, :);
    end
    
    
    %% update B
    XTX = zeros(n);
    XTY = zeros(n);
    for m = 1:M
        aux = A*Xs(:, :, m);
        XTX = XTX + aux'*aux;
        XTY = XTY + aux'*Ys(:, :, m);
    end
    
    B = (XTX\XTY)';
    %% update X because of B
    for j = 1:n
        aux = norm(B(:, j));
        B(:, j) = B(:, j)/aux;
        Xs(:, j, :) = aux*Xs(:, j, :);
    end
    
    %% update sparse representations
    for m = 1:M
        Xs(:, :, m) = pair_omp_s_m(Ys(:, :, m), A, B', s);
    end
    
    %% update error
     err_old = err_new;
%     err_new = 0;
%     for m = 1:M
%         err_new = err_new + norm(A*Xs(:, :, m)*B' - Ys(:, :, m), 'fro')^2;
%     end
    
    for k = 1:N
        Nerr(k) = norm(Ys(:,:,k) - A*Xs(:,:,k)*B', 'fro')^2;
    end
    err_new = sqrt(sum(Nerr))/sqrt(p1*p2*N);
    
    %% to be removed
    errs = [errs err_new];
    stop = 1;
end
