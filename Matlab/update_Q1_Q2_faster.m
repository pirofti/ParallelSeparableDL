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

function [Q1, Q2, Xs, errs] = update_Q1_Q2_faster(Q1, Q2, Xs, Ys, epsilon_to_stop, s)
%% update Q1 and Q2 to minimize || Q1*X1*Q2' - Y1 || + ... + || Q1*XM*Q2' - YM ||
%% Q1 and Q2 are orthonormal

[n, ~, M] = size(Xs);
[p1, p2, N] = size(Ys);

Nerr = zeros(N,1);
err_old = inf;
errs = [];
for k = 1:N
    Nerr(k) = norm(Ys(:,:,k) - Q1*Xs(:,:,k)*Q2', 'fro')^2;
end
err_new = sqrt(sum(Nerr))/sqrt(p1*p2*N);

errs = [errs err_new];

while (err_old - err_new >= epsilon_to_stop)
    %% update useful matrices
    XXT = zeros(n);
    XYT = zeros(n);
    for m = 1:M
        XXT = XXT + Xs(:, :, m)*Xs(:, :, m)';
        XYT = XYT + Xs(:, :, m)*Q2'*Ys(:, :, m)';
    end
    
    
    %% update A
    Z = XYT'*XXT;
    [U, ~, V] = svd(Z);
    Q1 = (U*V');
    
    
    %% update B
    XTY = zeros(n);
    XTX = zeros(n);
    for m = 1:M
        XTX = XTX + Xs(:, :, m)'*Xs(:, :, m);
        XTY = XTY + Xs(:, :, m)'*Q1'*Ys(:, :, m);
    end
    
    Z = XTY'*XTX;
    [U, ~, V] = svd(Z);
    Q2 = (U*V');
    
    
    %% update representations
    for m = 1:M
        aux = Q1'*Ys(:, :, m)*Q2;
        [~, ind_sort] = sort(abs(aux(:)), 'descend');
        aux(ind_sort(s+1:end)) = 0;

        Xs(:, :, m) = aux;
    end
    
    %% update error
    err_old = errs(end);
    for k = 1:N
        Nerr(k) = norm(Ys(:,:,k) - Q1*Xs(:,:,k)*Q2', 'fro')^2;
    end
    err_new = sqrt(sum(Nerr))/sqrt(p1*p2*N);
    
    %% to be removed
    errs = [errs err_new];
    stop = 1;
end
