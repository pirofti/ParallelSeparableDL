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

function [A, B, Xs, errs] = update_A_B_X_vec(A, B, Xs, Ys, epsilon_to_stop, s, min_iter)
%% update A, B and X to minimize || A*X1*B' - Y1 || + ... + || A*XM*B' - YM ||
%% A and B are general and have normalized columns

[n, M] = size(Xs);
sqrtn = sqrt(n);
[p, N] = size(Ys);
sqrtp = sqrt(p);

err_old = inf;
Q = kron(B, A);
err_new = norm(Ys - Q*Xs, 'fro')/sqrt(p*N);

errs = [err_new];
iter = 0;
%while (err_old - err_new >= epsilon_to_stop) || (iter <= min_iter)
while (iter <= min_iter)
    iter = iter +1;
    
    %% update A
    XXT = zeros(sqrtn);
    XYT = zeros(sqrtn, sqrtp);
    for m = 1:M
        aux = reshape(Xs(:, m), sqrtn, sqrtn)*B';
        XXT = XXT + aux*aux';
        XYT = XYT + aux*reshape(Ys(:, m), sqrtp, sqrtp)';
    end
    
    A = (XXT\XYT)';
    [~, msgid] = lastwarn;
    if ~isempty(msgid)
        if strcmp(msgid, 'MATLAB:nearlySingularMatrix') || ...
            strcmp(msgid, 'MATLAB:singularMatrix')
            A = (pinv(XXT)*XYT)';
        end
    end
    %% update X because of A
%     A = normc(A);
    for j = 1:sqrtn
        aux = norm(A(:, j));
        if (aux < 10e-15)
            A(:, j) = randn(sqrtp, 1);
            aux = norm(A(:, j));
        end
        
        A(:, j) = A(:, j)/aux;
%         Xs(j, :, :) = aux*Xs(j, :, :);
    end
    
    
    %% update B
    XTX = zeros(sqrtn);
    XTY = zeros(sqrtn, sqrtp);
    for m = 1:M
        aux = A*reshape(Xs(:, m), sqrtn, sqrtn);
        XTX = XTX + aux'*aux;
        XTY = XTY + aux'*reshape(Ys(:, m), sqrtp, sqrtp);
    end
    
    B = (XTX\XTY)';
    [~, msgid] = lastwarn;
    if ~isempty(msgid)
        if strcmp(msgid, 'MATLAB:nearlySingularMatrix') || ...
            strcmp(msgid, 'MATLAB:singularMatrix')
            B = (pinv(XTX)*XTY)';
        end
    end
    %% update X because of B
%     B = normc(B);
    for j = 1:sqrtn
        aux = norm(B(:, j));
        if (aux < 10e-15)
            B(:, j) = randn(sqrtp, 1);
            aux = norm(B(:, j));
        end
        
        B(:, j) = B(:, j)/aux;
%         Xs(:, j, :) = aux*Xs(:, j, :);
    end
    
    %% update sparse representations
    Q = kron(B, A);
    Xs = omp_sparse(Q'*Ys, kron(B'*B, A'*A), s);
    
    %% update error
    err_old = err_new;
    err_new = norm(Ys - Q*Xs, 'fro')/sqrt(p*N);
    
    %% to be removed
    errs = [errs err_new];
    stop = 1;
end
