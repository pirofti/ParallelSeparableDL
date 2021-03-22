% Copyright (c) 2020 Cristi Rusu <cristi.rusu.tgm@gmail.com>
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

function cI = readImages(images, thesize)

if (nargin < 2)
    thesize = 64;
    thefraction = sqrt(64);
end

thefraction = sqrt(thesize);

cI = [];
for i=1:length(images)
    disp(['processing ' char(images(i)) ' ...']);
    I = double(imread(char(images(i))));
    I = I(:, :, 1);
    [m n] = size(I);
    
%     if (n<512)
%         n = 256;
%         m = 256;
%         I = I(1:256, 1:256);
%     end
%     
%     if (n>=1200)
%         n = 1200;
%         m = 1200;
%         I = I(1:n, 1:m);
%     end
    
    newn = n*m/thesize;

    C = mat2cell(I, thefraction*ones(1, m/thefraction), thefraction*ones(1, n/thefraction));
    [mc nc] = size(C);
    I = zeros(thesize, newn);
    
    index = 0;
    for indexi = 1:mc
        for indexj = 1:nc
            
            aux = reshape(C{indexi,indexj}, 1, thesize);
            aux = aux - mean(aux);
            if (norm(aux)>=0)
                index = index + 1;
                I(:,index) = aux;
            end
        end
    end
    
    cI = [cI I(:, 1:index)];
end

% means = mean(cI);
% for i=1:length(cI)
%     cI(:, i) = cI(:, i) - means(i);
% end
% 
% themax = max(max(abs(cI)));
% cI = cI./themax;
