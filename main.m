% This is an implementation of Nonnegative matrix factorisation which is
% proposed by Lee and Seung in the following reference:

% Learning the parts of objects by non-negative matrix factorization,
% Nature, 401, (1999).

% We applied the algorithm to a simple missing data problem.

clc;
clear;

load olivettifaces.mat
Xr = faces;
ind = find(Xr == 0);
Xr(ind) = 1;

[row,col] = size(Xr);

M = ones(row,col); % Mask for missing data, it is assumed known.

for k = 1:col
    
    missingAmount = row * (4/16);
    
    missStart = randi([1 row-missingAmount]);
    missStart = missStart - mod(missStart,64);
    M(missStart+1:missStart+missingAmount,k) = 0;
    
end

Y = M .* Xr; % Y is the dataset

[m,n] = size(Y);

r = 40; % approximation rank

W = 70*(rand(m,r)); % random initisalisation of factors
H = (rand(r,n));

for k = 1:1000
    
    W = W .* (Y * H')./((M.*(W*H)) * H');
    H = H .* (W' * Y) ./ (W' * (M .* (W*H)));
    if mod(k,50) == 0
        display(['NMF Iter: ' num2str(k)]);
    end
    
end

% nmf dict plot
z = 5;
A = [];
for j = 1:z
    A = [A; reshape(W(:,(j-1)*(r/z)+1:j*(r/z)),64,(r/z)*64)];
end

% Reconstructions
Ynmf = M.*Y + (1-M) .* (W * H);

fs = [23,34,191,71];
Ims = [reshape(Xr(:,fs(1)),64,64),reshape(Xr(:,fs(2)),64,64);
    reshape(Xr(:,fs(3)),64,64),reshape(Xr(:,fs(4)),64,64)];

Imsmiss = [reshape(Y(:,fs(1)),64,64),reshape(Y(:,fs(2)),64,64);
    reshape(Y(:,fs(3)),64,64),reshape(Y(:,fs(4)),64,64)];

Recn = [reshape(Ynmf(:,fs(1)),64,64),reshape(Ynmf(:,fs(2)),64,64);
    reshape(Ynmf(:,fs(3)),64,64),reshape(Ynmf(:,fs(4)),64,64)];

close all;

fsize = 30;
figure,imagesc(Ims);colormap gray;set(gca,'xtick',[],'fontsize',fsize);set(gca,'ytick',[]);title('Original Images');xlabel('(a)');
figure,imagesc(Imsmiss);colormap gray;set(gca,'xtick',[],'fontsize',fsize);set(gca,'ytick',[]);title('Corrupted Images');xlabel('(b)');
figure,imagesc(Recn);colormap gray;set(gca,'xtick',[],'fontsize',fsize);set(gca,'ytick',[]);title('NMF');xlabel('(e)');