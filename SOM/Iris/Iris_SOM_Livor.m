clear all;
clc;
M = rand(64,2);
Q = zeros(64,1);
for t = 1:1000
X = rand(1,2); % training input
% Winner search
for i = 1:64
Q(i,1) = norm(X(1,:) - M(i,:));
end
[C,c] = min(Q);
% Updating the neighborhood
denom = 1 + t/300000; % time-dependent parameter
a = .3/denom; % learning coefficient
r = round(3/denom); % neighborhood radius
M = reshape(M,[8 8 2]);
X = reshape(X,[1 1 2]);
ch = mod(c-1,8) + 1; % c starts at top left of the
cv = floor((c-1)/8) + 1; % 2D SOM array and runs downwards!
for h = max(ch-r,1):min(ch+r,8)
for v = max(cv-r,1):min(cv+r,8)
M(h,v,:) = M(h,v,:) + a*(X(1,1,:) - M(h,v,:));
end
end

plot(M(:,:,1), M(:,:,2),'k*',M(:,1,1),M(:,1,2),'k-',M(:,2,1),M(:,2,2),'k-',M(:,3,1),M(:,3,2),'k-',M(:,4,1),M(:,4,2),'k-',M(:,5,1),M(:,5,2),'k-',M(:,6,1),M(:,6,2),'k-',M(:,7,1),M(:,7,2),'k-',M(:,8,1),M(:,8,2),'k-',M(1,:,1),M(1,:,2),'k-',M(2,:,1),M(2,:,2),'k-',M(3,:,1),M(3,:,2),'k-',M(4,:,1),M(4,:,2),'k-',M(5,:,1),M(5,:,2),'k-',M(6,:,1),M(6,:,2),'k-',M(7,:,1),M(7,:,2),'k-',M(8,:,1),M(8,:,2),'k-',M(:,3,1),M(:,3,2),'k-',M(:,4,1),M(:,4,2),'k-',0,0,'.',1,1,'.');

M = reshape(M,[64 2]);
%X = reshape(X,[t 2]);
end