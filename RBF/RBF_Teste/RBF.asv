clear all;
clc;
c1 = rand(50,2);
l1 = ones(50,1);
c2 = rand(50,2)+1;
l2 = ones(50,1)*-1;
dados = [c1;c2];
label = [l1;l2];
mist = randperm(size(dados,1));
Xd = dados(mist,:)';
t = label(mist,:);
P = 1;
% P = 100;
for i = 1:size(Xd,2)
    for j = 1:size(Xd,2)
        Phi(i,j) = exp(-norm(Xd(:,i)-Xd(:,j))^2);
    end
end
w = inv(Phi'*Phi)*Phi'*t;
X = zeros(2,10000);
for i=1:200
    for j=1:200
        X(:,200*(i-1)+j) = [i/100;j/100];
    end;
end;
Y = Xd;
figure; hold on;
for n=1:size(X,2)
    o = exp(-P*sum((Y-repmat(X(:,n),1,100)).^2,1))*w;
    if o<0
        plot(X(1,n),X(2,n),'b');
    else
        plot(X(1,n),X(2,n),'k.');
    end
end
plot(Y(1,:),Y(2,:),'r.');