function  E  =  ErroQ(dados,M)

for t = 1:size(dados,1)
    X = dados(t,:);
    nNeuro = size(M);
    for i = 1:nNeuro
        Q(i,1) = norm(X(1,:) - M(i,:));
    end
    [C,c] = min(Q);
    Es(t) = norm(X(1,:) - M(c,:));
end
E = sum(Es)/size(dados,1);