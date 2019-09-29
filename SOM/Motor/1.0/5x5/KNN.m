% apenas p/K =1;
function [labelM Ec] = KNN(dados,dadosTeste,M,K)
nAtribut = size(dados,2)-1;
label = dados(:,nAtribut+1);
dados = dados(:,1:nAtribut);
labelTeste = dadosTeste(:,nAtribut+1);
dadosTeste = dadosTeste(:,1:nAtribut);
nDados = size(dados,1);
nNeuro = size(M,1);
nDadosTeste = size(dadosTeste,1);

for n = 1:nNeuro
    for i = 1:nDados
        Q(i) = norm(M(n,:)-dados(i,:));
    end
    if K == 1
        [D,m] = min(Q);
        labelM(n,1) = label(m);
    else
        for i = 1:size(Q)
        end
    end
end

for t = 1:nDadosTeste
    X = dadosTeste(t,:);
    for i = 1:nNeuro
        P(i,1) = norm(X(1,:) - M(i,:));
    end
    [C,c] = min(P);
    labelTesteM(t) = labelM(c);
end
E = abs(labelTeste-labelTesteM');
Acertos = find(E == 0);
Ec = (nDadosTeste - size(Acertos,1))/nDadosTeste;
Ec = Ec*100;