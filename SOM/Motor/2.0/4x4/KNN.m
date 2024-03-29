% apenas p/K =1;
function [labelM Ac MatC] = KNN(dados,dadosTeste,M,nClasses,K)
nAtribut = size(dados,2)-1;
label = dados(:,nAtribut+1);
dados = dados(:,1:nAtribut);
labelTeste = dadosTeste(:,nAtribut+1);
dadosTeste = dadosTeste(:,1:nAtribut);
nDados = size(dados,1);
nNeuro = size(M,1);
nDadosTeste = size(dadosTeste,1);

for n = 1:size(M,1)
    for i = 1:nDados
        Q(i) = norm(M(n,:)-dados(i,:));
    end
    [Qsort,Min] = sort(Q);
    Mk = zeros(nClasses,1);
    Klabel = label(Min(1:K));
    for i = 1:K
    Mk(Klabel(i)+1) = Mk(Klabel(i)+1)+1;
    end
    [D,Max] = max(Mk);
    Kiguais = find(Mk==Mk(Max));
    
    if size(Kiguais,1) == 1
        labelM(n,1) = Max-1;
    else
        [D,Min] = min(Q);
        labelM(n,1) = label(Min);
    end
%     end
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
Ac = 1-((nDadosTeste - size(Acertos,1))/nDadosTeste);
Ac = Ac*100;
labelTesteM = labelTesteM';
MatC = zeros(nClasses);
for i = 1:nDadosTeste
MatC(labelTesteM(i)+1,labelTeste(i)+1) = MatC(labelTesteM(i)+1,labelTeste(i)+1)+1;
end