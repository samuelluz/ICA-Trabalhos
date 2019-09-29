clear all;
close all;
clc

Dad = importdata('data_ESPEC_1.mat');% carrega os dados do arquivo ".txt" para a variavel "dados".
dadosTotal = Dad.data1';
label = Dad.rot';
nDT = size(dadosTotal,1);

dadosN=dadosTotal(1:42,:);
dadosF=dadosTotal(43:nDT,:);
for q = 1:4
    Misturar = randperm(size(dadosN,1));
    dadosN = dadosN(Misturar,:);
for i = 1:42
    aux = 42*q+i;
    dadosC(aux,:) = dadosN(i,:);
    for j = 1:6
        ruido = (rand(1)*2-1)/1000;
        dadosC(aux,j) = dadosC(aux,j)+ruido;
    end
end
end

for r = 1:50
MatConfTreino = zeros(2);
MatConfTeste = zeros(2);
Misturar = randperm(size(dadosN,1));
dadosN = dadosN(Misturar,:);
dN = dadosC;
dNt = dadosN;
Misturar = randperm(size(dadosF,1));
dadosF = dadosF(Misturar,:);
dF = dadosF(1:214,:);
dFt = dadosF(215:size(dadosF,1),:);


%-------treinamento-----------
    mdN = mean(dN);
    mdF = mean(dF);
    vdN = var(dN);
    vdF = var(dF);
    cdN = cov(dN);
    cdF = cov(dF);
    dcdN = det(cdN);
    dcdF = det(cdF);
    PdN = size(dN,1)/nDT;
    PdF = size(dF,1)/nDT;
%------------------------------

%--------Treino-----------------
dadosTreino = [dN;dF];
labelTreino = [zeros(size(dN,1),1);ones(size(dF,1),1)];
for i = 1:size(dadosTreino,1);
X = dadosTreino(i,:);
gN = log(PdN)-(0.5*log(dcdN))-(0.5*(X-mdN)*inv(cdN)*(X-mdN)');
gF = log(PdF)-(0.5*log(dcdF))-(0.5*(X-mdF)*inv(cdF)*(X-mdF)');

if gN>gF
lSTreino(i,1) = 0;
else
lSTreino(i,1) = 1;
end
MatConfTreino(lSTreino(i)+1,labelTreino(i)+1) = MatConfTreino(lSTreino(i)+1,labelTreino(i)+1)+1;
end
AccTreino(r) = 1-(sum(abs(lSTreino-labelTreino))/size(dadosTreino,1));

if r==1
    AccTreinoMax = AccTreino(r);
    MatConfTreinoMin = MatConfTreino;
else
    if AccTreino(r)>AccTreinoMax
        AccTreinoMax = AccTreino(r);
        MatConfTreinoMin = MatConfTreino;
    end
end
%--------Teste-----------------
dadosTeste = [dNt;dFt];
labelTeste = [zeros(size(dNt,1),1);ones(size(dFt,1),1)];
for i = 1:size(dadosTeste,1);
X = dadosTeste(i,:);
gN = log(PdN)-(0.5*log(dcdN))-(0.5*(X-mdN)*inv(cdN)*(X-mdN)');
gF = log(PdF)-(0.5*log(dcdF))-(0.5*(X-mdF)*inv(cdF)*(X-mdF)');

if gN>gF
lSTeste(i,1) = 0;
else
lSTeste(i,1) = 1;
end
MatConfTeste(lSTeste(i)+1,labelTeste(i)+1) = MatConfTeste(lSTeste(i)+1,labelTeste(i)+1)+1;
end
AccTeste(r) = 1-(sum(abs(lSTeste-labelTeste))/size(dadosTeste,1));
if r==1
    AccTesteMax = AccTeste(r);
    MatConfTesteMin = MatConfTeste;
else
    if AccTeste(r)>AccTesteMax
        AccTesteMax = AccTeste(r);
        MatConfTesteMin = MatConfTeste;
    end
end
end
MediaAccTreino = mean(AccTreino)
VarAccTreino = var(AccTreino)
AccTreinoMax
MatConfTreinoMin

MediaAccTeste = mean(AccTeste)
VarAccTeste = var(AccTeste)
AccTesteMax
MatConfTesteMin
