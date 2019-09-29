clear all;
close all;
clc

Dad = importdata('data_ESPEC_1.mat');% carrega os dados do arquivo ".txt" para a variavel "dados".
dadosTotal = Dad.data1';
labelTotal = Dad.rot';
nDT = size(dadosTotal,1);

numTreino = 32;
numTeste = 10;
numClass = 7;

MatConfTreino = zeros(numClass);
MatConfTeste = zeros(numClass);
for r = 1:50
    for i=0:numClass-1
        aux = i*(numTreino+numTeste)+1;
        aux2 = (i+1)*(numTreino+numTeste);
        dados = dadosTotal(aux:aux2,:);
        label = labelTotal(aux:aux2,:);
        
        mist = randperm(numTreino+numTeste);
        dados = dados(mist,:);
        label = label(mist,:);
        
        if i==0
            dadosTreino = dados(1:numTreino,:);
            dadosTeste = dados(numTreino+1:numTreino+numTeste,:);
            labelTreino = label(1:numTreino,:);
            labelTeste = label(numTreino+1:numTreino+numTeste,:);
        else
            dadosTreino = [dadosTreino;dados(1:numTreino,:)];
            dadosTeste = [dadosTeste;dados(numTreino+1:numTreino+numTeste,:)];
            labelTreino = [labelTreino;label(1:numTreino,:)];
            labelTeste = [labelTeste;label(numTreino+1:numTreino+numTeste,:)];
        end
    end
    
% 
%-------treinamento-----------
    i = 0;
    aux = i*(numTreino)+1;
    aux2 = (i+1)*(numTreino);
    md0 = mean(dadosTreino(aux:aux2,:));
    vd0 = var(dadosTreino(aux:aux2,:));
    cd0 = cov(dadosTreino(aux:aux2,:));
    dcd0 = det(cd0);
    Pd0 = numTreino/size(dadosTreino,1);
    
    i = 1;
    aux = i*(numTreino)+1;
    aux2 = (i+1)*(numTreino);
    md1 = mean(dadosTreino(aux:aux2,:));
    vd1 = var(dadosTreino(aux:aux2,:));
    cd1 = cov(dadosTreino(aux:aux2,:));
    dcd1 = det(cd1);
    Pd1 = numTreino/size(dadosTreino,1);
    
    i = 2;
    aux = i*(numTreino)+1;
    aux2 = (i+1)*(numTreino);
    md2 = mean(dadosTreino(aux:aux2,:));
    vd2 = var(dadosTreino(aux:aux2,:));
    cd2 = cov(dadosTreino(aux:aux2,:));
    dcd2 = det(cd2);
    Pd2 = numTreino/size(dadosTreino,1);
    
    i = 3;
    aux = i*(numTreino)+1;
    aux2 = (i+1)*(numTreino);
    md3 = mean(dadosTreino(aux:aux2,:));
    vd3 = var(dadosTreino(aux:aux2,:));
    cd3 = cov(dadosTreino(aux:aux2,:));
    dcd3 = det(cd3);
    Pd3 = numTreino/size(dadosTreino,1);
    
    i = 4;
    aux = i*(numTreino)+1;
    aux2 = (i+1)*(numTreino);
    md4 = mean(dadosTreino(aux:aux2,:));
    vd4 = var(dadosTreino(aux:aux2,:));
    cd4 = cov(dadosTreino(aux:aux2,:));
    dcd4 = det(cd4);
    Pd4 = numTreino/size(dadosTreino,1);
    
    i = 5;
    aux = i*(numTreino)+1;
    aux2 = (i+1)*(numTreino);
    md5 = mean(dadosTreino(aux:aux2,:));
    vd5 = var(dadosTreino(aux:aux2,:));
    cd5 = cov(dadosTreino(aux:aux2,:));
    dcd5 = det(cd5);
    Pd5 = numTreino/size(dadosTreino,1);
    
    i = 6;
    aux = i*(numTreino)+1;
    aux2 = (i+1)*(numTreino);
    md6 = mean(dadosTreino(aux:aux2,:));
    vd6 = var(dadosTreino(aux:aux2,:));
    cd6 = cov(dadosTreino(aux:aux2,:));
    dcd6 = det(cd6);
    Pd6 = numTreino/size(dadosTreino,1);
    
    ErroTreino = 0;
for i = 1:size(dadosTreino,1);

    X = dadosTreino(i,:);
    g(1) = log(Pd0)-(0.5*log(dcd0))-(0.5*(X-md0)*inv(cd0)*(X-md0)');
    g(2) = log(Pd1)-(0.5*log(dcd1))-(0.5*(X-md1)*inv(cd1)*(X-md1)');
    g(3) = log(Pd2)-(0.5*log(dcd2))-(0.5*(X-md2)*inv(cd2)*(X-md2)');
    g(4) = log(Pd3)-(0.5*log(dcd3))-(0.5*(X-md3)*inv(cd3)*(X-md3)');
    g(5) = log(Pd4)-(0.5*log(dcd4))-(0.5*(X-md4)*inv(cd4)*(X-md4)');
    g(6) = log(Pd5)-(0.5*log(dcd5))-(0.5*(X-md5)*inv(cd5)*(X-md5)');
    g(7) = log(Pd6)-(0.5*log(dcd6))-(0.5*(X-md6)*inv(cd6)*(X-md6)');
    
    [D, m] = max(g);
    lSTreino(i,1) = m-1;
    
    MatConfTreino(lSTreino(i)+1,labelTreino(i)+1) = MatConfTreino(lSTreino(i)+1,labelTreino(i)+1)+1;
    if (lSTreino(i) - labelTreino(i))~= 0
    ErroTreino  = ErroTreino+1;
    end
end
ErroTeste = 0;
for i = 1:size(dadosTeste,1);

    X = dadosTeste(i,:);
    g(1) = log(Pd0)-(0.5*log(dcd0))-(0.5*(X-md0)*inv(cd0)*(X-md0)');
    g(2) = log(Pd1)-(0.5*log(dcd1))-(0.5*(X-md1)*inv(cd1)*(X-md1)');
    g(3) = log(Pd2)-(0.5*log(dcd2))-(0.5*(X-md2)*inv(cd2)*(X-md2)');
    g(4) = log(Pd3)-(0.5*log(dcd3))-(0.5*(X-md3)*inv(cd3)*(X-md3)');
    g(5) = log(Pd4)-(0.5*log(dcd4))-(0.5*(X-md4)*inv(cd4)*(X-md4)');
    g(6) = log(Pd5)-(0.5*log(dcd5))-(0.5*(X-md5)*inv(cd5)*(X-md5)');
    g(7) = log(Pd6)-(0.5*log(dcd6))-(0.5*(X-md6)*inv(cd6)*(X-md6)');
    
    [D, m] = max(g);
    lSTeste(i,1) = m-1;
    
    MatConfTeste(lSTeste(i)+1,labelTeste(i)+1) = MatConfTeste(lSTeste(i)+1,labelTeste(i)+1)+1;
    if (lSTeste(i) - labelTeste(i))~= 0
    ErroTeste = ErroTeste+1;
    end
end
AccTreino(r) = (size(labelTreino,1) - ErroTreino)/size(labelTreino,1);
AccTeste(r) = (size(labelTeste,1) - ErroTeste)/size(labelTeste,1);
if r==1
    AccTreinoMax = AccTreino(r);
    MatConfTreinoMin = MatConfTreino;
    
    AccTesteMax = AccTeste(r);
    MatConfTesteMin = MatConfTeste;
else
    if AccTreino(r)>AccTreinoMax
        AccTreinoMax = AccTreino(r);
        MatConfTreinoMin = MatConfTreino;
    end
    if AccTeste(r)>AccTesteMax
        AccTesteMax = AccTeste(r);
        MatConfTesteMin = MatConfTeste;
    end
end

end
MediaAccTreino = mean(AccTreino)
VarAccTreino = var(AccTreino)
AccTreinoMax
MatConfTesteMin

MediaAccTeste = mean(AccTeste)
VarAccTeste = var(AccTeste)
AccTesteMax
MatConfTesteMin