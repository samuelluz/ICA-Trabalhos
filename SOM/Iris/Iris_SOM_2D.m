clear all; %limpar os dados carregados anterior mente
close all; %feixar todas as janelas
clc; %limpar a tela 
dadosLoad = load ('dados_iris.txt');% carrega os dados do arquivo ".txt" para a variavel "dados".

nDados = size(dadosLoad,1);
nTeste = 15;
nTreino = nDados- 3*nTeste;
nAtributos = size(dadosLoad,2)-1;
nEpocas = 500;

nNeuronios = 9;
Raio =1;
TaxaInicial = 0.1;
TaxaFinal = 0.01;
Vinicial = 0.5;
Vfinal = 0.01;
%----------normaliza��o
for i = 1:nDados
    dadosLoad(i,1:4) = dadosLoad(i,1:4)/sqrt(sum(dadosLoad(i,1:4).^2));
end

setosa = dadosLoad(1:50,:);% selecional os valores das linhas de 1 a 50 e da coluna 1 a 4 (correspondem aos dados da setosa)
versicolor = dadosLoad(51:100,:);% selecional os valores das linhas de 51 a 100 e da coluna 1 a 4 (correspondem aos dados da versicolor)
virginica = dadosLoad(101:150,:);% selecional os valores das linhas de 101 a 150 e da coluna 1 a 4 (correspondem aos dados da virginica)


Misturar=randperm(50);
setosaM = setosa(Misturar,:);
Misturar=randperm(50);
versicolorM = versicolor(Misturar,:);
Misturar=randperm(50);
virginicaM = virginica(Misturar,:);

if nTeste == 0
    Misturar = randperm(150);
    dados = dadosLoad(Misturar,:);
else
    Misturar = randperm(nTreino);
    dadosTeste = [setosaM(1:nTeste,:);versicolorM(1:nTeste,:);virginicaM(1:nTeste,:)];
    dados= [setosaM(nTeste+1:50,:);versicolorM(nTeste+1:50,:);virginicaM(nTeste+1:50,:)];
    dados = dados(Misturar,:);
end


M = 1:nNeuronios;
lado = sqrt(nNeuronios)
M = reshape(M,[lado lado]);
M=M'
Taxa = TaxaInicial;
V = Vinicial;

for i = 1:nNeuronios
    %W(i,:) = dados(1,1:4);
    W(i,:) = mean(dados(:,1:4));
    %W(i,:) = median(dados(:,1:4));
end

for epoca = 1:nEpocas
    Misturar = randperm(nTreino);
    dados = dados(Misturar,:);
X = dados(:,1:4);
%--------Inicializa os pesos

for j = 1:nTreino
         uMax = W*X(j,:)';
         [a,p] = max(uMax);

r = Raio*((1/Raio)^(j/(nTreino*nEpocas)));
ch = mod(p-1,size(M,1)) + 1; % c starts at top left of the
cv = floor((p-1)/size(M,1)) + 1; % 2D SOM array and runs downwards!
for h = max(ch-r,1):min(ch+r,size(M,1))
for v = max(cv-r,1):min(cv+r,size(M,1))
i = M(h,v,:);
W(i,:) = W(i,:)+Taxa*(X(j,:)-W(i,:));
end
end
         %Taxa = TaxaInicial*(1-(j/nTreino));
         %Taxa = TaxaInicial/(1+j);
         Taxa = TaxaInicial*((TaxaFinal/TaxaInicial)^(j/(nTreino*nEpocas)));

figure(1)
title('3x4')
plot(setosa(:,3),setosa(:,4),'.r');
hold on
plot(versicolor(:,3),versicolor(:,4),'.g');
plot(virginica(:,3),virginica(:,4), 'ob');
plot(W(:,3),W(:,4),'-oy');
grid on
hold off         
end

end
%---Gr�fico dos atributos 1x2

figure(2)
title('3x4')
plot(setosa(:,1),setosa(:,2),'.r');
hold on
plot(versicolor(:,1),versicolor(:,2),'.g');
plot(virginica(:,1),virginica(:,2), 'ob');
plot(W(:,1),W(:,2),'-oy');
grid on
hold off

