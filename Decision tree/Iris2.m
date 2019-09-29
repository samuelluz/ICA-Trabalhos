% Análise de dados para escolha da mehor topologia
clear all; %limpar os dados carregados anterior mente
close all; %feixar todas as janelas
clc; %limpar a tela 
dados = load ('dados_iris.txt');% carrega os dados do arquivo ".txt" para a variavel "dados".
nDados = size(dados,1);
nTeste = 15;

setosa = dados(1:50,:);% selecional os valores das linhas de 1 a 50 e da coluna 1 a 4 (correspondem aos dados da setosa)
versicolor = dados(51:100,:);% selecional os valores das linhas de 51 a 100 e da coluna 1 a 4 (correspondem aos dados da versicolor)
virginica = dados(101:150,:);% selecional os valores das linhas de 101 a 150 e da coluna 1 a 4 (correspondem aos dados da virginica)

% Misturar=randperm(50);
% setosaM = setosa(Misturar,:);
% Misturar=randperm(50);
% versicolorM = versicolor(Misturar,:);
% 
% Misturar=randperm(50);
% virginicaM = virginica(Misturar,:);
% 
% dadosTeste = [setosaM(1:nTeste,:);versicolorM(1:nTeste,:);virginicaM(1:nTeste,:)];
% dadosTreno = [setosaM(nTeste:50,:);versicolorM(nTeste:50,:);virginicaM(nTeste:50,:)];


%------Gráfico dos atributos 1x2
subplot(4,3,1)
hold on
title('1x2')
plot(setosa(:,1),setosa(:,2),'.r');
plot(versicolor(:,1),versicolor(:,2),'.g');
plot(virginica(:,1),virginica(:,2),'ob');
grid on
hold off

%------Gráfico dos atributos 1x3
subplot(4,3,2)
hold on
title('1x3')
plot(setosa(:,1),setosa(:,3),'.r');
plot(versicolor(:,1),versicolor(:,3),'.g');
plot(virginica(:,1),virginica(:,3),'ob');
grid on
hold off

%------Gráfico dos atributos 1x4
subplot(4,3,3)
hold on
title('1x4')
plot(setosa(:,1),setosa(:,4),'.r');
plot(versicolor(:,1),versicolor(:,4),'.g');
plot(virginica(:,1),virginica(:,4),'ob');
grid on
hold off

%------Gráfico dos atributos 2x3
subplot(4,3,4)
hold on
title('2x3')
plot(setosa(:,2),setosa(:,3),'.r');
plot(versicolor(:,2),versicolor(:,3),'.g');
plot(virginica(:,2),virginica(:,3),'ob');
grid on
hold off

%------Gráfico dos atributos 2x4
subplot(4,3,5)
hold on
title('2x4')
plot(setosa(:,2),setosa(:,4),'.r');
plot(versicolor(:,2),versicolor(:,4),'.g');
plot(virginica(:,2),virginica(:,4),'ob');
grid on
lineX = 0:5;
%lineY = 0.49*lineX+0.28;
lineY = 1.55*lineX-2.9;
plot(lineX,lineY,'-r');
hold off
hold off

%------Gráfico dos atributos 3x4
subplot(4,3,6)
hold on
title('3x4')
plot(setosa(:,3),setosa(:,4),'.r');
plot(versicolor(:,3),versicolor(:,4),'.g');
plot(virginica(:,3),virginica(:,4),'ob');

plot(ones(5)*2.5,(0:4),'-r');
plot(ones(5)*4.95,(0:4),'-g');
plot(ones(5)*5.2,(0:4),'-b');
plot((0:10),ones(11)*1.6,'-g');
plot((0:10),ones(11)*1.9,'-b');
grid on
hold off

%------Clacudo dos Limiares

Ys = zeros(size(dados(:,5)));
p=1;
for i = 1:150
    
    if dados(i,3) <= 2.5
    Ys(i) = 1;
    else
        if dados(i,3)>2.5 && dados(i,3)<= 4.9 && dados(i,4)<=1.65
        Ys(i) = 2;
        else
            if dados(i,3)>= 5.2 || dados(i,4)>=1.9
            Ys(i) = 3;
            else
                Ys(i) = 3;
                dados2(p,:) = dados(i,:);
                p =p+1;
            end
        end
    end

end %for6.0000    2.7000    5.1000    1.6000    2.0000


versicolor2 = dados2(1:2,:);% selecional os valores das linhas de 51 a 100 e da coluna 1 a 4 (correspondem aos dados da versicolor)
virginica2 = dados2(3:size(dados2,1),:);% selecional os valores das linhas de 101 a 150 e da coluna 1 a 4 (correspondem aos dados da virginica)

%------Gráfico dos atributos 1x2
subplot(4,3,7)
hold on
title('1x2')
plot(versicolor2(:,1),versicolor2(:,2),'.g');
plot(virginica2(:,1),virginica2(:,2),'ob');
grid on
hold off

%------Gráfico dos atributos 1x3
subplot(4,3,8)
hold on
title('1x3')
plot(versicolor2(:,1),versicolor2(:,3),'.g');
plot(virginica2(:,1),virginica2(:,3),'ob');
grid on
hold off

%------Gráfico dos atributos 1x4
subplot(4,3,9)
hold on
title('1x4')
plot(versicolor2(:,1),versicolor2(:,4),'.g');
plot(virginica2(:,1),virginica2(:,4),'ob');
grid on
hold off

%------Gráfico dos atributos 2x3
subplot(4,3,10)
hold on
title('2x3')
plot(versicolor2(:,2),versicolor2(:,3),'.g');
plot(virginica2(:,2),virginica2(:,3),'ob');
grid on
hold off

%------Gráfico dos atributos 2x4
subplot(4,3,11)
hold on
title('2x4')
plot(versicolor2(:,2),versicolor2(:,4),'.g');
plot(virginica2(:,2),virginica2(:,4),'ob');
grid on

lineX = 2.5:3.5;
%lineY = 0.49*lineX+0.28;
lineY = 1.55*lineX-2.9;
plot(lineX,lineY,'-r');
hold off

%------Gráfico dos atributos 3x4
subplot(4,3,12)
hold on
title('3x4')
plot(versicolor2(:,3),versicolor2(:,4),'.g');
plot(virginica2(:,3),virginica2(:,4),'ob');
grid on
hold off

%-------calculo do erro
Yf = abs(dados(:,5)-Ys);
Erro = sum(Yf)

AcertoTotal = (1-(Erro/150))*100
