clear all; %limpar os dados carregados anterior mente
close all; %feixar todas as janelas
clc; %limpar a tela 
dados = load ('dados_iris.txt');% carrega os dados do arquivo ".txt" para a variavel "dados".


setosa = dados(1:50,1:4);% selecional os valores das linhas de 1 a 50 e da coluna 1 a 4 (correspondem aos dados da setosa)
versicolor = dados(51:100,1:4);% selecional os valores das linhas de 51 a 100 e da coluna 1 a 4 (correspondem aos dados da versicolor)
virginica = dados(101:150,1:4);% selecional os valores das linhas de 101 a 150 e da coluna 1 a 4 (correspondem aos dados da virginica)

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
hold off

%------Gráfico dos atributos 3x4
subplot(4,3,6)
hold on
title('3x4')
plot(setosa(:,3),setosa(:,4),'.r');
plot(versicolor(:,3),versicolor(:,4),'.g');
plot(virginica(:,3),virginica(:,4),'ob');

plot(ones(5)*2.5,(0:4),'-r');
plot(ones(5)*4.83,(0:4),'-r');
% plot((0:10),ones(11),'-r');
grid on
hold off

%------Gráfico dos atributos 3
subplot(4,3,7:9)
hold on
title('3')
plot(setosa(:,3),1,'or');
plot(versicolor(:,3),1,'og');
plot(virginica(:,3),1,'.b');

plot(ones(3)*2.5,(0:2),'-r');
plot(ones(3)*4.7,(0:2),'-r');
grid minor
hold off

%------Clacudo dos Limiares
Xd = dados(:,3:4);
Yd = dados(:,5);
Ys = zeros(size(Yd));

Limiar1 = 2.5;
p = 1;
for j = 2.5:0.01:7;
    Limiar2(p) = j;
for i = 1:150
    
    if Xd(i,1) <= Limiar1
    Ys(i) = 1;
    else
        if Xd(i,1)<= Limiar2(p)
        Ys(i) = 2;
        else
            Ys(i) = 3;
        end
    end

end %for
Yf = abs(Yd-Ys);
Erro(p) = sum(Yf);

if p == 1
ErroMin = Erro(p);
else
    if Erro(p)< ErroMin
    ErroMin = Erro(p);
    LimiarMin = Limiar2(p);
    pMin = p;
    end
end

p=p+1;
end
LimiarMin
ErroTotal = (1-(ErroMin/150))*100
%------Gráfico Erro X Limiar
subplot(4,3,10:12)
hold on
plot(Limiar2,Erro,'-b');
plot(LimiarMin,Erro(pMin),'or');
grid 
