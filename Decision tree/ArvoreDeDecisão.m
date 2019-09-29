clear all; %limpar os dados carregados anterior mente
close all; %feixar todas as janelas
clc; %limpar a tela 
dadosLoad = load ('dados_iris.txt');% carrega os dados do arquivo ".txt" para a variavel "dados".
nDados = size(dadosLoad,1);
nTeste = 10;
nTreino = nDados- 3*nTeste;

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

%------Gráfico das regras nos atributos 3x4
subplot(1,2,1)
hold on
title('3x4')

plot(ones(5)*2.5,(0:4),'-r');
plot(ones(3)*4.9,(0:2),'-g');
plot((0:5),ones(6)*1.65,'-g');
grid on


McTreino = zeros(3);
Ys = zeros(size(dados(:,5)));


%--------dados de "Treino"

for i = 1:size(dados,1)
    
    if dados(i,3) <= 2.5
    Ys(i) = 1;
        if(dados(i,5) == 1)
            plot(dados(i,3),dados(i,4),'.r');
        else
            if (dados(i,5) == 2)
                plot(dados(i,3),dados(i,4),'.g');
            else
                plot(dados(i,3),dados(i,4),'ob');
            end
        end
    else
        if dados(i,3)>2.5 && dados(i,3)<= 4.9 && dados(i,4)<=1.65
            Ys(i) = 2;
                if(dados(i,5) == 1)
                    plot(dados(i,3),dados(i,4),'.r');
                else
                    if (dados(i,5) == 2)
                        plot(dados(i,3),dados(i,4),'.g');
                    else
                        plot(dados(i,3),dados(i,4),'ob');
                    end
                end
        else
            Ys(i) = 3;
                if(dados(i,5) == 1)
                    plot(dados(i,3),dados(i,4),'.r');
                else
                    if (dados(i,5) == 2)
                        plot(dados(i,3),dados(i,4),'.g');
                    else
                        plot(dados(i,3),dados(i,4),'ob');
                    end
                end
        end
    end
        McTreino(dados(i,5),Ys(i)) = McTreino(dados(i,5),Ys(i))+1;
end %for
McTreino
hold off

%--------dados de Teste
McTeste = zeros(3);
YsTeste = zeros(size(dadosTeste(:,5)));

subplot(1,2,2)
hold on
title('3x4')

plot(ones(5)*2.5,(0:4),'-r');
plot(ones(3)*4.9,(0:2),'-g');
plot((0:5),ones(6)*1.65,'-g');
grid on
for i = 1:size(dadosTeste,1)
    
    if dadosTeste(i,3) <= 2.5
    YsTeste(i) = 1;
        if(dadosTeste(i,5) == 1)
            plot(dadosTeste(i,3),dadosTeste(i,4),'.r');
        else
            if (dadosTeste(i,5) == 2)
                plot(dadosTeste(i,3),dadosTeste(i,4),'.g');
            else
                plot(dadosTeste(i,3),dadosTeste(i,4),'ob');
            end
        end
    else
        if dadosTeste(i,3)>2.5 && dadosTeste(i,3)<= 4.9 && dadosTeste(i,4)<=1.65
            YsTeste(i) = 2;
                if(dadosTeste(i,5) == 1)
                    plot(dadosTeste(i,3),dadosTeste(i,4),'.r');
                else
                    if (dadosTeste(i,5) == 2)
                        plot(dadosTeste(i,3),dadosTeste(i,4),'.g');
                    else
                        plot(dadosTeste(i,3),dadosTeste(i,4),'ob');
                    end
                end
        else
            YsTeste(i) = 3;
                if(dadosTeste(i,5) == 1)
                    plot(dadosTeste(i,3),dadosTeste(i,4),'.r');
                else
                    if (dadosTeste(i,5) == 2)
                        plot(dadosTeste(i,3),dadosTeste(i,4),'.g');
                    else
                        plot(dadosTeste(i,3),dadosTeste(i,4),'ob');
                    end
                end
        end
    end
        McTeste(dadosTeste(i,5),YsTeste(i)) = McTeste(dadosTeste(i,5),YsTeste(i))+1;
end %for
McTeste
%-------calculo do erro

