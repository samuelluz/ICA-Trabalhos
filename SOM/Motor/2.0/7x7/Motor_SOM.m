% Análise de dados para escolha da mehor topologia
clear all; %limpar os dados carregados anterior mente
close all; %feixar todas as janelas
clc; %limpar a tela 
%format long


%inicializar variáveis---------------------------
Arquivo = 'teste7x7.txt';
savefile = 'Mmin7x7.mat';
imgSave = 'Dados2D7x7.jpg';
nR = 50;
nE = 100;
nN = 64;
nC = 7;
aI = 0.3;
aF = 0.01;
rI = 3;
rF = 0;
K = 3;
%------------------------------------------------


Dad = importdata('data_ESPEC_1.mat');% carrega os dados do arquivo ".txt" para a variavel "dados".
dadosTotal = Dad.data1';
label = Dad.rot';
dadosTotal = dadosTotal*1000;
dadosTotal = [dadosTotal label];
%Normalização---------------------------
% media = mean(dadosTotal(:,:));
% %----------normalização
% for i = 1:size(dadosTotal,2)
%     dadosTotal(:,i) = dadosTotal(:,i)-media(:,i)/var(dadosTotal(:,i));
% end
% %------------------------
%------------------------------------------

nTeste = 10;
nAtribut = size(dadosTotal,2)-1;
side = sqrt(nN);%lado da matriz quadrada de neuronios
for r = 1:nR
    r
    %-------------------------------------------------
    Misturar=randperm(42);
    SF = dadosTotal(1:42,:);
    SF = SF(Misturar,:);
    A1 = dadosTotal(43:84,:);
    A1 = A1(Misturar,:);
    A2 = dadosTotal(85:126,:);
    A2 = A2(Misturar,:);
    A3 = dadosTotal(127:168,:);
    A3 = A3(Misturar,:);
    B1 = dadosTotal(169:210,:);
    B1 = B1(Misturar,:);
    B2 = dadosTotal(211:252,:);
    B2 = B2(Misturar,:);
    B3 = dadosTotal(253:294,:);
    B3 = B3(Misturar,:);

if nTeste == 0
else
    dTeste = [SF(1:nTeste,:);A1(1:nTeste,:);A2(1:nTeste,:);A3(1:nTeste,:);B1(1:nTeste,:);B2(1:nTeste,:);B3(1:nTeste,:)];
    dados= [SF(nTeste+1:42,:);A1(nTeste+1:42,:);A2(nTeste+1:42,:);A3(nTeste+1:42,:);B1(nTeste+1:42,:);B2(nTeste+1:42,:);B3(nTeste+1:42,:)];
    Misturar = randperm(size(dados,1));
    dados = dados(Misturar,:);
    labelDados = dados(:,nAtribut+1);
    labelTeste = dTeste(:,nAtribut+1);
    dados = dados(:,1:nAtribut);
    dTeste = dTeste(:,1:nAtribut);
end
    %--------SOM------------------------------------- 
    [M,E(r)] = SOM(dados,dTeste,nE,nN,aI,aF,rI,rF);
    %------------------------------------------------
    if r == 1
    Emin = E(r);
    Mmin = M;
    
%      %--------plot-----------------------------  
%         nAtribut = size(dados,2);
%         M = reshape(M,[side side nAtribut]);
%         figure(2)
%         %title('Distribuicão dos dados em 2D');
%         plot(dados(:,1), dados(:,2),'og');
%         hold on
%         plot(dTeste(:,1), dTeste(:,2),'.y');
%         plot(M(:,:,1), M(:,:,2),'k*');
%         for l = 1:side
%            plot(M(:,l,1), M(:,l,2),'r-') 
%         end
%         for l = 1:side
%            plot(M(l,:,1), M(l,:,2),'b-') 
%         end
%         hold off
%         saveas(gcf,'Dados2D.jpg');
%         M = reshape(M,[nN nAtribut]);
%     %-----------------------------------------  
    
    
    else
        if Emin > E(r);
            Emin = E(r);
            Mmin = M;
            
% %      --------plot-----------------------------  
%         M = reshape(M,[side side nAtribut]);
%         figure(2)
%         title('Distribuicão dos dados em 2D')
%         plot(dados(:,1), dados(:,2),'og');
%         hold on
%         plot(dTeste(:,1), dTeste(:,2),'.y');
%         plot(M(:,:,1), M(:,:,2),'k*');
%         for l = 1:side
%            plot(M(:,l,1), M(:,l,2),'r-') 
%         end
%         for l = 1:side
%            plot(M(l,:,1), M(l,:,2),'b-') 
%         end
%         hold off
%         saveas(gcf,'Dados2D.jpg');
%          M = reshape(M,[nN nAtribut]);
% %     -----------------------------------------  
            
        end
    end
end

%------KNN------------------------------
[labelM Ac MatC] = KNN([dados labelDados],[dTeste labelTeste],Mmin,nC,K);
%----------------------------------------- 

VarErro = var(E);
MedErro = mean(E);
figure(3)
title('Variancia do Erro de quantizacao');
plot(1,E,'ob');
hold on;
plot(1,MedErro,'*r');
plot(1,Emin,'*g');
hold off;
saveas(gcf,'VarErroQuant.jpg')

%----arquivar---------------------------------
fid = fopen(Arquivo,'wt');
fprintf(fid,Arquivo);
fprintf(fid,'\n');
fprintf(fid,'Pre-processamento: multiplicar todos os dados por 1000 \n');
fprintf(fid,'----------TOPOLOGIA:\n');
fprintf(fid,'Dados para treinamento: %f\n',size(dados,1));
fprintf(fid,'Dados para teste: %f\n',nTeste);
fprintf(fid,'Numero de rodadas: %f\n',r);
fprintf(fid,'nEpocas: %f\n',nE);
fprintf(fid,'nNeuro: %f\n',nN);
fprintf(fid,'nClasses: %f\n',nC);
fprintf(fid,'aInicial: %f\n',aI);
fprintf(fid,'aFinal: %f\n',aF);
fprintf(fid,'rInicial: %f\n',rI);
fprintf(fid,'rFinal: %f\n',rF);
fprintf(fid,'K: %f\n',K);
fprintf(fid,'----------RESULTADOS:\n');
fprintf(fid,'Variância do erro de quantização: %f\n',VarErro);
fprintf(fid,'Erro de quantização medio: %f\n',MedErro);
fprintf(fid,'Menor Erro de quantização: %f\n',Emin);
fprintf(fid,'Acerto KNN: %f%%\n',Ac);
fprintf(fid,'\n*(Neuronio sobre Neuronio)\n*M = Mmin.mat\n');

labelM = reshape(labelM,[side side]);
fprintf(fid,'\nLabel Matricial M =\n');
for a = 1:side
    fprintf(fid,'%f\t',labelM(a,:));
    fprintf(fid,'\n');
end
fprintf(fid,'\nMatriz Confusão=\n');
for a = 1:nC
    fprintf(fid,'%f\t',MatC(a,:));
    fprintf(fid,'\n');
end
save(savefile, 'Mmin')

MatC
Ac
fclose(fid);


figure(1)
hold on
for i = 1:side
    for j = 1:side
        if labelM(i,j) == 0
            plot(i,j,'*b')
        end
        if labelM(i,j) == 1
            plot(i,j,'og')
        end
        if labelM(i,j) == 2
            plot(i,j,'og')
        end
        if labelM(i,j) == 3
            plot(i,j,'oy')
        end
        if labelM(i,j) == 4
            plot(i,j,'oy')
        end
        if labelM(i,j) == 5
            plot(i,j,'or')
        end
        if labelM(i,j) == 6
            plot(i,j,'or')
        end
        if labelM(i,j) == 7
            plot(i,j,'.k')
        end
    end
end
saveas(gcf,imgSave);