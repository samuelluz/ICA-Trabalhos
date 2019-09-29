% X = Vetor de entrada
% d = saida desejada (escalar)
% W = Matriz de pesos Entrada -> Camada Oculta
% M = Matriz de Pesos Camada Oculta -> Camada saida
% eta = taxa de aprendizagem
% alfa = fator de momento

clear; clc; close all;

%% Preparar Dados

Dad = importdata('dados_iris.txt');% carrega os dados do arquivo ".txt" para a variavel "dados".
dadosBrut = Dad(:,1:4);
Label = Dad(:,5);
% Embaralha vetores de entrada e saidas desejadas
[LinD ColD]=size(dadosBrut);

% Normaliza componentes para media zero e variancia unitaria
for i=1:LinD,
	mi=mean(dadosBrut(i,:),2);  % Media das linhas
        di=std(dadosBrut(i,:));   % desvio-padrao das linhas 
	dadosBrut(i,:)= (dadosBrut(i,:) - mi)./di;
end 
dadosTotal = [dadosBrut Label];
nAtribut = size(dadosTotal,2)-1;
% LIMITES DOS EIXOS X e Y (superficie de decisao)
%Xmin=min(Dn(1,:)); Xmax=max(Dn(1,:));
%Ymin=min(Dn(2,:)); Ymax=max(Dn(2,:));

% Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn=10;    % numero de dados para treino

% DEFINE ARQUITETURA DA REDE
%===========================
Ne = 100; % No. de epocas de treinamento
Nr = 5;   % No. de rodadas de treinamento/teste
Nh = 10;   % No. de neuronios na camada oculta
No = 3;   % No. de neuronios na camada de saida
valInitOcult=1; % No. de variacao dos pesos iniciais
valInitSaida=1;
eta=0.1;   % Passo de aprendizagem
mom=0.00;  % Fator de momento


%% Inicio do Treino
for r=1:Nr,  % LOOP DE RODADAS TREINO/TESTE
contTotal=zeros(1,No);
contClasses=zeros(1,No);
    Rodada=r

%     I=randperm(ColD);
%     Dn=Dn(:,I);
%     alvos=alvos(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

    nTeste=ptrn;
    
    
    Misturar=randperm(50);
    A0 = dadosTotal(1:50,:);
    A0 = A0(Misturar,:);
    A1 = dadosTotal(51:100,:);
    A1 = A1(Misturar,:);
    A2 = dadosTotal(101:150,:);
    A2 = A2(Misturar,:);

    if nTeste == 0
    else
    dTeste = [A0(1:nTeste,:);A1(1:nTeste,:);A2(1:nTeste,:)];
    dados= [A0(nTeste+1:42,:);A1(nTeste+1:42,:);A2(nTeste+1:42,:)];
    Misturar = randperm(size(dados,1));
    dados = dados(Misturar,:);
    labelDados = dados(:,nAtribut+1);
    labelTeste = dTeste(:,nAtribut+1);
    dados = dados(:,1:nAtribut);
    dTeste = dTeste(:,1:nAtribut);
    end
    labelDados = labelDados';
    labelTeste = labelTeste';
    
    alvos = zeros(3,size(labelDados,2));
    
    for rr = 1:size(labelDados,2)
        alvos(labelDados(rr),rr) = 1;
    end
    
    alvosTeste = zeros(3,size(labelTeste,2));
    
    for rr = 1:size(labelTeste,2)
        alvosTeste(labelTeste(rr),rr) = 1;
    end
    
    
    % Vetores para treinamento e saidas desejadas correspondentes
    P = dados'; T1 = alvos;
    [lP cP]=size(P);   % Tamanho da matriz de vetores de treinamento

    % Vetores para teste e saidas desejadas correspondentes
    Q = dTeste'; T2 = alvosTeste;
    [lQ cQ]=size(Q);   % Tamanho da matriz de vetores de teste

    % Inicia matrizes de pesos
    WW=2*valInitOcult*rand(Nh,lP+1)-valInitOcult;   % Pesos entrada -> camada oculta
    WW_old=WW;              % Necessario para termo de momento

    MM=2*valInitSaida*rand(No,Nh+1)-valInitSaida;   % Pesos camada oculta -> camada de saida
    MM_old = MM;            % Necessario para termo de momento
    
    %%% ETAPA DE TREINAMENTO
    for t=1:Ne,   % Inicia LOOP de epocas de treinamento
        Epoca=t;
        confusionMatrix=zeros(No,No);
        I=randperm(cP); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento
        
        EQ=0;
        HID1=[];
        for tt=1:cP,   % Inicia LOOP de iteracoes em uma epoca de treinamento
            % CAMADA OCULTA
            X  = [-1; P(:,tt)];   % Constroi vetor de entrada com adicao da entrada x0=-1
            Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
            Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)
            HID1=[HID1 Yi];

            % CAMADA DE SAIDA
            Y  = [-1; Yi];        % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
            Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
            Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)

            % CALCULO DO ERRO
            Ek = T1(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede
            EQ = EQ + 0.5*sum(Ek.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA

            %%% CALCULO DOS GRADIENTES LOCAIS
            Dk = Ok.*(1 - Ok);  % derivada da sigmoide logistica (camada de saida)
            DDk = Ek.*Dk;       % gradiente local (camada de saida)

            Di = Yi.*(1 - Yi); % derivada da sigmoide logistica (camada oculta)
            DDi = Di.*(MM(:,2:end)'*DDk);    % gradiente local (camada oculta)

            % AJUSTE DOS PESOS - CAMADA DE SAIDA
            MM_aux=MM;
            MM = MM + eta*DDk*Y' + mom*(MM - MM_old);
            MM_old=MM_aux;

            % AJUSTE DOS PESOS - CAMADA OCULTA
            WW_aux=WW;
            WW = WW + eta*DDi*X' + mom*(WW - WW_old);
            WW_old=WW_aux;
        end   % Fim de uma epoca
        %HID1, pause;
        
        EQM(t)=EQ/cP;  % MEDIA DO ERRO QUADRATICO POR EPOCA
    end   % Fim do loop de treinamento
    
    figure(1); plot(EQM);  % Plota Curva de Aprendizagem
    xlabel('Épocas de treinamento');  ylabel('Erro Quadrático Médio (EQM)');
    title('Curva de Aprendizagem MLP')

%% ETAPA DE GENERALIZACAO  %%%
    EQTreino=0; OutTreino=[];
    for tt=1:cP,
        % CAMADA OCULTA
        X=[-1; P(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)

        % CAMADA DE SAIDA
        Y=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
        OutTreino=[OutTreino Ok];       % Armazena saida da rede

        % ERRO QUADRATICO GLOBAL (todos os neuronios) POR VETOR DE ENTRADA
        EQTreino = EQTreino + 0.5*sum(Ek.^2);
    end

    % MEDIA DO ERRO QUADRATICO COM REDE TREINADA (USANDO DADOS DE TESTE)
    EQMTreino=EQTreino/cP;

    % CALCULA TAXA DE ACERTO
    count_OK_Treino=0;  % Contador de acertos
    for t=1:cQ,
        [T1max iT1max]=max(T1(:,t));  % Indice da saida desejada de maior valor
        [OutTreino_max iOutTreino_max]=max(OutTreino(:,t)); % Indice do neuronio cuja saida eh a maior
        if iT1max==iOutTreino_max,   % Conta acerto se os dois indices coincidem
            count_OK_Treino=count_OK_Treino+1;
            contClassesTreino(iT1max)=contClassesTreino(iT1max)+1;
        end
        
       contTotalTreino(iT1max)=contTotalTreino(iT1max)+1;    
        confusionMatrixTreino(iT1max,iOutTreino_max)= confusionMatrixTreino(iT1max,iOutTreino_max)+1;
    end

    % Taxa de acerto global
    Tx_OK_Treino(r)=count_OK_Treino/cP;
    Tx_Classes_Treino(r,:)=100*contClassesTreino./contTotalTreino;
    confusionMatrixTreino

    %% ETAPA DE GENERALIZACAO  %%%
    EQ2=0; HID2=[]; OUT2=[];
    for tt=1:cQ,
        % CAMADA OCULTA
        X=[-1; Q(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)

        % CAMADA DE SAIDA
        Y=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
        OUT2=[OUT2 Ok];       % Armazena saida da rede

        % ERRO QUADRATICO GLOBAL (todos os neuronios) POR VETOR DE ENTRADA
        EQ2 = EQ2 + 0.5*sum(Ek.^2);
    end

    % MEDIA DO ERRO QUADRATICO COM REDE TREINADA (USANDO DADOS DE TESTE)
    EQM2=EQ2/cQ;

    % CALCULA TAXA DE ACERTO
    count_OK=0;  % Contador de acertos
    for t=1:cQ,
        [T2max iT2max]=max(T2(:,t));  % Indice da saida desejada de maior valor
        [OUT2_max iOUT2_max]=max(OUT2(:,t)); % Indice do neuronio cuja saida eh a maior
        if iT2max==iOUT2_max,   % Conta acerto se os dois indices coincidem
            count_OK=count_OK+1;
            contClasses(iT2max)=contClasses(iT2max)+1;
        end
        
       contTotal(iT2max)=contTotal(iT2max)+1;    
        confusionMatrix(iT2max,iOUT2_max)= confusionMatrix(iT2max,iOUT2_max)+1;
    end

    % Taxa de acerto global
    Tx_OK(r)=count_OK/cQ;
    Tx_Classes(r,:)=100*contClasses./contTotal;
    confusionMatrix

end % FIM DO LOOP DE RODADAS TREINO/TESTE

% Tx_minima=min(Tx_OK);
% Tx_maxima=max(Tx_OK);
% Tx_media=mean(Tx_OK),  % Taxa media de acerto global
% Tx_std=std(Tx_OK), % Desvio padrao da taxa media de acerto 
% 
% Tx_Classes_min=min(Tx_Classes);
% Tx_Classes_std=std(Tx_Classes);
% Tx_Classes_max=max(Tx_Classes);
% Tx_Classes_mean=mean(Tx_Classes);
% 
% %Escrita dos resultados no arquivo
% nIterations=Nr;
% fid = fopen('resultadoMLP.txt', 'wt');
% 
% nClasses=No;
% maxTaxa=100*max(Tx_OK);
% fprintf(fid, '%6.2f \t', maxTaxa);
% minTaxa=100*min(Tx_OK);
% fprintf(fid, '%6.2f \t', minTaxa);
% medTaxa=100*mean(Tx_OK);
% fprintf(fid, '%6.2f \t', medTaxa);
% sum=100*Tx_OK-medTaxa;
% sum=sum.*sum;
% soma=0;
% for i=1:nIterations
%     soma=soma+sum(i);
% end
% soma=soma/nIterations;
% desvioPadrao=sqrt(soma)
% fprintf(fid, '%6.2f \t', desvioPadrao);
% fprintf(fid,'\n');
% for i=1:nClasses
%     for j=1:nClasses
%     fprintf(fid, '%6.2f \t', confusionMatrix(i,j));
%     end
%     fprintf(fid,'\n');
% end
%  fprintf(fid,'\n');
%  fprintf(fid,'\n');
% 
% 
% fprintf(fid,'\n');
% medTaxaClass=mean(Tx_Classes)
% fprintf(fid, '%6.2f \t', medTaxaClass);
% 
% fprintf(fid,'\n');
% maxTaxaClass=max(Tx_Classes)
% fprintf(fid, '%6.2f \t', maxTaxaClass);
% 
% fprintf(fid,'\n');
% minTaxaClass=min(Tx_Classes)
% fprintf(fid, '%6.2f \t', minTaxaClass);
% 
% for i=1:nClasses
% 
%     
%  sumClass=Tx_Classes(:,i)-medTaxaClass(i);
%  sumClass=sumClass.*sumClass  ;
%  soma=0;
% for j=1:nIterations
%     soma=soma+sumClass(j);
% end
% 
% soma=soma/nIterations;
% desvioPadraoClass(i)=sqrt(soma);
% end
%   fprintf(fid,'\n');
% fprintf(fid, '%6.2f \t', desvioPadraoClass);
% h=figure(1);
% saveas(h,'MLP.eps','psc2');
% fclose(fid)