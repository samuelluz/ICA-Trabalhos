clear all;
clc;
c1 = rand(50,2);
l1 = ones(50,1);
c2 = rand(50,2)+1;
l2 = ones(50,1)+1;
dados = [c1;c2];
label = [l1;l2];
mist = randperm(size(dados,1));
dados = dados(mist,:);
label = label(mist,:);

% DEFINE ARQUITETURA DA REDE
%=====================================================
% SOM
nEpocas = 10;
nNeuro = 4;
aInicial = 0.3;
aFinal = 0.1;
rInicial = 4;
rFinal = 0;
% Saida
Ne = 1000; % No. de epocas de treinament
No = 2;   % No. de neuronios na camada de saida
Nh = nNeuro;
valInitSaida=0.1;% No. de variacao dos pesos iniciais
eta=0.1;   % Passo de aprendizagem
%======================================================
side = sqrt(nNeuro);%lado da matriz quadrada de neuronios
nAtribut = size(dados,2);
for i = 1:nNeuro
M(i,:) = mean(dados);
end
Q = zeros(nNeuro,1);
nInter = 0;
for e = 1:nEpocas
    mist = randperm(size(dados,1));
    dados = dados(mist,:); 
    for t = 1:size(dados,1)
        X = dados(t,:); % training input
        % Winner search
            for i = 1:nNeuro
                Q(i,1) = norm(X(1,:) - M(i,:));
            end
        [C,c] = min(Q);
        % Updating the neighborhood
        nInter = nInter+1;
        a = aInicial*((aFinal/aInicial)^(nInter/(size(dados,1)*nEpocas)));
        r = round(rInicial*(((rFinal+0.01)/rInicial)^(nInter/(size(dados,1)*nEpocas))));
        M = reshape(M,[side side nAtribut]);
        X = reshape(X,[1 1 nAtribut]);
        ch = mod(c-1,side) + 1; % c starts at top left of the
        cv = floor((c-1)/side) + 1; % 2D SOM array and runs downwards!
        for h = max(ch-r,1):min(ch+r,side)
            for v = max(cv-r,1):min(cv+r,side)
                M(h,v,:) = M(h,v,:) + a*(X(1,1,:) - M(h,v,:));
            end
        end
        figure(1)
        plot(dados(:,1), dados(:,2),'og');
        hold on
        plot(M(:,:,1), M(:,:,2),'k*');
        for l = 1:side
           plot(M(:,l,1), M(:,l,2),'r-') 
        end
        for l = 1:side
           plot(M(l,:,1), M(l,:,2),'b-') 
        end
        hold off
        M = reshape(M,[nNeuro nAtribut]);
    end
end
for i = 1:size(M,1)
    for j = 1:size(M,1)
        dM(i,j) = norm(M(i)-M(j));
    end
end
dM = max(dM);
dM = max(a);
    spread = dM/sqrt(2*nNeuro);
for i = 1:size(dados,1)
    for j = 1:size(M,1)
        Yi(j,i) = exp(-(norm(dados(i)-M(j))^2)/(2*spread^2));
    end 
end
for i = 1:size(dados,1)
Yi(:,i)=Yi(:,i)/sum(Yi(:,i));
end

labelDados = label';
 
alvos = zeros(2,size(labelDados,2));
 
for rr = 1:size(labelDados,2)
    alvos(labelDados(rr),rr) = 1;
end  
    
    % Vetores para treinamento e saidas desejadas correspondentes
P = dados; T1 = alvos;
[lP cP]=size(P');
MM=2*valInitSaida*rand(No,Nh+1)-valInitSaida;
for t=1:Ne
    EQ=0;
    Epoca=t;
    for tt=1:cP   % Inicia LOOP de iteracoes em uma epoca de treinamento

            % CAMADA DE SAIDA
            Y  = [-1; Yi(:,tt)];        % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
            Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida (eq 7.2)
            Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)(eq 7.4)

            % CALCULO DO ERRO
            Ek = T1(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede (eq 7.7)
            EQ = EQ + 0.5*sum(Ek.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA (eq 7.9)

            %%% CALCULO DOS GRADIENTES LOCAIS
            Dk = Ok.*(1 - Ok);  % derivada da sigmoide logistica (camada de saida)(eq 7.13)
            DDk = Ek.*Dk;       % gradiente local (camada de saida)(eq 7.12)

            % AJUSTE DOS PESOS - CAMADA DE SAIDA
            MM = MM + eta*DDk*Y';

     end   % Fim de uma epoca
        %HID1, pause;
        
     EQM(t)=EQ/cP;  % MEDIA DO ERRO QUADRATICO POR EPOCA
end   % Fim do loop de treinamento

    figure(2); plot(EQM);  % Plota Curva de Aprendizagem
    xlabel('Épocas de treinamento');  ylabel('Erro Quadrático Médio (EQM)');
    title('Curva de Aprendizagem MLP')
    
 q = 1;
for i = 0:0.1:10
    for j = 0:0.1:10
        J(q,:) = [i,j];
        q = q+1;
    end
end
figure(3);
hold on;
for i = 1:size(J,1)
    for j = 1:size(M,1)
        o(j) = exp(-(norm(J(i)-M(j))^2)/(2*spread^2));
    end
    oo = [-1 o];
    Uk  = MM*oo';
    Ok = 1./(1+exp(-Uk));
    [a,b] = max(Ok);
    if b==1
        plot(J(i,1),J(i,2),'b.');
    else
        plot(J(i,1),J(i,2),'r.');
    end
end