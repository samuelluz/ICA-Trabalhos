%Imput:
%   dados - Banco de dados para treinar a rede
%   dTeste - Banco de dados para treinar a rede
%   nEpocas - Numero de epocas de treinamento
%   nNeuro - Numero de Neuronios
%   aInicial - Taxa de aprendisagem inicial
%   aFinal - Taxa de aprendizagem final
%   rInicial - Visinhanca(neighborhood) inicial
%   rFinal - Visinhanca final
%
function  [Mmin,Emin]  =  SOM(dados,dTeste,nEpocas,nNeuro,aInicial,aFinal,rInicial,rFinal)


nAtribut = size(dados,2);
side = sqrt(nNeuro);%lado da matriz quadrada de neuronios
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
        Es(t) = norm(X(1,:) - M(c,:));
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
% % --------plot-----------------------------     
% %         figure(3)
% %        title('Distribuicão dos dados em 2D');
% %         xlabem('atributo 1');
% %         ylabem('atributo 2');
% %         hold on
% %         plot(M(:,:,1), M(:,:,2),'k*');
% %         for l = 1:side
% %            plot(M(:,l,1), M(:,l,2),'r-') 
% %         end
% %         for l = 1:side
% %            plot(M(l,:,1), M(l,:,2),'b-') 
% %         end
% %         hold off
% % -----------------------------------------
         M = reshape(M,[nNeuro nAtribut]);
    end
    
    Erros(e) =  ErroQ(dados,M);
    ErrosTest(e) =  ErroQ(dTeste,M);
    
    if e == 1
    Emin = Erros(e) + ErrosTest(e);
    Mmin = M;
    eMin = e;
    else
        if Emin > Erros(e) + ErrosTest(e);
            Emin = Erros(e) + ErrosTest(e);
            Mmin = M;
            eMin = e;
        end
    end
    
end
figure(1)
%title('Erro de quantizacao X Epoca');
plot(Erros,'b')
hold on
plot(ErrosTest,'r')
plot(eMin,0.002,'*g')
hold off
saveas(gcf,'ErroQuant.jpg');