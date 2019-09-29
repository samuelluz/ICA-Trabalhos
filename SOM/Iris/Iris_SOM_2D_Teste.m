clear all;
clc;
c1 = rand(50,2);
c2 = rand(50,2)+1;
dados = [c1;c2];
mist = randperm(size(dados,1));
dados = dados(mist,:);

nEpocas = 10;
nNeuro = 16;
aInicial = 0.3;
aFinal = 0.1;
rInicial = 4;
rFinal = 0;

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
