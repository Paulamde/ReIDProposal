close all;
clear all;
clc;

AiCityFolder = './AICITY';
addpath(AiCityFolder);
names_query='track2_AicSyntestAic.txt';
name_track='test_track_id.txt';
fid=fopen(names_query);
fid2=fopen(name_track);


%% Seleccionar la query que queremos
for query_id=1:1052
     query_matches= textscan(fid,'%s',1,'delimiter','\n');%, 'headerlines',query_id-1);
    aux=(strsplit(query_matches{1,1}{1,1},' '));
    aux=aux(1:100);
    query_matches=aux(1:100);
    fid2=fopen(name_track);
    A=fgetl(fid2);
    cont=1;
    k=1;
    clear track_row Hist trackImages_notAdded track_position trackImages_notAdded...
        Hist_norm track_position track_lines infoFinal images_toadd Final_result_aux images_track...
        index index_add_images index_higher index_lower aux idx_sort images_added
    while ischar(A)
        track_row{cont}=(strsplit(A,' '));
        Hist(cont)=0;
        trackImages_notAdded{cont}=[];
        for i=1:numel(track_row{1,cont})-1
            %{1,cont}{1,i}=sprintf("%06d",str2double(track_row{1,cont}{1,i}));
            [index,aux]=find(strcmp(track_row{1,cont}{1,i},query_matches'));
            Hist(cont)=Hist(cont)+numel(find(aux==1));
            if aux
                %1-indice en top100
                %2-Id del coche (nombre de la imagen
                %3-Linea del track train
                %4-Numero total de imagenes del mismo ID en esa linea
                track_position(k,:)=[index, str2num(track_row{1,cont}{1,i}), cont, numel(track_row{1,cont})-1];
                k=k+1;
                trackImages_notAdded{cont}=[trackImages_notAdded{cont};0];
            else
                trackImages_notAdded{cont}=[trackImages_notAdded{cont}; string(track_row{1,cont}{1,i})];
            end
        end
        Hist_norm(cont)=Hist(cont)/(numel(track_row{1,cont})-1);
        A = fgetl(fid2);
        cont=cont+1;
    end
    [~,idx_sort]=sort(track_position(:,1));
    track_position=track_position(idx_sort,:);
    %% Comprobar las primeras posiciones del query_matches a qué columna pertenece, Max Min, Porcentaje...
    track_lines=unique(track_position(:,3),'stable');

    for i=1:numel(track_lines)
        [index,~]=find(track_position(:,3)==track_lines(i));
        %Lineas del track donde detectamos nuestra query
        infoFinal{i,1}=track_lines(i);
        %Mínimo valor y máximo del top100 donde aparece cada linea del track
        infoFinal{i,2}=[min(index) max(index)];
        %Porcentaje de aparición por linea
        infoFinal{i,3}=Hist(track_lines(i))/(numel(track_row{1,track_lines(i)})-1);
        %Numero de coches del track line que falta
        infoFinal{i,4}=(numel(track_row{1,track_lines(i)})-1)-Hist(track_lines(i));
        %Numero de coches que aparecen del track
        infoFinal{i,5}=Hist(track_lines(i));
        %Todos los indices del top100 donde aparece cada fila
        infoFinal{i,6}=index;
        %ID imágenes que matchea de cada linea 
        infoFinal{i,7}=track_position(index,2);

    end


%     figure
%     bar(Hist_norm)
% 
%     figure
%     bar(Hist)

    [~,index]=sort([infoFinal{:,3}],'descend');
    infoFinal(:,1)=infoFinal(index,1);
    infoFinal(:,2)=infoFinal(index,2);
    infoFinal(:,3)=infoFinal(index,3);
    infoFinal(:,4)=infoFinal(index,4);
    infoFinal(:,5)=infoFinal(index,5);
    infoFinal(:,6)=infoFinal(index,6);


    %Cogemos 4 con  mayores porcentajes

    number_lines_track=size(infoFinal,1);
    position=[infoFinal{:,2}];
    index_odd= 1:2:size(position,2);
    position=position(index_odd);
    [~,idx_sortedTop]=sort(position);
    infoFinal=infoFinal(idx_sortedTop,:);
    index_higher= 1:number_lines_track;
    im_missing=sum([infoFinal{index_higher,4}]);
    %Que imagenes añadir de arriba (sabiendo que añadimos 
    cont=0;
    i=1;
if im_missing ~= 0
    while cont<100
        %Mirar que  la linea
        if ([infoFinal{i,4}]+[infoFinal{i,5}]+cont<=100)
            
            images_toadd(i,1)=[infoFinal{i,4}];%Cuantas imagenes añadir
            images_added(i,1)=[infoFinal{i,5}];
            images_toadd(i,2)=[infoFinal{i,1}];%de que line track
            images_added(i,2)=[infoFinal{i,1}];
            cont=[infoFinal{i,4}]+[infoFinal{i,5}]+cont;
            i=i+1;
            
        elseif cont<=100
            aux=100-cont;
            if [infoFinal{i,5}]<=aux
                images_added(i,1)=[infoFinal{i,5}];%Cuantas imagenes añadir
                images_added(i,2)=[infoFinal{i,1}];%de que line track
                aux=100-cont-[infoFinal{i,5}];
                images_toadd(i,1)=aux;%Cuantas imagenes añadir
                images_toadd(i,2)=[infoFinal{i,1}];%de que line track
                aux=100-cont;
            else
                images_added(i,1)=aux;%Cuantas imagenes añadir
                images_added(i,2)=[infoFinal{i,1}];%de que line track
            end
            cont=aux+cont;
        else
            break;
        end
        
    end
    Final_result_aux=[];
     if size(images_added,1)>size(images_toadd,1)
        size_add=size(images_added,1);
        index_toadd=size(images_toadd,1);
        index_added=size(images_added,1);
        images_toadd(index_toadd+1:index_added,:)=zeros(index_added-index_toadd,2);
        images_toadd(index_toadd+1:index_added,2)=images_added(index_toadd+1:index_added,2);
    elseif size(images_added,1)<size(images_toadd,1)
        size_add=size(images_toadd,1);
        index_toadd=size(images_toadd,1);
        index_added=size(images_added,1);
        images_added(index_added+1:index_toadd,:)=zeros(index_toadd-index_added,2);
        images_added(index_added+1:index_toadd,:)=images_toadd(index_added+1:index_toadd,:);
     elseif  size(images_added,1)==size(images_toadd,1)
         size_add=size(images_added,1);
     end
     
     cont_100=0;
    for i=1:size_add
        for t=1:images_added(i,1)+images_toadd(i,1)
            cont_100=cont_100+1;
        	Final_result_aux(cont_100)=str2num(track_row{1,images_toadd(i,2)}{1,t});
            
        end
       
    end
    
elseif im_missing==0
    for t=1:100
        Final_result_aux(t,:)= str2num([query_matches{1,t}]);
    end

end
    IDquery=num2str(query_id,'%06d');
    nameID_gallery =strcat(IDquery,'.txt');
    
    fid3=fopen(nameID_gallery,'w');
    fprintf(fid3,'%d \n',Final_result_aux); 
    Final_result(query_id,:)=Final_result_aux';
    fclose(fid2);
    fclose(fid3);
end
dlmwrite('track2_AicSyntestAic2.txt', Final_result, 'delimiter', ' ', 'newline', 'pc');
% dlmwrite('track2_ranks.txt',Final_result,'Delimiter',' ');
fclose(fid);
