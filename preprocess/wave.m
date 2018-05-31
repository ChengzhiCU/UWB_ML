function [ data, dist, Er, T_EMD, T_RMS, Kur,rise_time,Maxamp] = wave(wavedata_direct,plot_figure)

lengthNUM = length(wavedata_direct);
NUMi = wavedata_direct(:,3:4065);
dist = wavedata_direct(:,1);
ih8=[];
%取出实部虚部高低8位
    
        rl8 = NUMi(:,1:4:end);
        rh8 = NUMi(:,2:4:end);
        il8 = NUMi(:,3:4:end);
        ih8 = NUMi(:,4:4:end);
        ih8(:,1016) = 0;

        %求补码
        rh8(rh8>=128)=rh8(rh8>=128)-256;
        
        %%%%%
        odd_point=[150,200,350,400,450,600,650,850,900];
        for j=1:size(ih8,1)
        for i=1:9
       if( (255-ih8(j,odd_point(i)))>ih8(j,odd_point(i)))
           ih8(j,odd_point(i)) = 0;
       else
           ih8(j,odd_point(i)) = 255;
       end
        end
        end
        %%%%%
        ih8(ih8>=128)=ih8(ih8>=128)-256;
           
        
        %组合成16位数据
        rr=rh8*256+rl8;
        ii=ih8*256+il8;
        data=rr+ii*1i;
        data=abs(data);
        [m,n] = size(data);
        if(plot_figure == 1)
            for i =1:m
                figure(i);
                plot(data(i,:));
                axis([0 1200 0 10000]);
            end
        end

        %%%%%%%%%%%%
%取一部分点
     %记录峰值点
%         peak=find(data == max(data));
%         totalCP=0;
%         for j = 1 : length(data)
%             totalCP = totalCP + data(j)^2;
%         end
%         Q(i) = 10*log10((data(peak)^2+data(peak+1)^2+data(peak+2)^2)/totalCP);
        %截取21个点
 
        %作截取点和原波形的对比图
    %     datashi(735:755)=data(735:755);
    %     dataxu=data;
    %     dataxu(735:755)=0;
    %     plot(datashi,'-');
    %     hold on
    %     plot(dataxu,'--');
%%%%%%%%%%%
%max amplitude
 [Maxamp,Maxpos] = max(data');
Maxamp = Maxamp';
Maxpos = Maxpos';
%rise time
rise_time=[];

for i = 1:size(data,1)
    try
  mean_n = mean(data(i,1:Maxpos(i)-100)')';
sigma_n = std(data(i,1:Maxpos(i)-100)')';
  
rise_t1 = find ( (data(i,:) > 6*(sigma_n+mean_n)));
rise_t2 = find( (data(i,:) >  0.6*Maxamp(i) ));
if(isempty(rise_t1)) rise_t1=0;end
if(isempty(rise_t2)) rise_t2=0;end
rise_time(i) = rise_t2(1)-rise_t1(1);
rise_win = rise_t1(end)-rise_t1(1);

    catch
        rise_time(i)=-1000;
    end
end
rise_time=rise_time';
%window
data_w=[];
for i = 1:size(data,1)
        if(dist(i)==0)
            data_w(i,:) = zeros(1,36);
        else
            try
       data_w(i,:)=data(i,Maxpos(i)-20:Maxpos(i)+15);
        [m_w,n_w] = size(data_w); 
            catch
                data_w(i,:)=zeros(1,36);
            end
            
    end
end
%Energy
 
 Er =  sum(power(data_w(:,1:n_w),2),2);
 
 %mean excess delay
 fhi=[];
 T_EMD=[];
 T_RMS=[];
 for i = 1:m_w
     data_sq = power(data_w(:,1:n_w),2);
     fhi(i,:) = data_sq(i,:) / Er(i);
     for j = 1:n_w
     T_EMD(i,1) = sum( j * fhi(i,j),2);
     T_RMS(i,1) = sum( (j-T_EMD(i))^2 * fhi(i,j),2);
     end
 end
 
%kurtosis
     mu=[];
     sigma=[];
     Kur=[];

        for k = 1:m_w
            mu(k,1) = sum(data_w(k,:))/n_w;
        end
     
               
        for k = 1 : m_w
            sigma2(k,1) = sum((data_w(k,:)-mu(k,1)).^2) / n_w;
        end
        
      
        for k = 1 : m_w
            Kur(k,1) = sum((data_w(k,:)-mu(k,1)).^4) / (sigma2(k,1)^2*n_w) ;
        end


    
end