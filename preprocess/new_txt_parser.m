clc;
clear;
close all;
%  LL = dir(['/Users/mcz/Downloads/data_UWB/raw/斜一墙/', '*txt']); 
LL = dir(['/home/maocz/data/raw/斜一墙/', '*txt']); 
% LL = dir(['/home/maocz/data/raw/新坐标系/', '*txt']); 
% LL = dir(['/home/maocz/data/raw/视距/', '*txt']); 

len_list = length(LL);

for i = 1:len_list
    filename = LL(i).name
    folder_path = LL(i).folder; 
    string_name = [folder_path, '/',filename];
    
    % save_path = ['/Users/mcz/Downloads/data_UWB/mat_files/', filename(1:end-4), '.mat'];
    save_path = ['/home/maocz/data/mat_data_6f/', filename(1:end-4), '.mat'];
    if exist(save_path, 'file') == 2
        continue
    end
    
    name_len = length(filename);
    int_part = 0;
    float_part = 0;
    int_flag=0;
    for i = 1:name_len
        alpha = filename(i);
        if alpha == 'D'
            int_flag=1;
            
            continue;
        end
        
        if alpha == '_' && int_flag==1
            int_flag=-1;
            continue;
        end
        
        if int_flag == 1
            int_part = int_part * 10 + str2num(alpha);
            continue;
        end
        
        if int_flag == -1 && alpha == '.'
            break
        end
        
        if int_flag == -1
            % come to float part
            float_part = float_part * 10 + str2num(alpha);
            continue;
        end
    end
    dis = int_part + float_part * 1.0 / 100;    
    
    if dis == []
        disp('error file name');
    else
        % data_read = load(string_name);
        fid = fopen(string_name);
        tline = fgetl(fid);
        ff = fopen('t.txt','wt');
        fprintf(ff, tline);
        fclose(ff);
        try
            data_read_line = load('t.txt');
            data_read = [];
            if size(data_read_line, 2) == 4065
                data_read = data_read_line;
            end
        catch
            data_read = [];
        end
        

        while ischar(tline)
            tline = fgetl(fid);
            if tline ~= -1
                ff = fopen('t.txt','wt');
                fprintf(ff, tline);
                fclose(ff);
                try
                    data_read_line = load('t.txt');
        %             delete t.txt
                    if size(data_read_line, 2) == 4065
                        data_read = [data_read; data_read_line];
                    end
                end
            end
        end
        len_wave = size(data_read,1);
        % save  log_20180306_nlosc_a1b1c1.mat data_read;
        plot_figure = 0;    
        [data_t, dist_esti, Er_t, T_EMD_t, T_RMS_t, Kur_t, Rise_time, Maxamp] = wave(data_read, plot_figure);

        all_wave = [];
        all_feature = [];

        combined_feature = [dist_esti, Er_t, T_EMD_t, T_RMS_t, Kur_t, Rise_time, Maxamp];

        % remove the wrong data
        for i = 1:size(dist_esti, 1)
            if dist_esti(i) < 30
                all_wave = [all_wave; data_t(i, :)];
                all_feature = [all_feature; combined_feature(i, :)];
            else
    %             disp('error dis');
                dist_esti(i);
            end
        end
        %

        data.wave = all_wave;
        data.feature = all_feature;

        data_num = size(all_feature,1);
        groundtruth = ones(data_num, 1) * dis;
        data.groundtruth = groundtruth;
        save(save_path, 'data');
    end
    % plot(data_t(1,:));
    
end