%% Date generation for CEC17

%% for normal function and hybrid function
%% From F1-F3: Unimodal Functions
%% From F4-F10: Multimodal Functions 
%% From F11-F20: Hybrid Functions

for func_num=1:20 
    OShift=-80+rand(1,100)*160;
    eval(['save shift_data_' num2str(func_num) '.txt OShift -ASCII -DOUBLE']);
end

%for composition function: cf_num=10
for func_num=21:30
    OShift=-80+rand(10,100)*160;
    a=[-80:16:80];
    for i=1:100
        for j=1:10
            OShift(j,i)=a(j)+rand*16;
        end
        tmp=randperm(10);
        OShift(:,i)=OShift(tmp,i);
    end
    
    %% Remove this line as it is the reason behind catching one of the local optima easily for some algos
%     OShift(3,:)=zeros(1,100);
    eval(['save shift_data_' num2str(func_num) '.txt OShift -ASCII -DOUBLE']);
end

%%%%%%%%%%%%%%%%%%%%%%%%
%% M matrix for normal function (Unimodal + Multimodal)
%Matrix for D=2
for func_num=1:10
    D=2;
    M=[];c=1;
    M=rot_matrix(D,c);
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M -ASCII -DOUBLE']);
end
%Matrix for =[10,30,50,100];
func_num=23;
D_choose=[10,30,50,100];
D=10;
for func_num=1:10;
    M=diag(ones(1,D));
    G=[3,3,4];
    C=[1,2,1];
    j=1;
    for i=1:length(G)
        M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C(i));
        j=G(i)+j;
    end

    S=randperm(D);
    M0=M(S,S);
    [tmp,SS]=sort(S);
    
    dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
end
D=20;
for func_num=1:10;
    M=diag(ones(1,D));
    G=[3,3,4];
    C=[1,2,1];
    j=1;
    for i=1:length(G)
        M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C(i));
        j=G(i)+j;
    end

    S=randperm(D);
    M0=M(S,S);
    [tmp,SS]=sort(S);
    
    dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
end
D=30;
for func_num=1:10;
    M=diag(ones(1,D));
    G=[2,3,4,5,7,9];
    C=[1,2,1,2,1,2];
    j=1;
    for i=1:length(G)
        M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C(i));
        j=G(i)+j;
    end

    S=randperm(D);
    M0=M(S,S);
    [tmp,SS]=sort(S);
    
    dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
end
D=50;
for func_num=1:10;
    M=diag(ones(1,D));
    G=[3,4,5,6,6,8,8,10];
    C=[1,2,1,2,1,2,1,2];
    j=1;
    for i=1:length(G)
        M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C(i));
        j=G(i)+j;
    end

    S=randperm(D);
    M0=M(S,S);
    [tmp,SS]=sort(S);
    
    dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
end
D=100;
for func_num=1:10;
    M=diag(ones(1,D));
    G=[6,6,8,8,10,10,12,12,14,14];
    C=[1,2,1,2,1,2,1,2,1,2];
    j=1;
    for i=1:length(G)
        M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C(i));
        j=G(i)+j;
    end

    S=randperm(D);
    M0=M(S,S);
    [tmp,SS]=sort(S);
    
    dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%% From F21-F30: Composition Functions
%M matrix for composition function
%D=2;
    D=2;
for func_num=21:30
    M=[];c=1;
    for j=1:10
        M0=rot_matrix(D,c);
    M=[M;M0];
         if j==3
            M = rot_matrix(2,3);
        end
    end
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M -ASCII -DOUBLE']);
end
%10,30,50,100
D_choose=[10,20,30,50,100];
for k=1:5
    D=D_choose(k);

for func_num=21:28  %% All except composition functions of hybrid ones
    M10=[];SS10=[];
    for j=1:10
        M=diag(ones(1,D));
        G=[6,6,8,8,10,10,12,12,14,14];
        C=[1,2,1,2,1,2,1,2,1,2];
        j=1;
        for i=1:length(G)
            M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C(i));
            j=G(i)+j;
        end
        if j==3
            M = rot_matrix(D,3);
        end
        S=randperm(D);
        M0=M(S,S);
        [tmp,SS]=sort(S);
        M10=[M10;M0];
        SS10=[SS10,SS];
    end
    dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS10, '\t');
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M10 -ASCII -DOUBLE']);
end
end
%%%
D_choose=[10,20,30,50,100];
for k=1:4
    D=D_choose(k);

for func_num=24:24  %% Why F24?? Ask
    M10=[];
    for j=1:10

        M= rot_matrix(D,1);
            
        M10=[M10;M];
;
    end

    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M10 -ASCII -DOUBLE']);
end
end
%%%%%%%%%%%
D_choose=[10,30,50,100];
for k=1:4
    D=D_choose(k);

for func_num=29:30  %% Yes, those for composition functions of hybrid ones
    M10=[];SS10=[];
    for j=1:10
        M=diag(ones(1,D));
        G=[6,6,8,8,10,10,12,12,14,14];
        C=[1,2,1,2,1,2,1,2,1,2];
        j=1;
        for i=1:length(G)
            M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C(i));
            j=G(i)+j;
        end
        S=randperm(D);
        M0=M(S,S);
        [tmp,SS]=sort(S);
        M10=[M10;M0];
        SS10=[SS10,SS];
    end
    dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS10, '\t');
    eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M10 -ASCII -DOUBLE']);
end
end

%%%
func_num=23;  %% Why F23?? Ask 
D=100;
cf_num=10;    %% Why cf_num = 10, we don't have 10 sub-problems in composition functions! Ask
M=[];c=1;
for i=1:cf_num
%M=[M;rot_matrix(D,c)];
M=[M;orthm_generator(D)];
end
eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M -ASCII -DOUBLE']);


%%%%%%%%%%%%
%% For Hybrid functions: F11-F20

%% hf01,hf02, hf03 ----> N=3
cf_num=3;
D_choose=[10,30,50,100];
for func_num=11:13
    if func_num == 11
         P=[0.2,0.4,0.4];
    end 
    if func_num == 12
         P=[0.3,0.3,0.4];
    end 
    if func_num == 13
         P=[0.3,0.3,0.4];
    end 
    for k=1:4
        D=D_choose(k);
        M=diag(ones(1,D));
%         P=[0.3,0.3,0.4];
        G(1:cf_num-1)=ceil(P(1:cf_num-1)*D);
        G(cf_num)=D-sum(G(1:cf_num-1));
%         C=[1,1,1,1]; %% Change Noor, Ask as it gives error! C should be
%         of length D!

        C = randi([1,2],1,1);
        j=1;
        for i=1:length(G)
            M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C);
            j=G(i)+j;
        end

        S=randperm(D);
        M0=M(S,S);
        [tmp,SS]=sort(S);

        dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
        eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
    end
end
%%%%%%%%%%%%%%

%% hf04,hf05, hf06 ----> N=4
cf_num=4;
D_choose=[10,30,50,100];
for func_num=14:16
    if func_num == 14
         P=[0.2,0.2,0.2,0.4];
    end 
    if func_num == 15
         P=[0.2,0.2,0.3,0.3];
    end 
    if func_num == 16
         P=[0.2,0.2,0.3,0.3];
    end 
    for k=1:4
        D=D_choose(k);
        M=diag(ones(1,D));
%         P=[0.2,0.2,0.3,0.3];
        G(1:cf_num-1)=ceil(P(1:cf_num-1)*D);
        G(cf_num)=D-sum(G(1:cf_num-1));
%         C=[1,2,1,2];
        C = randi([1,2],1,1);
        j=1;
        for i=1:length(G)
            M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C);
            j=G(i)+j;
        end

        S=randperm(D);
        M0=M(S,S);
        [tmp,SS]=sort(S);

        dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
        eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
    end
end
%%%%%%%%%%%%%%

%% hf07,hf08, hf09 ----> N=5
cf_num=5;
D_choose=[10,30,50,100];
for func_num=17:19
    if func_num == 17
         P=[0.1,0.2,0.2,0.2,0.3];
    end 
    if func_num == 18
         P=[0.2,0.2,0.2,0.2,0.2];
    end 
    if func_num == 19
         P=[0.2,0.2,0.2,0.2,0.2];
    end 
    for k=1:4
        D=D_choose(k);
        M=diag(ones(1,D));
%         P=[0.1,0.2,0.2,0.2,0.3];
        G(1:cf_num-1)=ceil(P(1:cf_num-1)*D);
        G(cf_num)=D-sum(G(1:cf_num-1));
%         C=[1,2,1,2,1];
        C =  randi([1,2],1,1);
        j=1;
        for i=1:length(G)
            M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C);
            j=G(i)+j;
        end

        S=randperm(D);
        M0=M(S,S);
        [tmp,SS]=sort(S);

        dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
        eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
    end
end


%% hf0109 ----> N=6
cf_num=6;
D_choose=[10,20,30,50,100];
for func_num=20:20
    for k=1:5
        D=D_choose(k);
        M=diag(ones(1,D));
        P=[0.1,0.1,0.2,0.2,0.2,0.2];
        G(1:cf_num-1)=ceil(P(1:cf_num-1)*D);
        G(cf_num)=D-sum(G(1:cf_num-1));
%         C=[1,2,1,2,1,2];
        C = randi([1,2],1,1);
        j=1;
        for i=1:length(G)
            M(j:j+G(i)-1,j:j+G(i)-1)= rot_matrix(G(i),C);
            j=G(i)+j;
        end

        S=randperm(D);
        M0=M(S,S);
        [tmp,SS]=sort(S);

        dlmwrite(strcat('shuffle_data_',char(num2str(func_num)),'_D',char(num2str(D)),'.txt'), SS, '\t');
        eval(['save M_' num2str(func_num) '_D' num2str(D) '.txt M0 -ASCII -DOUBLE']);
    end
end
