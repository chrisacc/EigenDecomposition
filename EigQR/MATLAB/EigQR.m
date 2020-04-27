%%%% Eigen decomposition of symmetric matrix A
%First - tridiagonalize the input matrix
%Second - diagonalize the tridiagalized result

%test 1:
%Ain=[4 1 -2 2; 1 2 0 1; -2 0 3 -2; 2 1 -2 -1];

%test 2:
%Ain=[3 1 -4 2; 1 4 3 -1; -4 3 -2 3; 2 -1 3 2];

%test 3:
%Ain=[2875.000000 , 1762.000000 , 2671.000000 , 1498.000000 , 2221.000000 , 1762.000000 , 2677.000000 , 1978.000000 , 2113.000000 , 1462.000000 , 2671.000000 , 1978.000000 , 2659.000000 , 1510.000000 , 2005.000000 , 1498.000000 , 2113.000000 , 1510.000000 , 2551.000000 , 2086.000000 , 2221.000000 , 1462.000000 , 2005.000000 , 2086.000000 , 2677.000000]; 
%Ain=reshape(Ain,[5 5]);

%test 4:
%Ain=[115490.000000 , 20164.000000 , 29668.000000 , 86978.000000 , 77474.000000 , 58180.000000 , 67684.000000 , 48962.000000 , 39458.000000 , 96196.000000 , 105700.000000 , 10946.000000 , 20164.000000 , 98210.000000 , 90434.000000 , 43492.000000 , 51268.000000 , 67106.000000 , 59330.000000 , 74596.000000 , 82372.000000 , 36002.000000 , 28226.000000 , 105700.000000 , 29668.000000 , 90434.000000 , 84386.000000 , 47812.000000 , 53860.000000 , 66242.000000 , 60194.000000 , 72004.000000 , 78052.000000 , 42050.000000 , 36002.000000 , 96196.000000 , 86978.000000 , 43492.000000 , 47812.000000 , 74018.000000 , 69698.000000 , 60772.000000 , 65092.000000 , 56738.000000 , 52418.000000 , 78052.000000 , 82372.000000 , 39458.000000 , 77474.000000 , 51268.000000 , 53860.000000 , 69698.000000 , 67106.000000 , 61636.000000 , 64228.000000 , 59330.000000 , 56738.000000 , 72004.000000 , 74596.000000 , 48962.000000 , 58180.000000 , 67106.000000 , 66242.000000 , 60772.000000 , 61636.000000 , 63650.000000 , 62786.000000 , 64228.000000 , 65092.000000 , 60194.000000 , 59330.000000 , 67684.000000 , 67684.000000 , 59330.000000 , 60194.000000 , 65092.000000 , 64228.000000 , 62786.000000 , 63650.000000 , 61636.000000 , 60772.000000 , 66242.000000 , 67106.000000 , 58180.000000 , 48962.000000 , 74596.000000 , 72004.000000 , 56738.000000 , 59330.000000 , 64228.000000 , 61636.000000 , 67106.000000 , 69698.000000 , 53860.000000 , 51268.000000 , 77474.000000 , 39458.000000 , 82372.000000 , 78052.000000 , 52418.000000 , 56738.000000 , 65092.000000 , 60772.000000 , 69698.000000 , 74018.000000 , 47812.000000 , 43492.000000 , 86978.000000 , 96196.000000 , 36002.000000 , 42050.000000 , 78052.000000 , 72004.000000 , 60194.000000 , 66242.000000 , 53860.000000 , 47812.000000 , 84386.000000 , 90434.000000 , 29668.000000 , 105700.000000 , 28226.000000 , 36002.000000 , 82372.000000 , 74596.000000 , 59330.000000 , 67106.000000 , 51268.000000 , 43492.000000 , 90434.000000 , 98210.000000 , 20164.000000 , 10946.000000 , 105700.000000 , 96196.000000 , 39458.000000 , 48962.000000 , 67684.000000 , 58180.000000 , 77474.000000 , 86978.000000 , 29668.000000 , 20164.000000 , 115490.000000];
%Ain=reshape(Ain, [sqrt(numel(Ain)), sqrt(numel(Ain))]);

%test 5: Rs from experimental data:
%Testing with experimental data:
%Read in binary matrix
filename='/Users/chrisacconcia/Desktop/QREigen_TransferToWorkComp/test4Rs.bin'; %128x128x1332
%format Rs into row-major.. doesn't matter actually because it's symmetric
fileID = fopen(filename,'r');

Rs=fread(fileID, 'single');
fclose(fileID);
Rs=reshape(Rs,[128 128 1332]);
Ain = Rs(:,:,101);
%%
%Ain=balance(Ain);
scaling=1; max(Ain(:)); 
Ain=Ain/scaling;
%Ain=single(Ain);
%cond(Ain) %characterizes how stable the matrix is: if cond(Ain)=inf, the matrix is singular. if cond(A)=1 the matrix is very stable
%condeig(Ain)

disp('My func output:')
[Atri,Qtri] = TridiagonalizeCPU(Ain);
[Eval,Evec] = DiagonalizeShiftedCPU(Atri,Qtri);
Evec=Evec';

temp1=scaling*Ain*Evec;
temp2=scaling*Evec*Eval;

disp('my error')
sqrt(sum(sum((temp1-temp2).^2)))

disp('With matlabs eig func: ')
[V,D]=eig(Ain,'nobalance');

temp1=scaling*Ain*V;
temp2=scaling*V*D;

disp('matlab error:')
sqrt(sum(sum((temp1-temp2).^2)))

close all;
figure;plot(sort(diag(Eval)), '*b'); hold on ; plot(diag(D), '^r'); title('Eigen values using matlab eig function and my function')
%% Extra stupid stuff
% %make test case of a symmetric matrix
% test=magic(12);
% test=test*test'; %symmetric
% %%% output list for testing in cuda
% for i=1:numel(test); fprintf('%f , ',test(i));end;

%%
%     %Testing with experimental data:
%     %Read in binary matrix
%     filename='/Users/chrisacconcia/Desktop/QREigen_TransferToWorkComp/test4Rs.bin'; %128x128x1332
%     %format Rs into row-major.. doesn't matter actually because it's symmetric
%     fileID = fopen(filename,'r');
%     
%     Rs=fread(fileID, 'single');
%     fclose(fileID);
%     Rs=reshape(Rs,[128 128 1332]);
%     %single 128x128 matrix took ~20s 
%     %test how long to decompose 1000 matrices:
% tic; 
% for i=1:1000; 
%     [V,D]=eig(Rs(:,:,i)); 
% end 
% toc;
