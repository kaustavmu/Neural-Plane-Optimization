
lb = [0, 0, 10, 0.3, 4];
ub = [9, 9, 20, 0.5, 10];

rng default  % For reproducibility
nvars = 5;

options = optimoptions('particleswarm','SwarmSize',20,'Display','iter', 'MaxStallIterations', 100);
x = particleswarm(@fun,nvars,lb,ub,options)

function [mass] = fun(x)
    A = round(x(1));
    B = round(x(2));
    C = round(x(3));
    fileID = fopen('C:\\Users\kaust\Downloads\PSOmatlab.txt', 'w');
    fprintf(fileID,"%d\n",A);
    fprintf(fileID,"%d\n",B);
    fprintf(fileID,"%d\n",C);
    fprintf(fileID,"%f\n",x(4));
    fprintf(fileID,"%f\n",x(5));
    fclose('all');
    while 1
        if isfile('C:\\Users\kaust\Downloads\PSOmatlab_return.txt')
            pause(0.1)
            break
        end
    end
    readFileID = fopen('C:\\Users\kaust\Downloads\PSOmatlab_return.txt', 'r');
    mass = str2num(fscanf(readFileID, "%s"));
    fclose('all');
    delete('C:\\Users\kaust\Downloads\PSOmatlab_return.txt');
end
