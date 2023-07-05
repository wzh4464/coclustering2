function [Nout Iout Jout] = findindex(I, Ngroup)

    Nout = [];
    Iout = [];
    Jout = [];
    N = length(I);
    Nc = zeros(1, N);
    Ic = [];
    Jc = [];
    a = [2:Ngroup];
    aa = a .^ 2;
    Nc(1:aa(1)) = a(1);

    for i = 2:length(a)
        Ntemp = sum(aa(1:i - 1));
        Nc(Ntemp + 1:Ntemp + aa(i)) = a(i);
    end

    for i = 2:length(a) + 1

        for j = 1:i
            Ic = [Ic ones(1, i) * j];
            Jc = [Jc [1:i]];
        end

    end

    for i = 1:N
        Nout = [Nout Nc(I(i))];
        Iout = [Iout Ic(I(i))];
        Jout = [Jout Jc(I(i))];
    end
