
function getA(m, n)
    A = zeros(Int, n, n)
    for d in 0:m-1
        for j in 1:n-d 
            A[j, j+d] = m-d
        end
    end
    return A
end;


getA(1, 4)

function y(x, m; showstate=false)
    n, T = size(x)
    A = getA(m, n)
    aaa_ = zeros(eltype(x), n, T)      # Store all the states a in a matrix 
    aaa_[:, 1] = x[:, 1]
    for t in 2:T
        aaa_[:, t] = A * aaa_[:, t-1] + x[:, t]
    end
    if showstate 
        print("a... = ")
        show(stdout, "text/plain", aaa_)
    end
    aaa_[1, T]
end;

# Generate data
N, T = 9, 11
X = reshape((1:N*T).%T, N, T)

y(X, 2, showstate=true)

using Zygote: gradient, @ignore
M = 2
dy(x) = gradient(X_ -> y(X_, M), x)[1]
dy(X)

function y1(x, m)
    n, T = size(x)
    A = @ignore getA(m, n)     # Using @ignore macro
    a = x[:, 1]
    for t in 2:T
        a = A*a       # We are not assigning to part of an array/matrix anymore 
        a += x[:, t]
    end
    a[1]
end
dy1(x) = gradient(X_ -> y1(X_, M), x)[1];
dy1(X)