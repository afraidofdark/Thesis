function [val, deg, dep] = nonmaxsup(A, n) % Adjusted to work between angles [1:180]
if mod(n, 2) == 0
    n = n + 1;
end
mid = ceil(n / 2);
wnd = (1:n) - mid;

l = length(A);
%asearch = [A((l-1) - (mid-1:-1:1)), A, A(2:mid)]; % Circular
asearch = [ones(mid-1,1)'*A(1), A, ones(mid-1,1)'*A(end)]; % Linear

val = [];
deg = [];
dep = [];
for i = mid : l + mid - 1
%     clf;
%     plot(asearch, "r");
%     hold on;
%     a = i + wnd;
%     scatter(a(1:mid-1), asearch(a(1:mid-1)));
%     hold on;
%     scatter(a(mid+1:end), asearch(a(mid+1:end)));
%     hold on;
%     scatter(i, asearch(i), "*");
%     pause(0.1);
    if min(asearch(i + wnd)) == asearch(i)
        val = [val, asearch(i)];
        deg = [deg, i - mid + 1];
        dep = [dep, (max(asearch(i + wnd)) - asearch(i))];
    end
end
end