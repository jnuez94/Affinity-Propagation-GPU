x_raw = fscanf(fopen('../data/short.xyz', 'r'), '%f', [5 Inf]);
x = x_raw(1:3, :)';
N = size(x, 1);
M = N*N-N;

j=1;
for i=1:N
  for k=[1:i-1,i+1:N]
    s(j,1)=i; s(j,2)=k; s(j,3)=-sum((x(i,:)-x(k,:)).^2);
    j=j+1;
  end;
end;
p=median(s(:,3)); % Set preference to median similarity
[idx,netsim,dpsim,expref]=apclusterSparse(s,p,'details','nonoise');
% fprintf('Number of clusters: %d\n',length(unique(idx)));
% fprintf('Fitness (net similarity): %f\n',netsim);
% figure; % Make a figures showing the data and the clusters
% for i=unique(idx)'
%   ii=find(idx==i)
%   h=plot(x(ii,1),x(ii,2),'o'); hold on;
%   col=rand(1,3); set(h,'Color',col,'MarkerFaceColor',col);
%   xi1=x(i,1)*ones(size(ii)); xi2=x(i,2)*ones(size(ii)); 
%   line([x(ii,1),xi1]',[x(ii,2),xi2]','Color',col);
% end;
% axis equal tight;