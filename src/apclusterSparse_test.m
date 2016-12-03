data_raw = fscanf(fopen('../data/short.xyz', 'r'), '%f', [5 Inf]);
data = data_raw(1:3, :)';
N = size(data, 1);
M = N*N-N;

j=1;
for i=1:N
  for k=[1:i-1,i+1:N]
    s(j,1)=i; s(j,2)=k; s(j,3)=-sum((data(i,:)-data(k,:)).^2);
    j=j+1;
  end;
end;
p=median(s(:,3)); % Set preference to median similarity
[idx,netsim,dpsim,expref]=apclusterSparse(s,p,'details');
