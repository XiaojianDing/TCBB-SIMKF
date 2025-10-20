function [X] = ProgressData(X)
%PROGRESS Survival analysis Data

X = table2array(X)';
X = Standard_Normalization(X);
X = standardizeCols(X);
%y = table2array(survivalTime(:,2:end));

%Data = Cox_data_processed( X,y(:,1),'censoring',~y(:,2) );

end

