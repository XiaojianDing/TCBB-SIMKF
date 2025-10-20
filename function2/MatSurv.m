function [varargout] = MatSurv(TimeVar, EventVar, GroupVar, varargin)
% USAGE:
%   MatSurv(TimeVar, EventVar, GroupVar,'param', value, ...) creates a Kaplan-Meier plot,
%   a risk table and calculates a log rank p-value
%
%   [p] = MatSurv( ... ) returns the log rank p-value
%   [p, fh] = MatSurv( ... ) returns both p-value and figure handle
%   [p, fh, stats] = MatSurv( ... ) returns additions stats from log rank test
%   [p, fh, stats] = MatSurv([], [], []) loads test dataset
%
% INPUTS:
% * 'TimeVar' is a vector with numeric time to event, either observed or
%   censored. Values less than zero will be removed by default
%
% * 'EventVar' is a vector or cell array defining events or censored
%   observation. Events are defined with a 1 and censored point with a 0. By
%   default 'Dead', 'Deceased', 'Relapsed', 'Yes', 'Event' 'Progression' &
%   'Progressed' are considered as events.
%   'Alive', 'Living', 'Not Relapsed', 'DiseaseFree', 'No' 'NoEvent'
%   'Censored' 'NoProgression' are considers as censored
%   'EventDefinition' can be used to define other types of events
%
% * 'GroupVar' is a vector or cell array defining the different groups.
%   If it is a continues variable median cut will be used as a default.
%
% OUTPUTS:
% * p       : log rank p-value
% * fh      : figure handle to KM-plot figure
% * stats   : Additional statistics from the log rank test
%
% stats =
%   struct with fields:
%
%               GroupNames: Cell with group names
%                     p_MC: log rank p-value (Mantel-Cox)
%                  Chi2_MC: Chi square (Mantel-Cox)
%               HR_logrank: Hazard Ratio (log rank)
%         HR_95_CI_logrank: 95% Confidence intervals [lower upper]
%           HR_logrank_Inv: Inverted Hazard Ratio (log rank)
%     HR_95_CI_logrank_Inv: Inverted 95% Confidence intervals [lower upper]
%                    HR_MH: Hazard Ratio (Mantel-Haenszel)
%              HR_95_CI_MH: 95% Confidence intervals [lower upper]
%                HR_MH_Inv: Inverted Hazard Ratio (Mantel-Haenszel)
%          HR_95_CI_MH_Inv: Inverted 95% Confidence intervals [lower upper]
%       MedianSurvivalTime: Median survival time for each group
%
%
% OTHER PARAMETERS (passed as parameter-value pairs)
% * 'NoPlot': A true/false value which, if true, no figure is created
%   (default: false)
% 
% * 'NoRiskTable': A true/false value which, if true, no risk table is
%   included in the KM-plot. (default: false)
%
% * 'CutPoint': Either a string or scalar/vector with cut points to be used
%   for defining groups based on a continuous 'GroupVar' input variable
%   Allowed names are: 'Median', 'Quartile' or 'Tertile'
%   If a scalar or vector is given the groups will be defined based on the
%   cut points. (default: 'Median')
%
% * 'GroupsToUse': Cell array defining what groups to use from the GroupVar
%   variable. Works only if GroupVar is a cell array. (default: all groups are used)
%
% * 'GroupOrder': A cell array defining the group order to be used in the
%   legend. (default: Groups are sorted alphabetically)
%
% * 'EventDefinition': Two element cell array where the first cell defines
%   the event and the second censored values. Example {'Dead','Alive'}
%
% * 'TimeMin': Scalar defining minimum valid time point. Subjects with time
%   values below this will be removed. (default: 0)
%
% * 'TimeMax': Scalar value defining righ censoring time. Subjects with
%   TimeVar > TimeMax will be set to TimeMax and considered as censored.
%   (default: [])
%
% * 'PairWiseP': A true/false for calculating pairwise log rank test
%   between group pairs, useful if there is more than two groups. (default: false)
%
% * 'Print': A true/false value which, if true, survival statistics are
%   printed in the command window(default: true)
%
% * 'NoWarnings': A true/false value which, if true, no warnings are printed
%   if subjects are removed. (default: false)
%
% * 'MedianLess': By default 'x < median' is used for median cut, but if false
%   'x > median' is used instead, only affect the results when there
%   is an odd number of samples (default: true)
%
%
% KM plot options
%
% * 'legend': Whether to show group legend. Default: true
%
% * 'LineColor': Either a matrix of size numLevels-by-3 representing the
%   colormap to be used or a string for a MATLAB colormap (lines, parula,
%   cool, prism) or 'JCO' 'nejm' 'Lancet' 'Science' 'Nature','lines' for a
%   set of Journal dependent palettes or my default 'aeb01' (default:'aeb01')
%
% * 'FlipGroupOrder': Flips the order of the groups in the legend.
%   (default: false)
%
% * 'FlipColorOrder': Flips the color order of the groups.
%   (default: false)
%
% * 'KM_position': Vector defining the KM axes for the KM plot
%   (default: [0.3 0.4 0.68 0.45])
%
% * 'RT_position': Vector defining the Risk Table axes for the KM plot
%   (default: [0.3 0.05 0.68 0.20])
%
% * 'TimeUnit': String defining time unit displayd on the x-axis.
%   (default: 'Months')
%
% * 'BaseFontSize': Base font size for all text in the plot
%   (default: 16)
%
% * 'DispP': A true/false value which, if true, log rank test p-value
%   is displayed on the KM-plot. (default: true)
%
% * 'DispHR': A true/false value which, if true, Hazard ration (HR)
%   is displayed on the KM-plot. (default: true)
%
% * 'Use_HR_MH': A true/false value which, if true, Mantel-Haenszel HR
%   is displayed instead of the logrank HR. (default: true)
%
% * 'InvHR': A true/false value which, if true, the inverted HR value
%   is displayed on the KM-plot. (default: false)
%
% * 'DrawMSL': A true/false value which, if true, a line for the median
%   survival time is drawn in the KM-plot. (default: false)
%
% * 'XLim': Vector defining the XLim. Do not affect the log rank test
%   (default: automatic)
%
% * 'LineWidth': Scalar defining the line width used in the KM-plot
%   (Default: 2)
%
% * 'LineStyle': Cell array defining the linestyle for the KM-plot.
%   If an array is given each group will have different linestyle, for example
%   'LineStyle',{'-','--',':','-.'}
%   (Default: {'-'})
%
% * 'CensorLineWidth': Scalar defining the linewith of the censored ticks
%   (default: 2)
%
% * 'CensorLineLength': Scalar defining the length of the censored ticks
%   (Default: 0.02)
%
% * 'CensorLineColor': Text string defining color of censor ticks. 'same'
%   gives the same colors as the lines while 'k' would make them all black
%   (Default: 'same')
%
% * 'Xstep': Scalar defining the X tick step length.
%   (defaut: automatic)
%
% * 'XTicks': Vector defining the position of the X-tick marks
%   (Default: automatic)
%
% * 'XMinorTick': Scalar defining the number of minor ticks between major X
%   ticks (Default: 1)
%
% * 'Xlabel': Text string for X-label (Default: 'Time (Months)' )
%
% * 'XlabelOptions': MATLAB Name-value pair arguments for xlabel (Default: '')
%
% * 'XLabelFontSize': Scalar describing Xlabel font size change compared
%   to base font size (Default: 0)
%
% * 'XTickFontSize': Scalar describing Xtick font size change compared
%   to base font size (Default: -2)
%
% * 'YTicks': Vector defining the position of the X-tick marks
%   (Default: [0:0.2:1])
%
% * 'YMinorTick': Scalar defining the number of minor ticks between major Y
%   ticks (Default: 1)
%
% * 'Ylabel': Text string for Y-label (Default: 'Survival Probability' )
%
% * 'YlabelOptions': MATLAB Name-value pair arguments for ylabel (Default: '')
%
% * 'YLabelFontSize': Scalar describing Ylabel font size change compared
%   to base font size (Default: 0)
%
% * 'YTickFontSize': Scalar describing Ytick font size change compared
%   to base font size (Default: -2)
%
% * 'Title': Text string for Title (Default: '' )
%
% * 'TitleOptions': MATLAB Name-value pair arguments for Title (Default: '')
%
% * 'LegendFontSize': Scalar describing Legend font size change compared
%   to base font size (Default: -2)
%
% * 'PvalFontSize': Scalar describing p-value font size change compared
%   to base font size (Default: 0)
%
% Risk table plot options
% * 'RT_FontSize': Scalar describing Risk Table font size change compared
%   to base font size (Default: 0)
%
% * 'RT_Color': Text string defining color of Risk table text. 'same'
%   gives the same colors as the groups in the KM plot while 'k' would make
%   them all black (Default: 'same')
%
% * 'RT_Title': Text string for Risk Table Title (Default: '' )
%
% * 'RT_TitleOptions': MATLAB Name-value pair arguments for Risk Table Titel (Default: '')
%
% * 'RT_YLabel': True/False for displaying the group names on the Risk table
%   Y-axis (Default: True )
%
% * 'CensorInRT': True/False for whether number censored should be listed
% in risk table (Default: False)
%
% * 'RTtitleAlignment': Where RT title should be algined (Default: middle)
%
%   EXAMPLES:
%   [p,fh,stats] = MatSurv([], [], [],'Xstep',4,'Title','MatSurv KM-Plot','FlipColor',1,'XMinorTick',3);
%
%
% MatSurv do NOT use any toolboxes
%
%   More examples can be found at: https://github.com/aebergl/MatSurv
%
% *** Anders Berglund ***
%
% Modification of Anders Berglund's excellent MatSurv function to accept
% more than two groups. Also implemented custom ordering of groups. Added
% option for legend to be off (default: on);
% -Patrick Leo, CCIPD


% Check TimeVar, EventVar, GroupVar variables
if nargin < 3
    error('MatSurv requires at least 3 input argument');
end

% Load test data
if isempty(TimeVar) && isempty(EventVar) && isempty(GroupVar)
    [TimeVar, EventVar, GroupVar] = MatSurvLoadTestData;
    varargin =[varargin,{'TimeUnit','Weeks'},{'Title',{'Test example taken from:';'Freireich, EJ et al. 1963, Blood, 21, 699-716)'}}];
end

% Check that they are all vectors
if min(size(TimeVar)) ~= 1 || min(size(EventVar)) ~= 1 || min(size(GroupVar)) ~= 1
    error('TimeVar, EventVar, GroupVar must all be vectors or cell arrays');
end

% Check that they all are equal length
if (length(TimeVar) ~= length(EventVar)) || (length(TimeVar) ~= length(GroupVar)) || (length(EventVar) ~= length(GroupVar))
    error('TimeVar, EventVar & GroupVar must all have equal length');
end

% Check for MATLAB version, currently MatSurv only work with 9.1 (2016b) and later
if verLessThan('matlab','9.1')
    error('MatSurv do not work with this version of MATLAB');
end

%Parse input and set default values
options = MatSurvParseInput(varargin{:});

% Check input and clean input data
[TimeVar, EventVar, GroupVar] = MatSurvCleanData(TimeVar, EventVar, GroupVar, options);

% Define events 1=event, 0=no event but it also checks for dead/alive etc use
% EventDefinition parameter for full control
[EventVarBin] = MatSurvDefineEventVar(EventVar, options);

% Censor data if TimeMax is given
if ~isempty(options.TimeMax)
    [TimeVar, EventVarBin] = MatSurvCensorTimeMax(TimeVar, EventVarBin, options);
end

% CreatGroups based on GroupVar and create DATA structure
[DATA,options] = MatSurvCreateGroups(TimeVar, EventVarBin, GroupVar, options);

% Flip Group Ordering
if options.FlipGroupOrder
    DATA.GROUPS = DATA.GROUPS(DATA.numGroups:-1:1);
end

if options.GroupOrder
    DATA.GROUPS = DATA.GROUPS(options.GroupOrder);
end


% Creat Survival table for plotting
[DATA] = MatSurvCreateTable(DATA);

% Do log rank test
[p,stats] = MatSurvLogRank(DATA);

if options.PairWiseP
    counter = 0;
    stats.ParwiseName = cell(DATA.numGroups * (DATA.numGroups - 1) / 2,1);
    for i = 1:DATA.numGroups - 1
        for j = i+1:DATA.numGroups
            counter  = counter + 1;
            DATA_tmp.numGroups = 2;
            DATA_tmp.GROUPS(1) = DATA.GROUPS(i);
            DATA_tmp.GROUPS(2) = DATA.GROUPS(j);
            [~,stats.ParwiseStats(counter)] = MatSurvLogRank(DATA_tmp);
            stats.ParwiseName{counter} = sprintf('%s vs. %s',DATA.GROUPS(i).GroupName{1},DATA.GROUPS(j).GroupName{1});
        end
    end
end

% Calculate median survival time if no plot is created
if options.NoPlot
    stats.MedianSurvivalTime=ones(DATA.numGroups,1) * NaN;
    for i=1:DATA.numGroups
        [xb,yb] = stairs(DATA.GROUPS(i).KM_ALL(:,1),DATA.GROUPS(i).KM_ALL(:,2));
        % Calculate Median Survival time:
        indx_MST = find((yb <= 0.5),1);
        if ~isempty(indx_MST)
            stats.MedianSurvivalTime(i) = xb(indx_MST);
%         else
%             % revert to other median
%             [~,indx_MST] = min(yb-median(yb));
%             stats.MedianSurvivalTime(i) = xb(indx_MST);
        end
    end
    fh = [];
    
else % Creat KM-Plot
    
    % Create Figure Window
    fh=figure('Name','MatSurv KM-Plot','Color','w','Tag','MatSurv KM-Plot figure');
    
    %Create Axes
    if options.NoRiskTable
        axh_KM = axes(fh,'NextPlot','add','tag','KM-Plot');
    else
        axh_KM = axes(fh,'Position',options.KM_position,'NextPlot','add','tag','KM-Plot');
        axh_RT = axes(fh,'Position',options.RT_position,'tag','Risk Table');
        % No axis for the Risk Table
        axh_RT.XAxis.Visible='off';
        axh_RT.YAxis.Visible='off';
    end
    
    % Adjust Colors for user input
    if ischar(options.LineColor)
        if any(strcmpi(options.LineColor,{'JCO','nejm','Lancet','Science','Nature','aeb01'}))
            cMAP = GetMatSurvColorPalette(options.LineColor);
        else
            cMAP = feval(options.LineColor, DATA.numGroups);
        end
    elseif ismatrix(options.LineColor)
        cMAP = options.LineColor;
        cMAP = cMAP(1:DATA.numGroups,:);
    else
        cMAP = GetMatSurvColorPalette;
    end
    
    if options.FlipColorOrder
        cMAP = flipud(cMAP);
    end
    
    % Adjust line style
    if ischar(options.LineStyle)
        LineStyles = cell(DATA.numGroups,1);
        LineStyles(:) = {options.LineStyle};
    elseif numel(options.LineStyle) == 1
        LineStyles = cell(DATA.numGroups,1);
        LineStyles(:) = options.LineStyle;
    elseif iscell(options.LineStyle)
        LineStyles = options.LineStyle;
    end
    
    % Adjust censoring markers
    if ischar(options.CensorLineColor) && strcmpi('same',options.CensorLineColor)
        cMAPCensor = cMAP;
    elseif ismatrix(options.CensorLineColor)
        cMAPCensor = options.CensorLineColor;
    end
    
    % Create stairs
    S=gobjects(DATA.numGroups,1);
    stats.MedianSurvivalTime=ones(DATA.numGroups,1) * NaN;
    for i=1:DATA.numGroups
        S(i)=stairs(axh_KM,DATA.GROUPS(i).KM_ALL(:,1),DATA.GROUPS(i).KM_ALL(:,2),'Color',cMAP(i,:),'Linewidth',options.LineWidth,'LineStyle',LineStyles{i});
        % Calculate Median Survival time:
        indx_MST = find((S(i).YData <= 0.5),1);
        if ~isempty(indx_MST)
            stats.MedianSurvivalTime(i) = S(i).XData(indx_MST);
            if options.DrawMSL
                line(axh_KM,[stats.MedianSurvivalTime(i) stats.MedianSurvivalTime(i)], [0.5 0],'LineStyle','--','Linewidth',1.5,'Color','k');
                line(axh_KM,[0 stats.MedianSurvivalTime(i)], [0.5 0.5],'LineStyle','--','Linewidth',1.5,'Color','k');
            end
        end
        if ~isempty(DATA.GROUPS(i).Censored_Points)
            % Draw marks for censored points
            line(axh_KM,[DATA.GROUPS(i).Censored_Points(:,1)'; DATA.GROUPS(i).Censored_Points(:,1)'],...
                [DATA.GROUPS(i).Censored_Points(:,2)'-options.CensorLineLength ; DATA.GROUPS(i).Censored_Points(:,2)'+options.CensorLineLength],...
                'Color',cMAPCensor(i,:),'Linewidth',options.CensorLineWidth);
        end
    end
    
    %Fix Y-Axis
    % Limit range from 0 to 1
    axh_KM.YLim = [0 1];
    axh_KM.YTick = options.YTick;
    YMinorStep =  (options.YTick(2) - options.YTick(1) ) / (1+options.YMinorTick);
    axh_KM.YAxis.MinorTickValues = YMinorStep:YMinorStep:1;
    axh_KM.YAxis.MinorTick = 'on';
    axh_KM.YAxis.TickDirection = 'out';
    
    % Y label
    axh_KM.YAxis.FontSize=options.BaseFontSize + options.YTickFontSize;
    ylabel(axh_KM,options.Ylabel,'FontSize',options.BaseFontSize + options.YLabelFontSize,options.YlabelOptions{:});
    
    % X label
    axh_KM.XAxis.FontSize=options.BaseFontSize + options.XTickFontSize;
    if isempty(options.Xlabel)
        xlabel_str = sprintf('Time (days)',options.TimeUnit);
    else
        xlabel_str = options.Xlabel;
    end
    xlabel(axh_KM,xlabel_str,'FontSize',options.BaseFontSize + options.XLabelFontSize,options.XlabelOptions{:});
    axh_KM.XAxis.TickDirection = 'out';
    
    % Title
    if ~isempty(options.Title)
        title(axh_KM,options.Title,'FontSize',18,options.TitleOptions{:});
    end
    
    % Set legend
    if options.legend
        h_LE=legend(S,[DATA.GROUPS(:).GroupName]);
        h_LE.Box='off';
        title(h_LE,DATA.GroupType);
        h_LE.FontSize=options.BaseFontSize + options.LegendFontSize;
    end
    
    % Get Xticks
    if ~isempty(options.XLim)
        axh_KM.XLim = [0 options.XLim];
    end
    
    max_X = axh_KM.XLim(2);
    Nudge_X = max_X / 50;
    
    if ~isempty(options.Xstep)
        axh_KM.XTick = 0:options.Xstep:max_X;
    end
    if ~isempty(options.XTicks)
        axh_KM.XTick = options.XTicks;
    end
    axh_KM.XAxis.MinorTick = 'on';
    XMinorStep =  (axh_KM.XTick(2) - axh_KM.XTick(1) ) / (1+options.XMinorTick);
    axh_KM.XAxis.MinorTickValues = XMinorStep:XMinorStep:axh_KM.XTick(end);
    axh_KM.LineWidth = 1.5;
    
    if options.DispP
        txt_str(1) = {sprintf('p = %.3g',p)};
        if options.DispHR && isfield(stats,'HR_logrank')
            if ~options.Use_HR_MH
                if options.InvHR
                    txt_str(2) = {sprintf('HR = %.3g (%.3g - %.3g)',stats.HR_logrank_Inv, stats.HR_95_CI_logrank_Inv(1), stats.HR_95_CI_logrank_Inv(2))};
                else
                    txt_str(2) = {sprintf('HR = %.3g (%.3g - %.3g)',stats.HR_logrank, stats.HR_95_CI_logrank(1), stats.HR_95_CI_logrank(2))};
                end
            else
                if options.InvHR
                    txt_str(2) = {sprintf('HR = %.3g (%.3g - %.3g)',stats.HR_MH_Inv, stats.HR_95_CI_MH_Inv(1), stats.HR_95_CI_MH_Inv(2))};
                else
                    txt_str(2) = {sprintf('HR = %.3g (%.3g - %.3g)',stats.HR_MH, stats.HR_95_CI_MH(1), stats.HR_95_CI_MH(2))};
                end
            end
        end
        text(axh_KM,Nudge_X,0.1,txt_str,'FontSize',options.BaseFontSize + options.PvalFontSize,'tag','p-value')
    end
   
    % And now to the Risk table
%     if ~options.NoRiskTable
%         axh_RT.XTick=axh_KM.XTick;
%         % Get number of samples for each time point
%         RT_X = zeros(length(axh_KM.XTick),DATA.numGroups);
%         RT_Xcensor = zeros(length(axh_KM.XTick),DATA.numGroups);
%         for i = 1:length(axh_KM.XTick)
%             for j = 1:DATA.numGroups
%                 %RT_X(i,j) = sum(DATA.GROUPS(j).TimeVar > axh_KM.XTick(i) & DATA.GROUPS(j).EventVar == 1) + sum(DATA.GROUPS(j).TimeVar >= axh_KM.XTick(i) & DATA.GROUPS(j).EventVar == 0);
%                 if(options.CensorInRT)
%                     RT_X(i,j) = sum(DATA.GROUPS(j).TimeVar >= axh_KM.XTick(i));
%                     RT_Xcensor(i,j) = sum(DATA.GROUPS(j).TimeVar < axh_KM.XTick(i) & DATA.GROUPS(j).EventVar == 0);
%                 else
%                     RT_X(i,j) = sum(DATA.GROUPS(j).TimeVar >= axh_KM.XTick(i));
%                 end
%             end
%             
%         end
% % %         axh_RT.YLim = [0.5 DATA.numGroups + 0.5];
% %         axh_RT.YLim = [0 DATA.numGroups + 0.5];
% %         axh_RT.YTick = 1:DATA.numGroups;
% %         linkaxes([axh_RT,axh_KM],'x')
% %         
% %         % Color OptionsFor Risk Table
% %         if ischar(options.RT_Color) && strcmpi('same',options.RT_Color)
% %             cMAP_RT = cMAP;
% %         elseif ismatrix(options.RT_Color)
% %             cMAP_RT = options.RT_Color;
% %             cMAP_RT = repmat(cMAP_RT,DATA.numGroups,1);
% %         end
% %         
% %         for i = 1:length(axh_KM.XTick)
% %             for j = 1:DATA.numGroups
% %                 %sprintf('%u',RT_X(i,j))
% %                 if(options.CensorInRT)
% %                     text(axh_RT,axh_RT.XTick(i),axh_RT.YTick(end-j+1),sprintf('%u (%u)',RT_X(i,j),RT_Xcensor(i,j)),...
% %                     'HorizontalAlignment',options.RTtitleAlignment,'VerticalAlignment','middle',...
% %                     'FontSize',options.BaseFontSize + options.RT_FontSize,'Color',cMAP_RT(j,:))
% %                 else
% %                 text(axh_RT,axh_RT.XTick(i),axh_RT.YTick(end-j+1),sprintf('%u',RT_X(i,j)),...
% %                     'HorizontalAlignment',options.RTtitleAlignment,'VerticalAlignment','middle',...
% %                     'FontSize',options.BaseFontSize + options.RT_FontSize,'Color',cMAP_RT(j,:))
% %                 end
% %             end
% %         end
% %         % Create Line
% %         %Get position for all text objects
% %         txt_pos = [axh_RT.Children(2:end).Extent];
% %         %get the second element for all text objects
% %         left_pos = min(txt_pos(1:4:end));
% %         nudge_x = abs(axh_RT.XLim(2) - axh_RT.XLim(1))/100;
% %         
% %         line(axh_RT,[left_pos-nudge_x left_pos-nudge_x],[axh_RT.YTick(1)-0.5 axh_RT.YTick(end)+0.5],'color','k','clipping','off','LineWidth',1.25)
% %          
% %         
% %         %Set Y label for risk table
% %         if options.RT_YLabel
% %             for j = 1:DATA.numGroups
% %                 text(axh_RT,left_pos-(nudge_x*2),axh_RT.YTick(end-j+1),DATA.GROUPS(j).GroupName,...
% %                     'HorizontalAlignment','right','VerticalAlignment','middle',...
% %                     'FontSize',options.BaseFontSize + options.RT_FontSize,'Color',cMAP_RT(j,:),'FontWeight','bold')
% %             end
% %         end
% %         % Title
% %         if ~isempty(options.RT_Title)
% %             ht = title(axh_RT,options.RT_Title,'FontSize',14,options.TitleOptions{:});
% %             ht.VerticalAlignment='middle';
% %         end
%     end
end

if options.Print 
    fprintf('\n')
    fprintf('p = %.3g\n',stats.p_MC)
    if options.CalcHR && isfield(stats,'HR_logrank')
        if options.InvHR
            fprintf('HR = %.3g (%.3g - %.3g)\n',stats.HR_logrank_Inv, stats.HR_95_CI_logrank_Inv(1), stats.HR_95_CI_logrank_Inv(2));
        else
            fprintf('HR = %.3g (%.3g - %.3g)\n',stats.HR_logrank, stats.HR_95_CI_logrank(1), stats.HR_95_CI_logrank(2));
        end
    end
    for i = 1: DATA.numGroups
        fprintf('Median Survival Time: (%s) = %g\n',stats.GroupNames{i},stats.MedianSurvivalTime(i))
    end
    fprintf('\n')
end

% Define output variables dependent of varargout
if nargout > 0
    varargout{1} = p;
end
if nargout > 1
    varargout{2} = fh;
end
if nargout > 2
    varargout{3} = stats;
end

end

function params = MatSurvParseInput(varargin)
%Parse input and set defualt values
p = inputParser;

p.addParameter('legend',true);

p.addParameter('NoPlot',false);
p.addParameter('NoRiskTable',false);
p.addParameter('CutPoint','Median');
p.addParameter('GroupOrder',[]);
p.addParameter('GroupsToUse',[]);
p.addParameter('EventDefinition',[]);
p.addParameter('TimeMin',0, @(x)isnumeric(x) && isscalar(x));
p.addParameter('TimeMax',[], @(x)isnumeric(x) && isscalar(x));
p.addParameter('FlipGroupOrder',0);
p.addParameter('FlipColorOrder',0);
p.addParameter('NoWarnings',false);
p.addParameter('TimeUnit','Months');
p.addParameter('PairWiseP',0);
p.addParameter('Print',1);
p.addParameter('MedianLess',1);

% Figure Options
p.addParameter('KM_position',[0.25 0.4 0.70 0.45]);
p.addParameter('RT_position',[0.25 0.05 0.70 0.20]);
p.addParameter('BaseFontSize',16);


% KM plot options
p.addParameter('DispP',1);
p.addParameter('DispHR',1);
p.addParameter('Use_HR_MH',0);
p.addParameter('DrawMSL',0);
p.addParameter('InvHR',0);
p.addParameter('Xstep',[], @(x)isnumeric(x) && isscalar(x));
p.addParameter('XTicks',[], @(x)isnumeric(x) && isvector(x));
p.addParameter('XMinorTick',1, @(x)isnumeric(x) && isscalar(x));

p.addParameter('XLim',[], @(x)isnumeric(x) && isscalar(x));
p.addParameter('LineColor','aeb01');
p.addParameter('LineWidth',2);
p.addParameter('LineStyle','-');
p.addParameter('CensorLineWidth',2);
p.addParameter('CensorLineLength',0.02);
p.addParameter('CensorLineColor','same');

p.addParameter('Xlabel',[]);
p.addParameter('XlabelOptions',cell(0,0));
p.addParameter('XLabelFontSize',0);
p.addParameter('XTickFontSize',-2);

p.addParameter('Ylabel','Survival Probability');
p.addParameter('YlabelOptions',cell(0,0));
p.addParameter('YLabelFontSize',0);
p.addParameter('YTickFontSize',-2);
p.addParameter('YTick',0:0.2:1);
p.addParameter('YMinorTick',1);

p.addParameter('Title',[]);
p.addParameter('TitleOptions',cell(0,0));
p.addParameter('LegendFontSize',-2);
p.addParameter('PvalFontSize',-2);

% Risk table plot options
p.addParameter('RT_FontSize',0);
p.addParameter('RT_Color','same');
p.addParameter('RT_YLabel',1);
p.addParameter('RT_Title',[]);
p.addParameter('RT_TitleOptions',cell(0,0));
p.addParameter('CensorInRT',false);
p.addParameter('RTtitleAlignment','center')

%Others
p.addParameter('CalcHR',1);


parse(p,varargin{:});
params = p.Results;

end

function [p,stats] = MatSurvLogRank(DATA)

% Merge tables from all groups
KM_ALL = vertcat(DATA.GROUPS.KM_Events);

% Get all time points with events
tf = KM_ALL(:,1);
tf = unique(tf);

% allocate matrices
n = length(tf);
mf = zeros(n,DATA.numGroups); % Observed failures
nf = zeros(n,DATA.numGroups); % Number at risk
ef = zeros(n,DATA.numGroups); % Expected number of failures

% Assign values
for i = 1:DATA.numGroups
    % Need to add censored time entries for group i
    tf_in = unique([tf;DATA.GROUPS(i).TimeVar]);
    [KM_Events, ~, ~] = MatSurvCalculateTables(tf_in,DATA.GROUPS(i).TimeVar,DATA.GROUPS(i).EventVar,tf);
    nf(:,i) = KM_Events(:,2);
    mf(:,i) = KM_Events(:,3);
end

% Calculate sums over all groups
nf_sum = sum(nf,2);
mf_sum = sum(mf,2);

% Calculated expected values
for i = 1:DATA.numGroups
    ef(:,i) = (nf(:,i)  ./ nf_sum) .* mf_sum;
end
%[tf mf nf ef]

d = sum(mf(:,1:end-1)-ef(:,1:end-1))';

%Calculate Variance
Var_OE=zeros(n,DATA.numGroups-1);
for i = 1:DATA.numGroups-1
    Var_OE(:,i) = (nf(:,i) .* (nf_sum - nf(:,i)) .* mf_sum .*(nf_sum - mf_sum)) ./ (nf_sum.^2 .* (nf_sum - 1));
end
Var_OE(isnan(Var_OE)) = 0;
Var_OE_sum = sum(Var_OE);

%Calculate covariance
Cov_OE = zeros(n,(DATA.numGroups-1)*(DATA.numGroups-2)/2);
if DATA.numGroups > 2 % If there are more than 2 groups
    counter = 0;
    for i = 1:DATA.numGroups-2
        for j = i+1:DATA.numGroups-1
            counter = counter + 1;
            Cov_OE(:,counter) = ( -nf(:,i) .* nf(:,j) .* mf_sum .* (nf_sum - mf_sum)) ./ (nf_sum.^2 .* (nf_sum -1));
        end
    end
    Cov_OE(isnan(Cov_OE)) = 0;
    Cov_OE_sum = sum(Cov_OE);
    V = zeros(DATA.numGroups-1);
    V(tril(true(DATA.numGroups-1),-1))=Cov_OE_sum;
    V(~tril(true(DATA.numGroups-1),0))=Cov_OE_sum;
    V(1:size(V,1)+1:end) = Var_OE_sum;
    
else % Special case for 2 groups
    V = Var_OE_sum;
end

% Mantel Cox
%Calculate Chi2
Chi2 = d'/V*d;

%p = 1 - gammainc(stats.Chi2/2,(DATA.numGroups-1)/2);
p = gammainc(Chi2/2,(DATA.numGroups-1)/2,'upper');

% Create stats output
stats.GroupNames = [DATA.GROUPS.GroupName]';
stats.p_MC = p;
stats.Chi2_MC = Chi2';

% Caclulate Hazard Ratio

if DATA.numGroups == 2
    stats.HR_logrank = (sum(mf(:,1)) / sum(ef(:,1))) / (sum(mf(:,2)) / sum(ef(:,2)));
    stats.HR_95_CI_logrank = [exp((log(stats.HR_logrank) - 1.96 * sqrt(1/sum(ef(:,1)) + 1/sum(ef(:,2))))), exp((log(stats.HR_logrank) + 1.96 * sqrt(1/sum(ef(:,1)) + 1/sum(ef(:,2)))))];
    stats.HR_logrank_Inv = 1/stats.HR_logrank;
    stats.HR_95_CI_logrank_Inv = flip(1 ./ stats.HR_95_CI_logrank);
    L = (sum(mf(:,1)) - sum(ef(:,1))) / Var_OE_sum;
    stats.HR_MH  = exp(L);
    stats.HR_95_CI_MH = [exp(L - 1.96/sqrt(Var_OE_sum)), exp(L + 1.96/sqrt(Var_OE_sum))];
    stats.HR_MH_Inv  = 1 / stats.HR_MH;
    stats.HR_95_CI_MH_Inv = flip(1 ./ stats.HR_95_CI_MH);
end


end

function [DATA] = MatSurvCreateTable(DATA)

for i=1:DATA.numGroups
    % Get unique time points including censored and add a leading 0
    tf = [0; unique(DATA.GROUPS(i).TimeVar)];
    
    [KM_Events, KM_ALL, Censored_Points] = MatSurvCalculateTables(tf,DATA.GROUPS(i).TimeVar,DATA.GROUPS(i).EventVar,[]);
    
    DATA.GROUPS(i).KM_Events=KM_Events;
    DATA.GROUPS(i).KM_ALL=KM_ALL;
    DATA.GROUPS(i).Censored_Points=Censored_Points;
end

end

function [KM_Events, KM_ALL, Censored_Points] = MatSurvCalculateTables(tf,TimeVar,EventVar,tf_out)

% Calculate number  of samples for each time point including censored
% Thanks to ashrafinia for identifying and fixing bug if there is only one group member
mf = sum(repmat(TimeVar,1,length(tf)) == repmat(tf',length(TimeVar),1),1)';

%Calculate number of samples left at each time point
mf_cumsum = cumsum(mf);
nf = ones(length(tf),1) * length(TimeVar);
nf(2:end) = nf(2:end) - mf_cumsum(1:end-1);

% Find censored points
indx_censor = (EventVar == 0);
tfq = unique(TimeVar(indx_censor));
mfq = sum(repmat(TimeVar(indx_censor),1,length(tfq)) == repmat(tfq',length(TimeVar(indx_censor)),1),1)';

% Find time points where there are censored data
[~,tf_indx,~]=intersect(tf,tfq,'stable');

%Adjust counts for censored data, a 0 will mean that only censored data
%was observed at that time
mf_true = mf;
mf_true(tf_indx) = mf((tf_indx)) - mfq;

% Calculate fraction alive
S = cumprod(1-(mf_true./nf));

% get index of censored samples for plotting
indx_censored = ((mf_true - mf) < 0);

%
if isempty(tf_out) % function called for plotting
    indx_observed = (mf_true ~= 0);
else % fynction called for log rank test
    [~,~,indx_observed] = intersect(tf_out,tf);
end

KM_Events = [tf(indx_observed) nf(indx_observed) mf_true(indx_observed)];
KM_ALL = [tf S nf];
Censored_Points=[tf(indx_censored) S(indx_censored)];

end

function [DATA,options] = MatSurvCreateGroups(TimeVar, EventVarBin, GroupVar, options)
% Create Group structure
DATA.numGroups = 0;
DATA.GROUPS = struct('GroupName',{},'TimeVar',[],'EventVar',[]);

% Define set of Groups to use
if ~isempty(options.GroupsToUse) % User defined Groups to use
    DATA.numGroups = numel(options.GroupsToUse);
    DATA.GroupType = 'Groups';
    for i = 1:DATA.numGroups
        if iscell(GroupVar)
            indx_group = strcmp(options.GroupsToUse(i),GroupVar);
            DATA.GROUPS(i).GroupName = options.GroupsToUse(i);
        elseif isnumeric(GroupVar)
            indx_group = (options.GroupsToUse(i) == GroupVar);
            DATA.GROUPS(i).GroupName = {num2str(options.GroupsToUse(i))};
        end
        DATA.GROUPS(i).TimeVar = TimeVar(indx_group);
        DATA.GROUPS(i).EventVar = EventVarBin(indx_group);
    end
    
    % If the Groupvariable is a cell vector
elseif iscell(GroupVar)
    Unique_Groups = unique(GroupVar);
    DATA.numGroups = length(Unique_Groups);
    DATA.GroupType = 'Groups';
    for i = 1:DATA.numGroups
        indx_group = strcmp(Unique_Groups(i),GroupVar);
        DATA.GROUPS(i).GroupName = Unique_Groups(i);
        DATA.GROUPS(i).TimeVar = TimeVar(indx_group);
        DATA.GROUPS(i).EventVar = EventVarBin(indx_group);
    end
    
    % If the Groupvariable is a numerical vector
elseif (strcmpi('Median',options.CutPoint) || isscalar(options.CutPoint)) && isnumeric(GroupVar)
    if strcmpi('Median',options.CutPoint)
        Cut_Val = median(GroupVar);
        DATA.GroupType = 'Median';
    elseif isscalar(options.CutPoint)
        Cut_Val = options.CutPoint;
        DATA.GroupType = 'Fixed value';
    end
    DATA.numGroups = 2;
    
    if options.MedianLess
        indx_Below  = (GroupVar < Cut_Val);
        indx_Above = ~indx_Below;
        DATA.GROUPS(1).GroupName = {sprintf('x >= %g',Cut_Val)};
        DATA.GROUPS(2).GroupName = {sprintf('x < %g',Cut_Val)};
    else
        indx_Above  = (GroupVar > Cut_Val);
        indx_Below  = ~indx_Above;
        DATA.GROUPS(1).GroupName = {sprintf('x > %g',Cut_Val)};
        DATA.GROUPS(2).GroupName = {sprintf('x <= %g',Cut_Val)};
    end
    
    DATA.GROUPS(1).TimeVar = TimeVar(indx_Above);
    DATA.GROUPS(1).EventVar = EventVarBin(indx_Above);    
    DATA.GROUPS(2).TimeVar = TimeVar(indx_Below);
    DATA.GROUPS(2).EventVar = EventVarBin(indx_Below);
    
elseif strcmpi('Quartile',options.CutPoint)  && isnumeric(GroupVar)
    Cut_Val = prctile(GroupVar,[25 75]);
    DATA.GroupType = 'Quartile';
    indx_Below  = (GroupVar < Cut_Val(1));
    indx_Above = (GroupVar > Cut_Val(2));
    DATA.numGroups = 2;
    DATA.GROUPS(1).GroupName = {sprintf('x > %g',Cut_Val(2))};
    DATA.GROUPS(1).TimeVar = TimeVar(indx_Above);
    DATA.GROUPS(1).EventVar = EventVarBin(indx_Above);
    DATA.GROUPS(2).GroupName = {sprintf('x < %g',Cut_Val(1))};
    DATA.GROUPS(2).TimeVar = TimeVar(indx_Below);
    DATA.GROUPS(2).EventVar = EventVarBin(indx_Below);
    
elseif strcmpi('Tertile',options.CutPoint)  && isnumeric(GroupVar)
    Cut_Val = prctile(GroupVar,[100/3 100/1.5]);
    DATA.GroupType = 'Tertile';
    indx_Below  = (GroupVar < Cut_Val(1));
    indx_Above = (GroupVar > Cut_Val(2));
    indx_Between = ~(indx_Below | indx_Above);
    DATA.numGroups = 3;
    %High
    DATA.GROUPS(1).GroupName = {sprintf('x > %g',Cut_Val(2))};
    DATA.GROUPS(1).TimeVar = TimeVar(indx_Above);
    DATA.GROUPS(1).EventVar = EventVarBin(indx_Above);
    %Medium
    DATA.GROUPS(2).GroupName = {sprintf('%g < x < %g',Cut_Val(1),Cut_Val(2))};
    DATA.GROUPS(2).TimeVar = TimeVar(indx_Between);
    DATA.GROUPS(2).EventVar = EventVarBin(indx_Between);
    %Low
    DATA.GROUPS(3).GroupName = {sprintf('x < %g',Cut_Val(1))};
    DATA.GROUPS(3).TimeVar = TimeVar(indx_Below);
    DATA.GROUPS(3).EventVar = EventVarBin(indx_Below);
    
    % Vector with several cut pints
elseif (isvector(options.CutPoint)) && isnumeric(GroupVar)
    CutPointSorted = sort(options.CutPoint,'descend');
    DATA.numGroups = length(CutPointSorted) + 1;
    DATA.GroupType = 'Cut Points';
    % For samples above
    indx_Above  = (GroupVar > CutPointSorted(1));
    DATA.GROUPS(1).GroupName = {sprintf('x > %g',CutPointSorted(1))};
    DATA.GROUPS(1).TimeVar = TimeVar(indx_Above);
    DATA.GROUPS(1).EventVar = EventVarBin(indx_Above);
    % For samples inbetween cut points
    for i = 1:length(CutPointSorted) - 1
        indx  = (GroupVar > CutPointSorted(i+1) & GroupVar <= CutPointSorted(i));
        DATA.GROUPS(i+1).GroupName = {sprintf('%g < x <= %g',CutPointSorted(i+1),CutPointSorted(i))};
        DATA.GROUPS(i+1).TimeVar = TimeVar(indx);
        DATA.GROUPS(i+1).EventVar = EventVarBin(indx);
    end
    %For samples below
    i = i + 1;
    indx  =  (GroupVar <= CutPointSorted(i));
    DATA.GROUPS(i+1).GroupName = {sprintf('x <= %g',CutPointSorted(i))};
    DATA.GROUPS(i+1).TimeVar = TimeVar(indx);
    DATA.GROUPS(i+1).EventVar = EventVarBin(indx);
end

% %Hazard ration can only be calculated if there is two groups
% if DATA.numGroups ~= 2
%     options.DispHR = 0;
%     options.CalcHR = 0;
% end

end

function [TimeVar, EventVarBin] = MatSurvCensorTimeMax(TimeVar, EventVarBin, options)
indx_TimeMax = (TimeVar > options.TimeMax);
TimeVar(indx_TimeMax) = options.TimeMax;
EventVarBin(indx_TimeMax) = 0;
end

function [EventVarBin] = MatSurvDefineEventVar(EventVar, options)

% Set all entries to zeros
EventVarBin = zeros(size(EventVar));

% Check if its a string array and convert to cell array
if isstring(EventVar)
    EventVar = cellstr(EventVar);    
end

if islogical(EventVar) % Set TRUE to 1
    EventVarBin(EventVar) = 1;
elseif isnumeric(EventVar) % set ones to 1
    EventVarBin(EventVar == 1) = 1;
elseif iscell(EventVar)
    if ~isempty(options.EventDefinition) % Set values based on user input
        indx_Event = strcmp(options.EventDefinition{1},EventVar);
        indx_NoEvent = strcmp(options.EventDefinition{2},EventVar);
        if sum(indx_Event) + sum(indx_NoEvent) == length(EventVar)
            EventVarBin(indx_Event) = 1;
            EventVarBin(indx_NoEvent) = 0;
        else
            error('Event variable do not match event type defined in options.EventDefinition')
        end
    else % Set values based on common event types such as dead/alive
        indx_Event = strcmpi('Dead',EventVar) | strcmpi('Deceased',EventVar) | strcmpi('Relapsed',EventVar)...
            |  strcmpi('Yes',EventVar) | strcmpi('Event',EventVar) | strcmpi('Progression',EventVar)...
            | strcmpi('Progressed',EventVar);
        
        indx_NoEvent = strcmpi('Alive',EventVar) | strcmpi('Living',EventVar) | strcmpi('NotRelapsed',EventVar)...
            | strcmpi('DiseaseFree',EventVar) | strcmpi('No',EventVar) | strcmpi('Censored',EventVar)...
            | strcmpi('NoProgression',EventVar) | strcmpi('NoEvent',EventVar);
        
        if sum(indx_Event) + sum(indx_NoEvent) == length(EventVar)
            EventVarBin(indx_Event) = 1;
            EventVarBin(indx_NoEvent) = 0;
        else
            error('Event variable has non recognazed type. Please check EventVar')
        end
    end
else
    error('Non supported Event variable input')
end

end

function [TimeVar, EventVar, GroupVar] = MatSurvCleanData(TimeVar, EventVar, GroupVar, options)
% Functions to check and clean input data

% Make sure that TimeVar, EventVar, GroupVar are all column vectors
% and not row vectors
if size(TimeVar,1) == 1
    TimeVar = TimeVar';
end
if size(EventVar,1) == 1
    EventVar = EventVar';
end
if size(GroupVar,1) == 1
    GroupVar = GroupVar';
end

% Check time variable for missing data and timepoints < TimeMin
rem_indx_time = ( isnan(TimeVar) | (TimeVar < options.TimeMin) );

% Check Event variable for missing data and/or empty cells and cells with NA
if isnumeric(EventVar) || islogical(EventVar)
    rem_indx_event = isnan(EventVar);
elseif iscell(EventVar)
    rem_indx_event = (cellfun('isempty',EventVar) | strcmpi('[Not Available]',EventVar) | strcmpi('NA',EventVar));
end
% Check Group variable for missing data and/or empty cells and cells with NA
if isnumeric(GroupVar) || islogical(GroupVar)
    rem_indx_group = isnan(GroupVar);
elseif iscell(GroupVar)
    rem_indx_group = (cellfun('isempty',GroupVar) | strcmpi('NA',GroupVar));
end

% Merge all indexes and remove entries
rem_indx = (rem_indx_time | rem_indx_event | rem_indx_group);
if sum(rem_indx) > 0
    TimeVar(rem_indx) = [];
    EventVar(rem_indx) = [];
    GroupVar(rem_indx) = [];
    if ~options.NoWarnings
        fprintf('*********************************\n');
        fprintf('\n');
        fprintf('%u sample have been removed\n',sum(rem_indx));
        fprintf('removed samples had missing data or time < %g\n',options.TimeMin)
        fprintf('\n');
        fprintf('*********************************\n');
    end
    
end

numEvenTypes = length(unique(EventVar));
if numEvenTypes > 2
    error('More then 2 event types in the Event variable');
end

end

function palette = GetMatSurvColorPalette(Id)
% Get different custom color palettes
% Valid names are {'JCO','nejm','Lancet','Science','Nature','aeb01'};

if strcmpi(Id,'JCO')
    palette = [
        0                   0.450980392156863   0.760784313725490
        0.937254901960784   0.752941176470588                   0
        0.525490196078431   0.525490196078431   0.525490196078431
        0.803921568627451   0.325490196078431   0.298039215686275
        0.478431372549020   0.650980392156863   0.862745098039216
        0                   0.235294117647059   0.403921568627451
        0.560784313725490   0.466666666666667                   0
        0.231372549019608   0.231372549019608   0.231372549019608
        0.654901960784314   0.188235294117647   0.188235294117647
        0.290196078431373   0.411764705882353   0.564705882352941];
    
elseif strcmpi(Id,'nejm')
    palette = [
        0.737254901960784   0.235294117647059   0.160784313725490
        0                   0.447058823529412   0.709803921568627
        0.882352941176471   0.529411764705882   0.152941176470588
        0.125490196078431   0.521568627450980   0.305882352941176
        0.470588235294118   0.462745098039216   0.694117647058824
        0.435294117647059   0.600000000000000   0.678431372549020
        1.000000000000000   0.862745098039216   0.568627450980392
        0.933333333333333   0.298039215686275   0.592156862745098];
    
elseif strcmpi(Id,'Lancet')
    palette = [
        0   0.274509803921569   0.545098039215686
        0.929411764705882                   0                   0
        0.258823529411765   0.709803921568627   0.250980392156863
        0                   0.600000000000000   0.705882352941177
        0.572549019607843   0.368627450980392   0.623529411764706
        0.992156862745098   0.686274509803922   0.568627450980392
        0.678431372549020                   0   0.164705882352941
        0.678431372549020   0.713725490196078   0.713725490196078
        0.105882352941176   0.098039215686275   0.098039215686275
        ];
elseif strcmpi(Id,'Nature')
    palette = [
        0.901960784313726   0.294117647058824   0.207843137254902
        0.301960784313725   0.733333333333333   0.835294117647059
        0                   0.627450980392157   0.529411764705882
        0.235294117647059   0.329411764705882   0.533333333333333
        0.952941176470588   0.607843137254902   0.498039215686275
        0.517647058823529   0.568627450980392   0.705882352941177
        0.568627450980392   0.819607843137255   0.760784313725490
        0.862745098039216                   0                   0
        0.494117647058824   0.380392156862745   0.282352941176471
        0.690196078431373   0.611764705882353   0.521568627450980
        ];
elseif strcmpi(Id,'Science')
    palette = [
        0.231372549019608   0.286274509803922   0.572549019607843
        0.933333333333333                   0                   0
        0                   0.545098039215686   0.270588235294118
        0.388235294117647   0.094117647058824   0.474509803921569
        0.372549019607843   0.333333333333333   0.607843137254902
        0.733333333333333                   0   0.129411764705882
        0                   0.509803921568627   0.501960784313725
        0.635294117647059                   0   0.337254901960784
        0.501960784313725   0.505882352941176   0.501960784313725
        0.105882352941176   0.098039215686275   0.098039215686275
        ];
elseif strcmpi(Id,'aeb01') % My own color palette
    palette = [
        0.737       0.235       0.161
        0           0.451       0.761
        0.937       0.753       0
        0.525       0.525       0.525
        0.125       0.522       0.306
        0.4940      0.1840      0.5560
        0.561       0.4667      0
        0.882       0.529       0.153
        0.435       0.600       0.678
        1.000       0.863       0.569
        0.933       0.298       0.592
        0.3010      0.7450      0.9330
        0.231       0.231       0.231
        ];
end

end

function [TimeVar, EventVar, GroupVar] = MatSurvLoadTestData
% Test example taken from "Freireich, EJ et al. 1963, Blood, 21, 699-716)"

t1=[6 6 6 7 10 13 16 22 23 6 9 10 11 17 19 20 25 32 32 34 35]';
t2=[1 1 2 2 3 4 4 5 5 8 8 8 8 11 11 12 12 15 17 22 23]';
TimeVar=[t1;t2];

e1=[1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]';
e2=ones(21,1);
EventVar = [e1;e2];

g1=cell(size(t1));
g1(:) = {'Group 1'};

g2=cell(size(t2));
g2(:) = {'Group 2'};

GroupVar = [g1;g2];

end
