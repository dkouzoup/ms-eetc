%-------------------------------------------------------------------------%
%                             Extract Solution                            %
%-------------------------------------------------------------------------%
solution = output.result.solution;

time = [];
state = [];
control = [];

numPhases = length(solution.phase);

for i = 1:numPhases

    time = vertcat(time, solution.phase(i).time);
    state = vertcat(state, solution.phase(i).state);
    control = vertcat(control, solution.phase(i).control);

end

% flip sign for braking

for i = 2:size(control,2)

    control(:,i) = - control(:,i);

end

%-------------------------------------------------------------------------%
%                              Plot Solution                              %
%-------------------------------------------------------------------------%
figure(1);
pp = plot(time,state,'-o');
xl = xlabel('$t$','Interpreter','LaTeX');
yl = ylabel('$x(t)$','Interpreter','LaTeX');
set(xl,'Fontsize',18);
set(yl,'Fontsize',18);
set(gca,'Fontsize',16,'FontName','Times');
set(pp,'LineWidth',1.25);
grid on
print -dpng trainState.png

figure(2);
pp = plot(time,control,'-o');
hold on
plot(points, zeros(size(points)), 'kx');
xl = xlabel('$t$','Interpreter','LaTeX');
yl = ylabel('$u(t)$','Interpreter','LaTeX');
set(xl,'Fontsize',18);
set(yl,'Fontsize',18);
set(gca,'Fontsize',16,'FontName','Times');
set(pp,'LineWidth',1.25);
grid on
print -dpng trainControl.png
