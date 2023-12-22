
clear all; clc

pseudospectral = false;  % true = GPOPS-I + IPOPT

%-------------------------------------------------------------------------%
%----------------------------- Train specs  ------------------------------%
%-------------------------------------------------------------------------%

g = 9.81;

minVelocity = 1;  % minimum velocity to avoid numerical issues [m/s]
maxVelocity = 160/3.6;  % maximum velocity [m/s]
mass = 391000;
rho = 1.06;

maxForce = 213900;
maxPower = 3129277;

minForcePn = 0*273500;
minForceRg = 213900;
minPowerRg = 3129277;

minAcc = -100;  % maximum deceleration [m/s^2] - basically inf
maxAcc = 100;  % maximum acceleration [m/s^2] - basically inf

r0 = 5854.0;
r1 = 74.16;
r2 = 12.96;

maxEnergy = 800;  % unrealistic consumption as upper bound [kWh]

etaTraction = 0.73;
etaRgBrake = 0.73;

% process specs

maxObjective = maxEnergy*(3.6/1e-6);

totalMass = mass*rho;
withRgBrake = minForceRg ~= 0;
withPnBrake = minForcePn ~= 0;

%-------------------------------------------------------------------------%
%----------------------------- Track specs  ------------------------------%
%-------------------------------------------------------------------------%

trackID = '00_var_speed_limit_100';

track.length = 48531.0;
track.time = 1541;

switch trackID

    case '00_reference'

        track.phases = 0;
        track.gradients = 0;
        track.speedLimits = 140/3.6;

    case '00_var_speed_limit_120'

        track.phases = [0, 25000, 35000];
        track.gradients = [0, 0, 0];
        track.speedLimits = [140, 120, 140]./3.6;

    case '00_var_speed_limit_110'

        track.phases = [0, 25000, 35000];
        track.gradients = [0, 0, 0];
        track.speedLimits = [140, 110, 140]./3.6;

    case '00_var_speed_limit_100'

        track.phases = [0, 25000, 35000];
        track.gradients = [0, 0, 0];
        track.speedLimits = [140, 100, 140]./3.6;

    case '00_var_gradient_minus_10'

        track.phases = [0, 25000, 35000];
        track.gradients = [0, -10, 0];
        track.speedLimits = [140, 140, 140]./3.6;

   case '00_var_gradient_minus_5'

        track.phases = [0, 25000, 35000];
        track.gradients = [0, -5, 0];
        track.speedLimits = [140, 140, 140]./3.6;

    case '00_var_gradient_plus_5'

        track.phases = [0, 25000, 35000];
        track.gradients = [0, 5, 0];
        track.speedLimits = [140, 140, 140]./3.6;

    case '00_var_gradient_plus_10'

        track.phases = [0, 25000, 35000];
        track.gradients = [0, 10, 0];
        track.speedLimits = [140, 140, 140]./3.6;

    case 'CH_StGallen_Wil'

        track.length = 29556.1;
        track.time = 1242;

        data = csvread('CH_StGallen_Wil.csv', 1, 0);
        track.phases = data(:,1)';
        track.gradients = data(:,2)';
        track.speedLimits = data(:,3)';

end

track.title = trackID;

% process specs

numPhases = length(track.phases);
track.phases = horzcat(track.phases, track.length);

if length(track.gradients) ~= numPhases || length(track.speedLimits) ~= numPhases

    error('Inconsistent dimensions in Track data!')

end

%-------------------------------------------------------------------------%
%----------------- Provide All Bounds for Problem ------------------------%
%-------------------------------------------------------------------------%

t0 = 0;
b0 = minVelocity^2;
bf = minVelocity^2;

uMin = 0;
uMax = maxForce/totalMass;

if withRgBrake

    uMin = horzcat(uMin, 0);
    uMax = horzcat(uMax, minForceRg/totalMass);

end

if withPnBrake

    uMin = horzcat(uMin, 0);
    uMax = horzcat(uMax, minForcePn/totalMass);

end

hMin = [minAcc, 0];
hMax = [maxAcc, maxPower/totalMass];

if withRgBrake

    hMin = horzcat(hMin, 0);
    hMax = horzcat(hMax, minPowerRg/totalMass);

end

%-------------------------------------------------------------------------%
%----------------------- Setup for Problem Bounds ------------------------%
%-------------------------------------------------------------------------%

for i = 1:numPhases

    bounds.phase(i).initialtime.lower = track.phases(i);
    bounds.phase(i).initialtime.upper = track.phases(i);

    bounds.phase(i).finaltime.lower = track.phases(i+1);
    bounds.phase(i).finaltime.upper = track.phases(i+1);

    if i == 1

        bounds.phase(i).initialstate.lower = [t0, b0];
        bounds.phase(i).initialstate.upper = [t0, b0];

        bounds.phase(i).finalstate.lower = [t0, minVelocity^2];
        bounds.phase(i).finalstate.upper = [track.time, min(maxVelocity^2, track.speedLimits(i)^2)];

    elseif i == numPhases

        bounds.phase(i).initialstate.lower = [t0, minVelocity^2];
        bounds.phase(i).initialstate.upper = [track.time, min(maxVelocity^2, track.speedLimits(i)^2)];

        bounds.phase(i).finalstate.lower = [track.time, bf];
        bounds.phase(i).finalstate.upper = [track.time, bf];

    else

        bounds.phase(i).initialstate.lower = [t0, minVelocity^2];
        bounds.phase(i).initialstate.upper = [track.time, min(maxVelocity^2, track.speedLimits(i)^2)];

        bounds.phase(i).finalstate.lower = [t0, minVelocity^2];
        bounds.phase(i).finalstate.upper = [track.time, min(maxVelocity^2, track.speedLimits(i)^2)];

    end

    if i < numPhases

        bounds.eventgroup(i).lower = zeros(1, 2);
        bounds.eventgroup(i).upper = zeros(1, 2);

    end

    bounds.phase(i).state.lower = [t0, minVelocity^2];
    bounds.phase(i).state.upper = [track.time, min(maxVelocity^2, track.speedLimits(i)^2)];

    bounds.phase(i).control.lower = uMin;
    bounds.phase(i).control.upper = uMax;
    bounds.phase(i).path.lower = hMin;
    bounds.phase(i).path.upper = hMax;
    bounds.phase(i).integral.lower = 0;
    bounds.phase(i).integral.upper = maxObjective/totalMass;

end

%-------------------------------------------------------------------------%
%---------------------- Provide Guess of Solution ------------------------%
%-------------------------------------------------------------------------%

u0 = zeros(1, 1 + withRgBrake + withPnBrake);
uf = u0;

for i = 1:numPhases

    x0 = [t0, bounds.phase(i).initialstate.upper(2)];
    xf = [track.time, bounds.phase(i).finalstate.upper(2)];

    guess.phase(i).time     = [bounds.phase(i).initialtime.lower; bounds.phase(i).finaltime.lower];
    guess.phase(i).state    = [x0; xf];
    guess.phase(i).control  = [u0; uf];
    guess.phase(i).integral = 0;

end

%-------------------------------------------------------------------------%
%----------Provide Mesh Refinement Method and Initial Mesh ---------------%
%-------------------------------------------------------------------------%

% available mesh methods:
% 'hp-PattersonRao' (default), 'hp-DarbyRao', 'hp-LiuRao', 'hp-LiuRao-Legendre'
% mesh.method = 'hp-LiuRao';

mesh.tolerance = 1e-6;

% mesh.maxiterations = 5;
% mesh.colpointsmin = 3;
% mesh.colpointsmax = 10;

if pseudospectral

    collPoints = 200;  % collocation points per phase + end point
    mesh.maxiterations = 0;

    for i = 1: numPhases

        mesh.phase(i).colpoints = collPoints;
        mesh.phase(i).fraction  = 1;

    end

end

%-------------------------------------------------------------------------%
%------------- Assemble Information into Problem Structure ---------------%
%-------------------------------------------------------------------------%

setup.name                           = 'Train-Problem';
setup.functions.continuous           = @trainContinuous;
setup.functions.endpoint             = @trainEndpoint;
setup.displaylevel                   = 2;
setup.bounds                         = bounds;
setup.guess                          = guess;
setup.mesh                           = mesh;
setup.nlp.solver                     = 'ipopt';
setup.nlp.ipoptoptions.maxiterations = 10000;
setup.nlp.ipoptoptions.linear_solver = 'ma57';
setup.nlp.ipoptoptions.tolerance     = 1e-8;
setup.derivatives.supplier           = 'sparseCD';
setup.derivatives.derivativelevel    = 'second';
setup.method                         = 'RPM-Differentiation';

setup.auxdata.g                      = g;
setup.auxdata.mass                   = mass;
setup.auxdata.rho                    = rho;
setup.auxdata.r0                     = r0;
setup.auxdata.r1                     = r1;
setup.auxdata.r2                     = r2;
setup.auxdata.gradients              = track.gradients/1000;
setup.auxdata.etaTraction            = etaTraction;
setup.auxdata.etaBrake               = etaRgBrake;
setup.auxdata.withRgBrake            = withRgBrake;
setup.auxdata.withPnBrake            = withPnBrake;


%-------------------------------------------------------------------------%
%---------------------- Solve Problem Using GPOPS2 -----------------------%
%-------------------------------------------------------------------------%

output = gpops2(setup);

energy = output.result.objective*(1e-6/3.6)*mass*rho;

sprintf("Consumed energy:\t %6.3f kWh", energy)
sprintf("Solved in :     \t %6.3f s", output.totaltime)

%-------------------------------------------------------------------------%
%------------------------- Save results to CSV ---------------------------%
%-------------------------------------------------------------------------%

sOpt  = vertcat(output.result.solution.phase.time);
xOpt  = vertcat(output.result.solution.phase.state);
uOpt  = vertcat(output.result.solution.phase.control)*totalMass;

tOpt  = xOpt(:,1);
vOpt  = sqrt(xOpt(:,2));
fTOpt = uOpt(:,1);

if withRgBrake

    fRbOpt = -uOpt(:,2);

else

    fRbOpt = 0*fTOpt;

end

if withPnBrake

    fPbOpt = -uOpt(:,2+withRgBrake);

else

    fPbOpt = 0*fTOpt;

end

for i=numPhases:-1:1

    meshOut(i).points = track.phases(i) + (track.phases(i+1)-track.phases(i))*cumsum(horzcat(0, output.meshhistory(end).result.setup.mesh.phase(i).fraction));
    meshOut(i).collDegrees = output.meshhistory(end).result.setup.mesh.phase(i).colpoints;
    meshOut(i).numPoints = length(meshOut(i).points) + sum(meshOut(i).collDegrees) - length(meshOut(i).collDegrees);

end

points = horzcat(meshOut.points);

titles = {"Time [s]", "Position [m]",  'Velocity [m/s]', 'Force (acc) [N]', 'Force (rgb) [N]', 'Force (pnb) [N]', 'Energy [kWh]', 'CPU Time [s]'};
data = horzcat(tOpt, sOpt, vOpt, fTOpt, fRbOpt, fPbOpt, energy*ones(size(fPbOpt)), output.totaltime*ones(size(fPbOpt)));

C = [titles; num2cell(data)];

if pseudospectral

    gpopsstr = 'GPOPSI';

else

    gpopsstr = 'GPOPSII';

end

writecell(C, strcat(track.title, gpopsstr, '.csv'))
