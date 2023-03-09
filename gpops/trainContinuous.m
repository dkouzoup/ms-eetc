
function phaseout = trainContinuous(input)

% parameters

p = input.auxdata;

totalMass = p.mass*p.rho;
sr0 = p.r0/totalMass;
sr1 = p.r1/totalMass;
sr2 = p.r2/totalMass;

numPhases = length(p.gradients);

for i= numPhases:-1:1
    
    % states
    
    x = input.phase(i).state;
    b = x(:,2);
    
    % controls
    
    u = input.phase(i).control;
    
    ftr = u(:,1);
    
    if p.withRgBrake
        
        frb = u(:,2);
    
    else
    
        frb = 0;
    
    end
    
    if p.withPnBrake
        
        fpb = u(:,2+p.withRgBrake);
    
    else
    
        fpb = 0;
    
    end
    
    % ODE
    
    rollingResistance = sr0 + sr1*sqrt(b) + sr2*b;
    
    acceleration = ftr - frb - fpb - rollingResistance - p.g*p.gradients(i)*(1/p.rho);
    
    tDot = 1./sqrt(b);
    bDot = 2*acceleration;
    xdot = [tDot, bDot];
    
    % output
    
    phaseout(i).dynamics = xdot;
    
    phaseout(i).path = [acceleration, ftr.*sqrt(b)];
    
    if p.withRgBrake
        
        phaseout(i).path = horzcat(phaseout(i).path, frb.*sqrt(b));
    
    end 
    
    phaseout(i).integrand = (1/p.etaTraction)*ftr - p.etaBrake.*frb;

end
