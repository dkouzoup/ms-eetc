
function output = trainEndpoint(input)

p = input.auxdata;
numPhases = length(p.gradients);

output.objective = 0;

for i = 1:numPhases 
    
    output.objective = output.objective + input.phase(i).integral;

end

for i = 2:numPhases

    xPrev = input.phase(i-1).finalstate;
    xNext = input.phase(i).initialstate;

    output.eventgroup(i-1).event = xNext-xPrev;

end