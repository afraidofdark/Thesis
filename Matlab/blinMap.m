function [po] = blinMap(pi, quat)
    % quad placement
    % q1  q2
    % q3  q4
    
    a1 = quat(1,:) + (quat(2,:) - quat(1,:)) * pi(1);
    a2 = quat(3,:) + (quat(4,:) - quat(3,:)) * pi(1);
    po = a2 + (a1 - a2) * pi(2);
end