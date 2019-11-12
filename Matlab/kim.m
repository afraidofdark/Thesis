pr = 1.0;
pt = 1.0;
k = 3.0;
t = [];
r = [];

for i = -pi : 0.01 : pi
  val = pr * cos(i * 0.5);
  if abs(i) >= pi * (1 - (1 / (2 * k)))
    val = val + pt * cos(k * (i - pi));
  end

  t = [t, val];
  r = [r, i];
end

figure
polarplot(r, t);