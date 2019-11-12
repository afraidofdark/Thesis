<shader>
<type name = "fragmentShader" />
<uniform name = "LightData" />
<uniform name = "CamData" />
<source> <!--
#version 300 es
precision mediump float;

#define PARAM_SET_B
#define SC 80

const float PI = 3.1415926535897932384626433832795;

// Fixed Declaretions
struct _LightData
{
  vec3 pos;
  vec3 dir;
  vec3 color;
};

struct _CamData
{
  vec3 pos;
  vec3 dir;
};

uniform _LightData LightData;
uniform _CamData CamData;
uniform sampler2D s_texture;

in vec4 v_pos;
in vec3 v_normal;
in vec3 v_bitan;
in vec2 v_texture;

out vec4 fragColor;

mat4 rotationMatrix(vec3 axis, float angle)
{
  axis = normalize(axis);
  float s = sin(angle);
  float c = cos(angle);
  float oc = 1.0 - c;

  return mat4
  (
    oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 0.0,
    oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, 0.0,
    oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c, 0.0,
    0.0, 0.0, 0.0, 1.0
  );
}

vec3 rotate(vec3 v, vec3 axis, float angle)
{
  mat4 m = rotationMatrix(axis, angle);
  return (m * vec4(v, 1.0)).xyz;
}

struct MicroParams
{
  vec3 A, n, t, b, i, r;
  float ni, nv, kd, ys, yv;
};

#ifdef PARAM_SET_A
#define TOFP 2
#define TOFV 2
#endif

#ifdef PARAM_SET_B
#define TOFP 4
#define TOFV 2
#endif

#ifdef PARAM_SET_C
#define TOFP 8
#define TOFV 2
#endif

#ifdef PARAM_SET_D
#define TOFP 8
#define TOFV 2
#endif

#ifdef PARAM_SET_E
#define TOFP 4
#define TOFV 2
#endif

#ifdef PARAM_SET_F
#define TOFP 2
#define TOFV 4
#endif

#define TLENP TOFP - 1
#define TLENV TOFV - 1

void fillParams(
  out MicroParams paramsP, out float tanOffsetsP[TOFP], out float tanLensP[TLENP], out float a1,
  out MicroParams paramsV, out float tanOffsetsV[TOFV], out float tanLensV[TLENV], out float a2,
  vec3 n, vec3 t, vec3 b, vec3 i, vec3 r
)
{
#ifdef PARAM_SET_A
  paramsP.n = n;
  paramsP.t = rotate(t, n, radians(135.0));
  paramsP.b = rotate(b, n, radians(135.0));
  paramsP.i = i;
  paramsP.r = r;
  paramsP.A = vec3(0.2, 0.8, 1.0) * 0.3;
  paramsP.ni = 1.0;
  paramsP.nv = 1.46;
  paramsP.kd = 0.3;
  paramsP.ys = radians(12.0);
  paramsP.yv = radians(24.0);

  tanOffsetsP[0] = radians(-25.0);
  tanOffsetsP[1] = radians(25.0);
  a1 = 0.33;

  tanLensP[0] = 1.0;

  paramsV.n = n;
  paramsV.t = rotate(t, n, radians(45.0));
  paramsV.b = rotate(b, n, radians(45.0));
  paramsV.i = i;
  paramsV.r = r;
  paramsV.A = vec3(0.2, 0.8, 1.0) * 0.3;
  paramsV.ni = 1.0;
  paramsV.nv = 1.46;
  paramsV.kd = 0.3;
  paramsV.ys = radians(12.0);
  paramsV.yv = radians(24.0);

  tanOffsetsV[0] = radians(-25.0);
  tanOffsetsV[1] = radians(25.0);
  a2 = 0.33;

  tanLensV[0] = 1.0;
#endif

#ifdef PARAM_SET_B
  paramsP.n = n;
  paramsP.t = rotate(t, n, radians(135.0));
  paramsP.b = rotate(b, n, radians(135.0));
  paramsP.i = i;
  paramsP.r = r;
  paramsP.A = vec3(1.0, 0.95, 0.05) * 0.12;
  paramsP.ni = 1.0;
  paramsP.nv = 1.345;
  paramsP.kd = 0.2;
  paramsP.ys = radians(5.0);
  paramsP.yv = radians(10.0);

  tanOffsetsP[0] = radians(-35.0);
  tanOffsetsP[1] = radians(-35.0);
  tanOffsetsP[2] = radians(35.0);
  tanOffsetsP[3] = radians(35.0);
  a1 = 0.75;

  tanLensP[0] = 1.0;
  tanLensP[1] = 1.0;
  tanLensP[2] = 1.0;

  paramsV.n = n;
  paramsV.t = rotate(t, n, radians(45.0));
  paramsV.b = rotate(b, n, radians(45.0));
  paramsV.i = i;
  paramsV.r = r;
  paramsV.A = vec3(1.0, 0.95, 0.05) * 0.16;
  paramsV.ni = 1.0;
  paramsV.nv = 1.345;
  paramsV.kd = 0.3;
  paramsV.ys = radians(18.0);
  paramsV.yv = radians(32.0);

  tanOffsetsV[0] = radians(0.0);
  tanOffsetsV[1] = radians(0.0);
  a2 = 0.25;

  tanLensV[0] = 1.0;
#endif

#ifdef PARAM_SET_C
  paramsP.n = n;
  paramsP.t = rotate(t, n, radians(90.0));
  paramsP.b = rotate(b, n, radians(90.0));
  paramsP.i = i;
  paramsP.r = r;
  paramsP.A = vec3(1.0, 0.37, 0.3) * 0.035;
  paramsP.ni = 1.0;
  paramsP.nv = 1.539;
  paramsP.kd = 0.1;
  paramsP.ys = radians(2.5);
  paramsP.yv = radians(5.0);

  tanOffsetsP[0] = radians(32.0);
  tanOffsetsP[1] = radians(32.0);
  tanOffsetsP[2] = radians(18.0);
  tanOffsetsP[3] = radians(0.0);
  tanOffsetsP[4] = radians(0.0);
  tanOffsetsP[5] = radians(-18.0);
  tanOffsetsP[6] = radians(-32.0);
  tanOffsetsP[7] = radians(-32.0);
  a1 = 0.9;

  tanLensP[0] = 1.33;
  tanLensP[1] = 0.66;
  tanLensP[2] = 2.0;
  tanLensP[3] = 2.0;
  tanLensP[4] = 2.0;
  tanLensP[5] = 0.66;
  tanLensP[6] = 1.33;

  paramsV.n = n;
  paramsV.t = rotate(t, n, radians(0.0));
  paramsV.b = rotate(b, n, radians(0.0));
  paramsV.i = i;
  paramsV.r = r;
  paramsV.A = vec3(1.0, 0.37, 0.3) * 0.2;
  paramsV.ni = 1.0;
  paramsV.nv = 1.539;
  paramsV.kd = 0.7;
  paramsV.ys = radians(30.0);
  paramsV.yv = radians(60.0);

  tanOffsetsV[0] = radians(0.0);
  tanOffsetsV[1] = radians(0.0);
  a2 = 0.1;

  tanLensV[0] = 1.0;
#endif

#ifdef PARAM_SET_D
  paramsP.n = n;
  paramsP.t = rotate(t, n, radians(135.0));
  paramsP.b = rotate(b, n, radians(135.0));
  paramsP.i = i;
  paramsP.r = r;
  paramsP.A = vec3(1.0, 0.37, 0.3) * 0.035;
  paramsP.ni = 1.0;
  paramsP.nv = 1.539;
  paramsP.kd = 0.1;
  paramsP.ys = radians(2.5);
  paramsP.yv = radians(5.0);

  tanOffsetsP[0] = radians(-30.0);
  tanOffsetsP[1] = radians(-30.0);
  tanOffsetsP[2] = radians(30.0);
  tanOffsetsP[3] = radians(30.0);
  tanOffsetsP[4] = radians(-5.0);
  tanOffsetsP[5] = radians(-5.0);
  tanOffsetsP[6] = radians(5.0);
  tanOffsetsP[7] = radians(5.0);
  a1 = 0.67;

  tanLensP[0] = 1.33;
  tanLensP[1] = 1.33;
  tanLensP[2] = 1.33;
  tanLensP[3] = 0.0;
  tanLensP[4] = 0.67;
  tanLensP[5] = 0.67;
  tanLensP[6] = 0.67;

  paramsV.n = n;
  paramsV.t = rotate(t, n, radians(45.0));
  paramsV.b = rotate(b, n, radians(45.0));
  paramsV.i = i;
  paramsV.r = r;
  paramsV.A = vec3(1.0, 0.37, 0.3) * 0.2;
  paramsV.ni = 1.0;
  paramsV.nv = 1.46;
  paramsV.kd = 0.7;
  paramsV.ys = radians(30.0);
  paramsV.yv = radians(60.0);

  tanOffsetsV[0] = radians(0.0);
  tanOffsetsV[1] = radians(0.0);
  a2 = 0.33;

  tanLensV[0] = 3.0;
#endif

#ifdef PARAM_SET_E
  paramsP.n = n;
  paramsP.t = rotate(t, n, radians(135.0));
  paramsP.b = rotate(b, n, radians(135.0));
  paramsP.i = i;
  paramsP.r = r;
  paramsP.A = vec3(0.1, 1.0, 0.4) * 0.2;
  paramsP.ni = 1.0;
  paramsP.nv = 1.345;
  paramsP.kd = 0.1;
  paramsP.ys = radians(4.0);
  paramsP.yv = radians(8.0);

  tanOffsetsP[0] = radians(-25.0);
  tanOffsetsP[1] = radians(-25.0);
  tanOffsetsP[2] = radians(25.0);
  tanOffsetsP[3] = radians(25.0);
  a1 = 0.86;

  tanLensP[0] = 1.33;
  tanLensP[1] = 2.67;
  tanLensP[2] = 1.33;

  paramsV.n = n;
  paramsV.t = rotate(t, n, radians(45.0));
  paramsV.b = rotate(b, n, radians(45.0));
  paramsV.i = i;
  paramsV.r = r;
  paramsV.A = vec3(1.0, 0.0, 0.1) * 0.6;
  paramsV.ni = 1.0;
  paramsV.nv = 1.345;
  paramsV.kd = 0.1;
  paramsV.ys = radians(5.0);
  paramsV.yv = radians(10.0);

  tanOffsetsV[0] = radians(0.0);
  tanOffsetsV[1] = radians(0.0);
  a2 = 0.14;

  tanLensV[0] = 1.0;
#endif

#ifdef PARAM_SET_F
  paramsP.n = n;
  paramsP.t = rotate(t, n, radians(135.0));
  paramsP.b = rotate(b, n, radians(135.0));
  paramsP.i = i;
  paramsP.r = r;
  paramsP.A = vec3(0.75, 0.02, 0.0) * 0.3;
  paramsP.ni = 1.0;
  paramsP.nv = 1.46;
  paramsP.kd = 0.1;
  paramsP.ys = radians(6.0);
  paramsP.yv = radians(12.0);

  tanOffsetsP[0] = radians(-90.0);
  tanOffsetsP[1] = radians(-50.0);
  a1 = 0.5;

  tanLensP[0] = 1.0;

  paramsV.n = n;
  paramsV.t = rotate(t, n, radians(45.0));
  paramsV.b = rotate(b, n, radians(45.0));
  paramsV.i = i;
  paramsV.r = r;
  paramsV.A = vec3(0.55, 0.02, 0.0) * 0.3;
  paramsV.ni = 1.0;
  paramsV.nv = 1.46;
  paramsV.kd = 0.1;
  paramsV.ys = radians(6.0);
  paramsV.yv = radians(12.0);

  tanOffsetsV[0] = radians(-90.0);
  tanOffsetsV[1] = radians(-55.0);
  tanOffsetsV[0] = radians(55.0);
  tanOffsetsV[1] = radians(90.0);
  a2 = 0.5;

  tanLensV[0] = 0.5;
  tanLensV[1] = 0.0;
  tanLensV[2] = 0.5;
#endif
}

float refract(float ia, float etai, float etat)
{
  float cosi = cos(ia);
  float sint = (etai / etat) * sqrt(max(0.0, 1.0 - cosi * cosi));
  return acos(sqrt(max(0.0, 1.0 - sint * sint)));
}

float fresnel(float ia, float etai, float etat)
{
  float cosi = cos(ia);
  if (cosi < 0.0)
  {
    float tmp = etai;
    etai = etat;
    etat = tmp;
  }

  float kr = 0.0;
  float sint = (etai / etat) * sqrt(max(0.0, 1.0 - cosi * cosi));
  if (sint > 1.0)
  {
    kr = 1.0;
  }
  else
  {
    float cost = sqrt(max(0.0, 1.0 - sint * sint));
    cosi = abs(cosi);
    float Rpar = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rper = ((etai * cost) - (etat * cosi)) / ((etai * cost) + (etat * cosi));
    kr = (Rpar * Rpar + Rper * Rper) / 2.0;
  }

  return kr;
}

float unitLengthGauss(float x, float s)
{
  return exp(-(x * x) / (s * s));
}

float unitAreaGauss(float x, float s)
{
  return unitLengthGauss(x, s) / (sqrt(PI) * s);
}

float bravais(float projectedAngle, float n)
{
  float sprj = sin(projectedAngle);
  float cprj = cos(projectedAngle);
  return sqrt(n * n - sprj * sprj) / cprj;
}

float adjust(float a, float b, float c)
{
  return (1.0 - a) * b * c + a * min(b, c);
}

struct SphericalCoordArgs
{
  vec3 n, b, t, i, r, ip, rp;
  float ti, tr, th, td, pi, pr, pd;
};

void project(in out SphericalCoordArgs args)
{
  args.ip = normalize(args.n * dot(args.i, args.n) + args.b * dot(args.i, args.b));
  args.rp = normalize(args.n * dot(args.r, args.n) + args.b * dot(args.r, args.b));
}

void theta(in out SphericalCoordArgs args)
{
  args.ti = acos(dot(args.ip, args.i));
  args.ti *= sign(dot(args.i, args.t));
  args.tr = acos(dot(args.rp, args.r));
  args.tr *= sign(dot(args.r, args.t));
  args.th = (args.ti + args.tr) / 2.0;
  args.td = (args.ti - args.tr) / 2.0;
}

void phi(in out SphericalCoordArgs args)
{
  args.pi = acos(dot(args.ip, args.n));
  args.pi *= sign(dot(args.ip, args.b));
  args.pr = acos(dot(args.rp, args.n));
  args.pr *= sign(dot(args.rp, args.b));
  args.pd = args.pi - args.pr;
}

vec3 micro(MicroParams params, in out float rw)
{
  SphericalCoordArgs args;
  args.n = params.n;
  args.b = params.b;
  args.t = params.t;
  args.i = params.i;
  args.r = params.r;

  project(args);
  theta(args);
  phi(args);

  //params.nv = bravais(args.ti, params.nv);
  float ia = acos(cos(args.td) * cos(args.pd / 2.0));
  float frs = fresnel(ia, params.ni, params.nv) * cos(args.pd / 2.0) * unitAreaGauss(args.th, params.ys);
  float F = (1.0 - fresnel(acos(dot(args.i, args.n)), params.ni, params.nv)) * (1.0 - fresnel(acos(dot(-args.r, args.n)), params.nv, params.ni));
  vec3 frv = F * (((1.0 - params.kd) * unitAreaGauss(args.th, params.yv) + params.kd) / (cos(args.ti) + cos(args.tr))) * params.A;
  vec3 res = cos(args.ti) * (vec3(frs) + frv) / (cos(args.td) * cos(args.td));

  float sm = adjust(unitLengthGauss(args.pd, radians(20.0)), max(cos(args.pi), 0.0), max(cos(args.pr), 0.0)); // Shadowing & masking

  // args.b = params.t * rw;
  // project(args);
  // phi(args);

  rw = 1.0; // adjust(unitLengthGauss(args.pd, radians(20.0)), max(cos(args.pi), 0.0), max(cos(args.pr), 0.0)); // Reweighting

  return rw * sm * res;
}

void toSpherical(vec3 n, out float zenith, out float azimuth)
{
  float r = length(n);
  azimuth = atan(n.x, n.z);
  zenith = acos(n.y / r);
}

vec3 appearance(MicroParams paramsP, MicroParams paramsV, float tanOffsetsP[TOFP], float tanLensP[TLENP], float tanOffsetsV[TOFV], float tanLensV[TLENV], float a1, float a2)
{
  MicroParams bckp = paramsP;
  vec3 sump = vec3(0.0, 0.0, 0.0);
  float trw = 0.0;

  // Create sampels per tangent curve.
  float curveLen = 0.0f;
  for (int i = 0; i < TLENP; i++)
  {
    curveLen += tanLensP[i];
  }

  float k = float(SC) / curveLen;
  int counter = 0;
  for (int i = 0; i < TLENP; i++)
  {
    if (tanLensP[i] == 0.0)
      continue;

    int c = int(round(k * tanLensP[i]));
    float ainc = (tanOffsetsP[i + 1] - tanOffsetsP[i]) / float(c);

    // Sum micro cylinder result per sample.
    for (int j = 0; j < c; j++)
    {
      float offAngle = tanOffsetsP[i] + ainc * float(j);
      paramsP.n = rotate(bckp.n, bckp.b, offAngle);
      paramsP.t = rotate(bckp.t, bckp.b, offAngle);
      float rw = tanLensP[i];
      sump += micro(paramsP, rw);
      trw += rw; // Reweighting is missing ! Implement it.
      counter++;
    }
  }

  // Average it.
  sump /= float(counter);
  counter = 0;

  // Do same for second yarn.
  vec3 sumv = vec3(0.0, 0.0, 0.0);
  bckp = paramsV;
  trw = 0.0;
  for (int i = 0; i < TLENV; i++)
  {
    if (tanLensV[i] == 0.0)
      continue;

    int c = int(round(k * tanLensV[i]));
    float ainc = (tanOffsetsV[i + 1] - tanOffsetsV[i]) / float(c);

    for (int j = 0; j < c; j++)
    {
      float offAngle = tanOffsetsV[i] + ainc * float(j);
      paramsV.n = rotate(bckp.n, bckp.b, offAngle);
      paramsV.t = rotate(bckp.t, bckp.b, offAngle);
      float rw = tanLensV[i];
      sumv += micro(paramsV, rw);
      trw += rw;
      counter++;
    }
  }

  sumv /= float(counter);

  return a1 * sump + a2 * sumv;
}

void main()
{
  vec3 n = normalize(v_normal);
  vec3 t = normalize(-v_bitan);
  vec3 b = normalize(cross(n, t));
  vec3 i = normalize(LightData.pos - v_pos.xyz);
  vec3 r = normalize(-CamData.dir);

  MicroParams paramsP;
  float tanOffsetsP[TOFP];
  float tanLensP[TLENP];
  float a1;

  MicroParams paramsV;
  float tanOffsetsV[TOFV];
  float tanLensV[TLENV];
  float a2;

  fillParams(
    paramsP, tanOffsetsP, tanLensP, a1,
    paramsV, tanOffsetsV, tanLensV, a2,
    n, t, b, i, r
  );

  vec3 val = appearance(paramsP, paramsV, tanOffsetsP, tanLensP, tanOffsetsV, tanLensV, a1, a2);

  //fragColor = vec4(val * PI, 1.0);
  float gamma = 2.2;
  fragColor.rgb = pow(val, vec3(1.0 / gamma));
}
--></source>
</shader>