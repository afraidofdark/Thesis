<shader>
  <type name = "fragmentShader" />
  <uniform name = "LightData" />
  <uniform name = "CamData" />
  <source>
    #version 300 es
    precision mediump float;

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

    void main()
    {
      vec3 n = normalize(v_normal);
      vec3 t = normalize(v_bitan);
      vec3 b = normalize(cross(n, t));
      vec3 l = normalize(LightData.pos - v_pos.xyz);
      vec3 e = normalize(CamData.pos - v_pos.xyz);

      // Kajiya part (Zenith)
      float kd = 0.9 * sin(acos(dot(t, l)));
      float ks = 0.1 * pow(dot(t, l) * dot(t, e) + sin(acos(dot(t, l))) * sin(acos(dot(t, e))), 300.0);
      ks = clamp(ks, 0.0, 1.0);
      
      // Kim part (Azimuth)
      mat3 globalToLocal = transpose(mat3(b, n, t));
      vec2 lp = normalize((globalToLocal * l).xy);
      vec2 ep = normalize((globalToLocal * e).xy);
      float azimuth = acos(dot(lp, ep));

      const float pr = 1.0;
      const float pt = 1.0;
      const float k = 3.0;

      float kkim = pr * cos(azimuth * 0.5);
      if (abs(azimuth) >= PI * (1.0 - (1.0 / (2.0 * k))))
      {
        kkim += pt * cos(k * (azimuth - PI));
      }

      vec4 pixColor = pow(texture(s_texture, v_texture), vec4(1.0 / 2.2));
      pixColor = pixColor * kd + vec4(LightData.color, 1.0) * ks;
      pixColor = pixColor * kkim;
      pixColor = pow(pixColor, vec4(2.2));
      pixColor.w = 1.0;

      fragColor = pixColor;
    }
  </source>
</shader>