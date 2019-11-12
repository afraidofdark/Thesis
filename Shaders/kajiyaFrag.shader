<shader>
  <type name = "fragmentShader" />
  <uniform name = "LightData" />
  <uniform name = "CamData" />
  <source>
    #version 300 es
    precision mediump float;

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
      vec3 l = normalize(LightData.pos - v_pos.xyz);
      vec3 e = normalize(CamData.pos - v_pos.xyz);
      float kd = 0.9 * sin(acos(dot(t, l)));
      float ks = 0.1 * pow(dot(t, l) * dot(t, e) + sin(acos(dot(t, l))) * sin(acos(dot(t, e))), 300.0);
      ks = clamp(ks, 0.0, 1.0);
      
      vec4 pixColor = texture(s_texture, v_texture);
      pixColor = pow(pixColor, vec4(1.0 / 2.2))  * kd + vec4(LightData.color, 1.0) * ks;
      pixColor = pow(pixColor, vec4(2.2));
      pixColor.w = 1.0;

      fragColor = pixColor;
    }
  </source>
</shader>