<shader>
  <type name = "vertexShader" />
  <uniform name = "ProjectViewModel" />
  <uniform name = "InverseTransModel" />
  <uniform name = "Model" />
  <source>
    #version 300 es

    // Fixed Attributes.
    in vec3 vPosition;
    in vec3 vNormal;
    in vec2 vTexture;
    in vec3 vBiTan;

    // Declare used variables in the xml unifrom attribute.
    uniform mat4 ProjectViewModel;
    uniform mat4 InverseTransModel;
    uniform mat4 Model;

    out vec4 v_pos;
    out vec3 v_normal;
    out vec3 v_bitan;
    out vec2 v_texture;
    
    void main()
    {
      v_pos = Model * vec4(vPosition, 1.0);
      v_texture = vTexture;
      
      v_normal = (InverseTransModel * vec4(vNormal, 1.0)).xyz;
      v_bitan = (InverseTransModel * vec4(vBiTan, 1.0)).xyz;

      gl_Position = ProjectViewModel * vec4(vPosition, 1.0);
    }
  </source>
</shader>