const initCanvas = () => {
    // Get All Canvasses
    /** @type {HTMLCanvasElement} */
    const cubeCanvas = document.querySelector(".canvas-cube");

    // Get WebGL Context
    /** @type {WebGLRenderingContext} */
    const gl = cubeCanvas.getContext("webgl");

    // Create the shaders
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);

    // Absorb the GLSL into the shader
    gl.shaderSource(vertexShader, vertexShaderText);
    gl.shaderSource(fragmentShader, fragmentShaderText);

    // Compile the shader
    gl.compileShader(vertexShader);
    gl.compileShader(fragmentShader);

    // Check any error in the GLSL syntaxes
    if (
        !gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) ||
        !gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)
    ) {
        console.log("Error Compiling Shader");
    }

    // Create a program and attach the shader
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);

    // Link the program and check if there is any error
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.log("Error Linking Program");
    }

    // Define Triangle Vertices
    // Sequentially coordinates first then colors
    // Top, left, right, counterclockwise sequence
    const vertices = [
        0.0, 0.5, -0.5, -0.5, 0.5, -0.5, 0.4, 0.1, 0.8, 0.1, 0.8, 0.6, 0.8, 0.2, 0.1,
    ];

    // Create and bind the Buffer
    const triangleBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, triangleBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    // Get the location of the position param attribute in the shader function
    // 0 for the first param
    const locationAttrPos = gl.getAttribLocation(program, "vertPosition");
    gl.vertexAttribPointer(
        locationAttrPos, // Attribute Position
        2, // Number of ELements per Attribute -> Vec2, Vec3, Vec4
        gl.FLOAT, // Element type
        false, // Normalized or not
        2 * Float32Array.BYTES_PER_ELEMENT, // Size of individual vertex
        0 // Offset from the start of the first vertex to this attribute
    );

    // Get the location of the color param attribute in the shader function
    const colorAttrPos = gl.getAttribLocation(program, "vertColor");
    gl.vertexAttribPointer(
        colorAttrPos, // Attribute Position
        3, // Number of ELements per Attribute -> Vec2, Vec3, Vec4
        gl.FLOAT, // Element type
        false, // Normalized or not
        5 * Float32Array.BYTES_PER_ELEMENT, // Size of individual vertex
        2 * Float32Array.BYTES_PER_ELEMENT // Offset from the start of the first vertex to this attribute
    );

    // Enable the attribute to use
    gl.enableVertexAttribArray(locationAttrPos);
    gl.enableVertexAttribArray(colorAttrPos);

    // Render loop
    gl.useProgram(program);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
};

// GLSL Shader Text
// Attribute is like parameters
// Varying is like return
const vertexShaderText = [
    `precision mediump float;
    attribute vec2 vertPosition;
    attribute vec3 vertColor;
    varying vec3 fragColor;
    void main(){
        fragColor = vertColor;
        gl_Position = vec4(vertPosition, 0.0, 1.0);
    }`,
].join("\n");

const fragmentShaderText = [
    `precision mediump float;
    varying vec3 fragColor;
    void main(){
    gl_FragColor = vec4(fragColor, 0.8);
    }`,
].join("\n");
