const initCanvas = () => {
    // Mat4 Library from GLMatrix
    const mat4 = glMatrix.mat4;

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
    // prettier-ignore
    const vertices = [
        // x, y, z, r, g, b
        0.0, 0.5, 0.0, 0.8, 0.1, 0.8,
        -1, -0.5, 0.0, 0.1, 0.8, 0.6,
        1, -0.5, 0.0, 0.8, 0.2, 0.1,
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
        3, // Number of ELements per Attribute -> Vec2, Vec3, Vec4
        gl.FLOAT, // Element type
        false, // Normalized or not
        6 * Float32Array.BYTES_PER_ELEMENT, // Size of individual vertex
        0 // Offset from the start of the first vertex to this attribute
    );

    // Get the location of the color param attribute in the shader function
    const colorAttrPos = gl.getAttribLocation(program, "vertColor");
    gl.vertexAttribPointer(
        colorAttrPos, // Attribute Position
        3, // Number of ELements per Attribute -> Vec2, Vec3, Vec4
        gl.FLOAT, // Element type
        false, // Normalized or not
        6 * Float32Array.BYTES_PER_ELEMENT, // Size of individual vertex
        3 * Float32Array.BYTES_PER_ELEMENT // Offset from the start of the first vertex to this attribute
    );

    // Enable the attribute to use
    gl.enableVertexAttribArray(locationAttrPos);
    gl.enableVertexAttribArray(colorAttrPos);

    // Use the program
    gl.useProgram(program);

    // Get the uniform matrices
    const mWorldPos = gl.getUniformLocation(program, "mWorld");
    const mViewPos = gl.getUniformLocation(program, "mView");
    const mProjPos = gl.getUniformLocation(program, "mProj");

    // Set the value
    let worldMatrix = new Float32Array(16);
    let viewMatrix = new Float32Array(16);
    let projMatrix = new Float32Array(16);

    // Set the matrices
    // Identity matrix for World
    mat4.identity(worldMatrix);
    // lookAt processing, the last elmt in the first vector will be zoom-in-out value
    mat4.lookAt(viewMatrix, [0, 0, -4], [0, 0, 0], [0, 1, 0]);
    // Perspective processing for Projection Matrix
    mat4.perspective(
        projMatrix,
        glMatrix.glMatrix.toRadian(45),
        cubeCanvas.width / cubeCanvas.height,
        0.1,
        1000.0
    );

    // Send the matrices as parameters to the program
    gl.uniformMatrix4fv(mWorldPos, false, worldMatrix);
    gl.uniformMatrix4fv(mViewPos, false, viewMatrix);
    gl.uniformMatrix4fv(mProjPos, false, projMatrix);

    // Render loop
    const identityMatrix = new Float32Array(16);
    mat4.identity(identityMatrix);
    let rotAngle;

    const renderLoop = () => {
        // Rotate on y-axis
        rotAngle = (performance.now() / 1000 / 6) * 2 * Math.PI;
        mat4.rotate(worldMatrix, identityMatrix, rotAngle, [0, 1, 0]);
        gl.uniformMatrix4fv(mWorldPos, false, worldMatrix);

        // Draw the vertices to create the triangle
        // mode: TRIANGLES, POINTS, LINES, TRIANGLES_STRIP, TRIANGLES_FAN
        // start: where the drawing start from the buffer array
        // count: number of vertices drawn
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        requestAnimationFrame(renderLoop);
    };

    requestAnimationFrame(renderLoop);
};

// GLSL Shader Text
// Attribute is like parameters
// Varying is like return
// View -> Camera, World -> Window, Proj -> Projection
const vertexShaderText = [
    `precision mediump float;
    attribute vec3 vertPosition;
    attribute vec3 vertColor;
    varying vec3 fragColor;
    uniform mat4 mWorld;
    uniform mat4 mView;
    uniform mat4 mProj;
    void main(){
        fragColor = vertColor;
        gl_Position = mProj * mView * mWorld * vec4(vertPosition, 1.0);
    }`,
].join("\n");

const fragmentShaderText = [
    `precision mediump float;
    varying vec3 fragColor;
    void main(){
    gl_FragColor = vec4(fragColor, 0.8);
    }`,
].join("\n");
