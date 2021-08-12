// Draw the canvas
const drawCanvas = () => {
    drawBoxie();
    drawPyra();
};

// Get Canvas
const getCanvas = (canvasName) => {
    return document.querySelector(canvasName);
};

// Get WebGL Context
const getContext = (canvas) => {
    return canvas.getContext("webgl");
};

// Fill and Compile Shader
const compileVertexShader = (glTarget, vertexShader, vertexShaderText) => {
    // Absorb the GLSL into the shader
    glTarget.shaderSource(vertexShader, vertexShaderText);
    // Compile the shader
    glTarget.compileShader(vertexShader);
    // Check any error in the GLSL syntaxes
    if (!glTarget.getShaderParameter(vertexShader, glTarget.COMPILE_STATUS)) {
        console.log("Error Compiling Vertex Shader");
    }
};

// Fill and Compile Shader
const compileFragmentShader = (glTarget, fragmentShader, fragmentShaderText) => {
    // Absorb the GLSL into the shader
    glTarget.shaderSource(fragmentShader, fragmentShaderText);
    // Compile the shader
    glTarget.compileShader(fragmentShader);
    // Check any error in the GLSL syntaxes
    if (!glTarget.getShaderParameter(fragmentShader, glTarget.COMPILE_STATUS)) {
        console.log("Error Compiling Fragment Shader");
    }
};

// Attach and link shaders with program
const attachAndLink = (glTarget, vertexShader, fragmentShader, program) => {
    // Attach the shaders to the program
    glTarget.attachShader(program, vertexShader);
    glTarget.attachShader(program, fragmentShader);

    // Link the program and check if there is any error
    glTarget.linkProgram(program);
    if (!glTarget.getProgramParameter(program, glTarget.LINK_STATUS)) {
        console.log("Error Linking Program");
    }
};

// Boxie
const drawBoxie = () => {
    // Define the Mat4 from glMatrix to make Boxie more visible
    const mat4 = glMatrix.mat4;

    /** @type {HTMLCanvasElement} */
    const cubeCanvas = getCanvas(".canvas-cube");

    /** @type {WebGLRenderingContext} */
    const glBoxie = getContext(cubeCanvas);

    // Enable depth and Cull Face, no invisible edge when rotating
    // Reduce computation behind the scene
    glBoxie.enable(glBoxie.DEPTH_TEST);
    glBoxie.enable(glBoxie.CULL_FACE);
    glBoxie.frontFace(glBoxie.CCW);
    glBoxie.cullFace(glBoxie.BACK);

    // Create the shaders
    const vertexShader = glBoxie.createShader(glBoxie.VERTEX_SHADER);
    const fragmentShader = glBoxie.createShader(glBoxie.FRAGMENT_SHADER);

    // Compile and absorb the shaders based on GLSL code
    compileVertexShader(glBoxie, vertexShader, vertexShaderText);
    compileFragmentShader(glBoxie, fragmentShader, fragmentShaderText);

    // Create a program and attach the shader
    const program = glBoxie.createProgram();
    attachAndLink(glBoxie, vertexShader, fragmentShader, program);

    // Define Boxie Vertices
    // prettier-ignore
    const vertices = [
        // Top
        -1.0, 1.0, -1.0, 0.5, 0.5, 0.5,
        -1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
        1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
        1.0, 1.0, -1.0, 0.5, 0.5, 0.5,
		// Left
		-1.0, 1.0, 1.0, 0.75, 0.25, 0.5,
		-1.0, -1.0, 1.0, 0.75, 0.25, 0.5,
		-1.0, -1.0, -1.0, 0.75, 0.25, 0.5,
		-1.0, 1.0, -1.0, 0.75, 0.25, 0.5,
		// Right
		1.0, 1.0, 1.0, 0.5, 0.5, 0.75,
		1.0, -1.0, 1.0, 0.5, 0.5, 0.75,
		1.0, -1.0, -1.0, 0.5, 0.5, 0.75,
		1.0, 1.0, -1.0, 0.5, 0.5, 0.75,
		// Front
		1.0, 1.0, 1.0, 0.85, 0.25, 0.15,
		1.0, -1.0, 1.0, 0.85, 0.25, 0.15,
		-1.0, -1.0, 1.0, 0.85, 0.25, 0.15,
		-1.0, 1.0, 1.0, 0.85, 0.25, 0.15,
		// Back
		1.0, 1.0, -1.0, 0.6, 0.2, 0.75,
		1.0, -1.0, -1.0, 0.6, 0.2, 0.75,
		-1.0, -1.0, -1.0, 0.6, 0.2, 0.75,
		-1.0, 1.0, -1.0, 0.6, 0.2, 0.75,
		// Bottom
		-1.0, -1.0, -1.0, 0.7, 0.6, 0.3,
		-1.0, -1.0, 1.0, 0.7, 0.6, 0.3,
		1.0, -1.0, 1.0, 0.7, 0.6, 0.3,
		1.0, -1.0, -1.0, 0.7, 0.6, 0.3,
    ];

    // Define Boxie Indices
    // Basically, tell WebGL which vertices created a triangle
    // 0, 1, 2, means that the first three vertices in the array before is one triangle
    // prettier-ignore
    const indices = [
        // Top
		0, 1, 2,
		0, 2, 3,
		// Left
		5, 4, 6,
		6, 4, 7,
		// Right
		8, 9, 10,
		8, 10, 11,
		// Front
		13, 12, 14,
		15, 14, 12,
		// Back
		16, 17, 18,
		16, 18, 19,
		// Bottom
		21, 20, 22,
		22, 20, 23
    ]

    // Create, set, upload and bind the buffer with the data from above
    // Each for vertices and indices
    const vertexBuffer = glBoxie.createBuffer();
    glBoxie.bindBuffer(glBoxie.ARRAY_BUFFER, vertexBuffer);
    glBoxie.bufferData(
        glBoxie.ARRAY_BUFFER,
        new Float32Array(vertices),
        glBoxie.STATIC_DRAW
    );

    const indexBuffer = glBoxie.createBuffer();
    glBoxie.bindBuffer(glBoxie.ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBoxie.bufferData(
        glBoxie.ELEMENT_ARRAY_BUFFER,
        new Uint16Array(indices),
        glBoxie.STATIC_DRAW
    );

    // Get the location of the position param attribute in the shader function
    // 0 for the first param
    const locationAttrPos = glBoxie.getAttribLocation(program, "vertPosition");
    glBoxie.vertexAttribPointer(
        locationAttrPos, // Attribute Position
        3, // Number of ELements per Attribute -> Vec2, Vec3, Vec4
        glBoxie.FLOAT, // Element type
        false, // Normalized or not
        6 * Float32Array.BYTES_PER_ELEMENT, // Size of individual vertex
        0 // Offset from the start of the first vertex to this attribute
    );

    // Get the location of the color param attribute in the shader function
    const colorAttrPos = glBoxie.getAttribLocation(program, "vertColor");
    glBoxie.vertexAttribPointer(
        colorAttrPos, // Attribute Position
        3, // Number of ELements per Attribute -> Vec2, Vec3, Vec4
        glBoxie.FLOAT, // Element type
        false, // Normalized or not
        6 * Float32Array.BYTES_PER_ELEMENT, // Size of individual vertex
        3 * Float32Array.BYTES_PER_ELEMENT // Offset from the start of the first vertex to this attribute
    );

    // Enable the attribute to use
    glBoxie.enableVertexAttribArray(locationAttrPos);
    glBoxie.enableVertexAttribArray(colorAttrPos);

    // Use the program
    glBoxie.useProgram(program);

    // Get the uniform matrices
    const mWorldPos = glBoxie.getUniformLocation(program, "mWorld");
    const mViewPos = glBoxie.getUniformLocation(program, "mView");
    const mProjPos = glBoxie.getUniformLocation(program, "mProj");

    // Set the value
    let worldMatrix = new Float32Array(16);
    let viewMatrix = new Float32Array(16);
    let projMatrix = new Float32Array(16);

    // Set the matrices
    // Identity matrix for World
    mat4.identity(worldMatrix);
    // lookAt processing, the last elmt in the first vector will be zoom-in-out value
    mat4.lookAt(viewMatrix, [0, 0, -8], [0, 0, 0], [0, 1, 0]);
    // Perspective processing for Projection Matrix
    mat4.perspective(
        projMatrix,
        glMatrix.glMatrix.toRadian(45),
        cubeCanvas.clientWidth / cubeCanvas.clientHeight,
        0.1,
        1000.0
    );

    // Send the matrices as parameters to the program
    glBoxie.uniformMatrix4fv(mWorldPos, false, worldMatrix);
    glBoxie.uniformMatrix4fv(mViewPos, false, viewMatrix);
    glBoxie.uniformMatrix4fv(mProjPos, false, projMatrix);

    // Rotate around X and Z
    const yRotMatrix = new Float32Array(16);
    const zRotMatrix = new Float32Array(16);

    // Render loop
    const identityMatrix = new Float32Array(16);
    mat4.identity(identityMatrix);
    let rotAngle;

    const renderLoop = () => {
        // Rotate on y-axis
        rotAngle = (performance.now() / 1000 / 6) * 2 * Math.PI;
        mat4.rotate(yRotMatrix, identityMatrix, rotAngle, [0, 1, 0]);
        mat4.rotate(zRotMatrix, identityMatrix, rotAngle, [0, 0, 1]);
        mat4.mul(worldMatrix, yRotMatrix, zRotMatrix);
        glBoxie.uniformMatrix4fv(mWorldPos, false, worldMatrix);

        // Draw the vertices with pre-defined indices to create the triangle
        // mode: TRIANGLES, POINTS, LINES, TRIANGLES_STRIP, TRIANGLES_FAN
        // count: number of indices drawn
        // type: drawn type, unsigned short
        // offset
        glBoxie.drawElements(
            glBoxie.TRIANGLES,
            indices.length,
            glBoxie.UNSIGNED_SHORT,
            0
        );

        requestAnimationFrame(renderLoop);
    };

    requestAnimationFrame(renderLoop);
};

// Draw Pyra
const drawPyra = () => {
    // Define the Mat4 from glMatrix to make Boxie more visible
    const mat4 = glMatrix.mat4;

    // Get Canvas
    /** @type {HTMLCanvasElement} */
    const pyraCanvas = getCanvas(".canvas-pyra");

    // Get WebGL Context
    /** @type {WebGLRenderingContext} */
    const glPyra = getContext(pyraCanvas);

    // Enable depth, no invisible edge when rotating
    glPyra.enable(glPyra.DEPTH_TEST);

    // Create the shaders
    const vertexShader = glPyra.createShader(glPyra.VERTEX_SHADER);
    const fragmentShader = glPyra.createShader(glPyra.FRAGMENT_SHADER);

    // Compile and absorb the shaders based on GLSL code
    compileVertexShader(glPyra, vertexShader, vertShaderTexture);
    compileFragmentShader(glPyra, fragmentShader, fragShaderTexture);

    // Create a program and attach the shader
    const program = glPyra.createProgram();
    attachAndLink(glPyra, vertexShader, fragmentShader, program);

    // Define Boxie Vertices
    // prettier-ignore
    const vertices = [
        // Bottom
        1.0, 1.0, 0.0, 1, 1,
		1.0, -1.0, 0.0, 1, 0,
		-1.0, -1.0, 0.0, 0, 0,
		-1.0, 1.0, 0.0, 0, 1,
        // 1st Triangle
        1.0, 1.0, 0.0, 0, 0,
		1.0, -1.0, 0.0, 1, 0,
		0.0, 0.0, 2.0, 1, 1,
        // 2nd Triangle
        -1.0, -1.0, 0.0, 0, 0,
		-1.0, 1.0, 0.0, 1, 0,
        0.0, 0.0, 2.0, 1, 1,
        // 3rd Triangle
        -1.0, -1.0, 0.0, 0, 0,
		1.0, -1.0, 0.0, 1, 0,
        0.0, 0.0, 2.0, 1, 1,
        // 4th Triangle
        1.0, 1.0, 0.0, 0, 0,
        -1.0, 1.0, 0.0, 1, 0,
        0.0, 0.0, 2.0, 1, 1,
    ];

    // Define Boxie Indices
    // Basically, tell WebGL which vertices created a triangle
    // 0, 1, 2, means that the first three vertices in the array before is one triangle
    // prettier-ignore
    const indices = [
        // Bottom
		0, 1, 2,
        0, 2, 3,
		// 1st Triangle
		4, 5, 6,
		// 2nd Triangle
		7, 8, 9,
		// 3rd Triangle
		10, 11, 12,
		// 4th Triangle
        13, 14, 15
    ]

    // Create, set, upload and bind the buffer with the data from above
    // Each for vertices and indices
    const vertexBuffer = glPyra.createBuffer();
    glPyra.bindBuffer(glPyra.ARRAY_BUFFER, vertexBuffer);
    glPyra.bufferData(
        glPyra.ARRAY_BUFFER,
        new Float32Array(vertices),
        glPyra.STATIC_DRAW
    );

    const indexBuffer = glPyra.createBuffer();
    glPyra.bindBuffer(glPyra.ELEMENT_ARRAY_BUFFER, indexBuffer);
    glPyra.bufferData(
        glPyra.ELEMENT_ARRAY_BUFFER,
        new Uint16Array(indices),
        glPyra.STATIC_DRAW
    );

    // Get the location of the position param attribute in the shader function
    // 0 for the first param
    const locationAttrPos = glPyra.getAttribLocation(program, "vertPosition");
    glPyra.vertexAttribPointer(
        locationAttrPos, // Attribute Position
        3, // Number of ELements per Attribute -> Vec2, Vec3, Vec4
        glPyra.FLOAT, // Element type
        false, // Normalized or not
        5 * Float32Array.BYTES_PER_ELEMENT, // Size of individual vertex
        0 // Offset from the start of the first vertex to this attribute
    );

    // Get the location of the color param attribute in the shader function
    const texCoordPos = glPyra.getAttribLocation(program, "vertTexCoord");
    glPyra.vertexAttribPointer(
        texCoordPos, // Attribute Position
        2, // Number of ELements per Attribute -> Vec2, Vec3, Vec4
        glPyra.FLOAT, // Element type
        false, // Normalized or not
        5 * Float32Array.BYTES_PER_ELEMENT, // Size of individual vertex
        3 * Float32Array.BYTES_PER_ELEMENT // Offset from the start of the first vertex to this attribute
    );

    // Enable the attribute to use
    glPyra.enableVertexAttribArray(locationAttrPos);
    glPyra.enableVertexAttribArray(texCoordPos);

    // Create Texture
    // Set the texture mode, CLAMP TO EDGE, REPEATED, or MIRROR
    // Absorb the source of the texture
    const texture = glPyra.createTexture();
    glPyra.bindTexture(glPyra.TEXTURE_2D, texture);
    glPyra.texParameteri(glPyra.TEXTURE_2D, glPyra.TEXTURE_WRAP_S, glPyra.CLAMP_TO_EDGE);
    glPyra.texParameteri(glPyra.TEXTURE_2D, glPyra.TEXTURE_WRAP_T, glPyra.CLAMP_TO_EDGE);
    glPyra.texParameteri(glPyra.TEXTURE_2D, glPyra.TEXTURE_MIN_FILTER, glPyra.LINEAR);
    glPyra.texParameteri(glPyra.TEXTURE_2D, glPyra.TEXTURE_MAG_FILTER, glPyra.LINEAR);
    glPyra.texImage2D(
        glPyra.TEXTURE_2D,
        0,
        glPyra.RGBA,
        glPyra.RGBA,
        glPyra.UNSIGNED_BYTE,
        document.querySelector(".texture-img")
    );

    // Ease GPU workload
    glPyra.bindTexture(glPyra.TEXTURE_2D, null);

    // Use the program
    glPyra.useProgram(program);

    // Get the uniform matrices
    const mWorldPos = glPyra.getUniformLocation(program, "mWorld");
    const mViewPos = glPyra.getUniformLocation(program, "mView");
    const mProjPos = glPyra.getUniformLocation(program, "mProj");

    // Set the value
    let worldMatrix = new Float32Array(16);
    let viewMatrix = new Float32Array(16);
    let projMatrix = new Float32Array(16);

    // Set the matrices
    // Identity matrix for World
    mat4.identity(worldMatrix);
    // lookAt processing, the last elmt in the first vector will be zoom-in-out value
    mat4.lookAt(viewMatrix, [0, 0, -8], [0, 0, 0], [0, 1, 0]);
    // Perspective processing for Projection Matrix
    mat4.perspective(
        projMatrix,
        glMatrix.glMatrix.toRadian(-45),
        pyraCanvas.clientWidth / pyraCanvas.clientHeight,
        0.1,
        1000.0
    );

    // Send the matrices as parameters to the program
    glPyra.uniformMatrix4fv(mWorldPos, false, worldMatrix);
    glPyra.uniformMatrix4fv(mViewPos, false, viewMatrix);
    glPyra.uniformMatrix4fv(mProjPos, false, projMatrix);

    // Rotate around X and Z
    const xyzRotMatrix = new Float32Array(16);
    const yRotMatrix = new Float32Array(16);

    // Render loop
    const identityMatrix = new Float32Array(16);
    mat4.identity(identityMatrix);
    let rotAngle;

    const renderLoop = () => {
        // Rotate on y-axis
        rotAngle = (performance.now() / 1000 / 2) * 2 * Math.PI;
        mat4.rotate(yRotMatrix, identityMatrix, rotAngle, [0, 1, 0]);
        mat4.rotate(xyzRotMatrix, identityMatrix, rotAngle, [1, 1, 1]);
        mat4.mul(worldMatrix, yRotMatrix, xyzRotMatrix);
        glPyra.uniformMatrix4fv(mWorldPos, false, worldMatrix);

        // Bind texture and set the active texture to the model
        glPyra.bindTexture(glPyra.TEXTURE_2D, texture);
        glPyra.activeTexture(glPyra.TEXTURE0);

        // Draw the vertices with pre-defined indices to create the triangle
        // mode: TRIANGLES, POINTS, LINES, TRIANGLES_STRIP, TRIANGLES_FAN
        // count: number of indices drawn
        // type: drawn type, unsigned short
        // offset
        glPyra.drawElements(glPyra.TRIANGLES, indices.length, glPyra.UNSIGNED_SHORT, 0);

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
];

const fragmentShaderText = [
    `precision mediump float;
    varying vec3 fragColor;
    void main(){
    gl_FragColor = vec4(fragColor, 0.8);
    }`,
];

// Texture Mapping Shader
// Use Texture Coordinate instead of Vertex Color
// Use sampler to sample the texture
// Update the Fragment Color to the 2D texture based on Texture Coordinates
const vertShaderTexture = [
    `precision mediump float;
    attribute vec3 vertPosition;
    attribute vec2 vertTexCoord;
    varying vec2 fragTexCoord;
    uniform mat4 mWorld;
    uniform mat4 mView;
    uniform mat4 mProj;
    void main(){
        fragTexCoord = vertTexCoord;
        gl_Position = mProj * mView * mWorld * vec4(vertPosition, 1.0);
    }`,
];

const fragShaderTexture = [
    `precision mediump float;
    varying vec2 fragTexCoord;
    uniform sampler2D sampler;
    void main(){
    gl_FragColor = texture2D(sampler, fragTexCoord);
    }`,
];
