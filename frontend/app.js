import * as THREE from "three";

import {
    OrbitControls
} from "three/addons/controls/OrbitControls.js";


// ============================================================
// DOM
// ============================================================

const reframeButton =
    document.getElementById(
        "reframe-button"
    );

const container =
    document.getElementById(
        "three-container"
    );

const statusText =
    document.getElementById(
        "status"
    );

const dhBody =
    document.getElementById(
        "dh-body"
    );

const parameterList =
    document.getElementById(
        "parameter-list"
    );

const homeList =
    document.getElementById(
        "home-list"
    );

const addRowButton =
    document.getElementById(
        "add-row"
    );

const removeRowButton =
    document.getElementById(
        "remove-row"
    );

const buildButton =
    document.getElementById(
        "build-button"
    );

const robotLibrary =
    document.getElementById(
        "robot-library"
    );

const linearJogButtons =
    document.querySelectorAll(
        ".linear-jog-button"
    );

const linearStepInput =
    document.getElementById(
        "linear-step"
    );

const revoluteStepInput =
    document.getElementById(
        "revolute-step"
    );

const prismaticStepInput =
    document.getElementById(
        "prismatic-step"
    );

const jointJogList =
    document.getElementById(
        "joint-jog-list"
    );

const tcpX =
    document.getElementById(
        "tcp-x"
    );

const tcpY =
    document.getElementById(
        "tcp-y"
    );

const tcpZ =
    document.getElementById(
        "tcp-z"
    );


// ============================================================
// CURRENT ROBOT STATE
// ============================================================

let currentDHTable = [];

let currentRobotValues = {};

let currentJointInfo = [];

let currentFrames = [];

let robotBuilt = false;

let firstFrameDone = false;


// ============================================================
// SMALL UTILITIES
// ============================================================

function sleep(
    milliseconds
) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )
    );
}


// ============================================================
// SAFE API RESPONSE
//
// Render bazen cold start / deploy sırasında JSON yerine:
//
// Not Found
// Bad Gateway
// Service Unavailable
//
// gibi plain-text cevaplar döndürebilir.
//
// response.json() kullanırsak:
// Unexpected token 'N'
//
// hatası oluşur.
//
// Bu helper önce text okur, sonra güvenli şekilde JSON parse eder.
// ============================================================

async function readApiResponse(
    response
) {

    const text =
        await response.text();


    let data = null;


    if (
        text.trim() !== ""
    ) {

        try {

            data =
                JSON.parse(
                    text
                );

        }

        catch {

            throw new Error(

                `Backend geçersiz cevap verdi ` +
                `(HTTP ${response.status}): ${text}`

            );
        }
    }


    if (
        !response.ok
    ) {

        throw new Error(

            data?.detail
            ??
            data?.message
            ??
            `Backend HTTP ${response.status} hatası verdi.`

        );
    }


    return data;
}


// ============================================================
// SAFE API REQUEST
//
// Geçici Render hatalarında birkaç kez tekrar dener.
// ============================================================
// ============================================================
// HOLD-TO-JOG
// ============================================================

let jogHoldTimeout = null;
let jogHoldInterval = null;

function startHoldJog(jogFunction) {

    // Önce tek jog hemen çalışsın
    jogFunction();

    // Kısa basış ile uzun basışı ayır
    jogHoldTimeout = setTimeout(() => {

        // Basılı tutuluyorsa sürekli jog
        jogHoldInterval = setInterval(() => {

            jogFunction();

        }, 80);

    }, 150);
}


function stopHoldJog() {

    if (jogHoldTimeout !== null) {

        clearTimeout(jogHoldTimeout);
        jogHoldTimeout = null;

    }

    if (jogHoldInterval !== null) {

        clearInterval(jogHoldInterval);
        jogHoldInterval = null;

    }
}

async function apiRequest(
    url,
    options = {},
    retries = 2
) {

    let lastError = null;


    for (
        let attempt = 0;
        attempt <= retries;
        attempt++
    ) {

        try {

            const response =
                await fetch(
                    url,
                    {
                        cache: "no-store",
                        ...options
                    }
                );


            // Render deploy / wake-up sırasında
            // geçici olarak bu kodları görebiliriz.

            const retryableStatus =
                response.status === 404
                ||
                response.status === 408
                ||
                response.status === 429
                ||
                response.status === 500
                ||
                response.status === 502
                ||
                response.status === 503
                ||
                response.status === 504;


            if (
                retryableStatus
                &&
                attempt < retries
            ) {

                await sleep(
                    800
                    * (attempt + 1)
                );

                continue;
            }


            return await readApiResponse(
                response
            );

        }

        catch (error) {

            lastError =
                error;


            if (
                attempt >= retries
            ) {

                throw error;
            }


            await sleep(
                800
                * (attempt + 1)
            );
        }
    }


    throw (
        lastError
        ??
        new Error(
            "Backend isteği başarısız."
        )
    );
}


// ============================================================
// WAIT FOR BACKEND
//
// Render Free instance uyuyorsa backend'in ayağa kalkmasını
// bekler.
//
// Böylece:
//
// checkBackend();
// loadPresetLibrary();
// createDemoRobot();
//
// üçlüsünü aynı anda ateşlemiyoruz.
// ============================================================

async function waitForBackend(
    maxAttempts = 45
) {

    disableLinearJog();


    for (
        let attempt = 1;
        attempt <= maxAttempts;
        attempt++
    ) {

        try {

            statusText.textContent =
                attempt === 1
                    ? "Backend bağlantısı bekleniyor..."
                    : `Backend uyanıyor... ${attempt}/${maxAttempts}`;


            const data =
                await apiRequest(
                    "/api/health",
                    {},
                    0
                );


            if (
                data?.status === "ok"
            ) {

                statusText.textContent =
                    "Backend bağlı";

                return true;
            }

        }

        catch (error) {

            console.log(
                `Backend henüz hazır değil (${attempt}/${maxAttempts})`,
                error.message
            );
        }


        await sleep(
            1500
        );
    }


    statusText.textContent =
        "Backend bağlantısı kurulamadı";


    return false;
}


// ============================================================
// THREE.JS SCENE
// ============================================================

const scene =
    new THREE.Scene();


scene.background =
    new THREE.Color(
        0x161616
    );


// ============================================================
// CAMERA - Z UP
// ============================================================

const camera =
    new THREE.PerspectiveCamera(
        45,
        1,
        0.1,
        10000
    );


camera.up.set(
    0,
    0,
    1
);


camera.position.set(
    500,
    -500,
    400
);


// ============================================================
// RENDERER
// ============================================================

const renderer =
    new THREE.WebGLRenderer({
        antialias: true
    });


renderer.setPixelRatio(

    Math.min(
        window.devicePixelRatio,
        2
    )

);


container.appendChild(
    renderer.domElement
);


// ============================================================
// CONTROLS
// ============================================================

const controls =
    new OrbitControls(
        camera,
        renderer.domElement
    );


controls.enableDamping =
    true;


controls.target.set(
    0,
    0,
    100
);


// ============================================================
// LIGHTING
// ============================================================

scene.add(

    new THREE.AmbientLight(
        0xffffff,
        1.5
    )

);


const directionalLight =
    new THREE.DirectionalLight(
        0xffffff,
        2
    );


directionalLight.position.set(
    300,
    -300,
    600
);


scene.add(
    directionalLight
);


// ============================================================
// XY GRID - Z UP
// ============================================================

const grid =
    new THREE.GridHelper(
        1000,
        20
    );


grid.rotation.x =
    Math.PI / 2;


scene.add(
    grid
);


// ============================================================
// WORLD AXES
// ============================================================

const worldAxes =
    new THREE.AxesHelper(
        150
    );


scene.add(
    worldAxes
);


// ============================================================
// ROBOT GROUP
// ============================================================

const robotGroup =
    new THREE.Group();


scene.add(
    robotGroup
);


// ============================================================
// RESIZE
// ============================================================

function resizeRenderer() {

    const width =
        container.clientWidth;


    const height =
        container.clientHeight;


    if (
        width <= 0
        ||
        height <= 0
    ) {

        return;
    }


    renderer.setSize(
        width,
        height,
        false
    );


    camera.aspect =
        width / height;


    camera.updateProjectionMatrix();
}


window.addEventListener(
    "resize",
    resizeRenderer
);


// ============================================================
// ANIMATION LOOP
// ============================================================

function animate() {

    requestAnimationFrame(
        animate
    );


    controls.update();


    renderer.render(
        scene,
        camera
    );
}


resizeRenderer();

animate();


// ============================================================
// ADD DH ROW
// ============================================================

function addDHRow(
    data = {}
) {

    const tr =
        document.createElement(
            "tr"
        );


    tr.dataset.min =
        data.min ?? "";


    tr.dataset.max =
        data.max ?? "";


    const fields = [
        "theta",
        "d",
        "a",
        "alpha"
    ];


    for (
        const field
        of fields
    ) {

        const td =
            document.createElement(
                "td"
            );


        const input =
            document.createElement(
                "input"
            );


        input.value =
            data[field]
            ?? "0";


        input.dataset.field =
            field;


        input.addEventListener(
            "input",
            onDHChanged
        );


        td.appendChild(
            input
        );


        tr.appendChild(
            td
        );
    }


    // ========================================================
    // TYPE
    // ========================================================

    const typeTd =
        document.createElement(
            "td"
        );


    const select =
        document.createElement(
            "select"
        );


    for (
        const type
        of [
            "R",
            "P",
            "FIXED",
            "TOOL"
        ]
    ) {

        const option =
            document.createElement(
                "option"
            );


        option.value =
            type;


        option.textContent =
            type;


        select.appendChild(
            option
        );
    }


    select.value =
        data.type
        ?? "FIXED";


    select.addEventListener(
        "change",
        onDHChanged
    );


    typeTd.appendChild(
        select
    );


    tr.appendChild(
        typeTd
    );


    dhBody.appendChild(
        tr
    );
}


// ============================================================
// READ DH TABLE
// ============================================================

function readDHTable() {

    const rows = [];


    for (
        const tr
        of dhBody.querySelectorAll(
            "tr"
        )
    ) {

        const inputs =
            tr.querySelectorAll(
                "input"
            );


        const select =
            tr.querySelector(
                "select"
            );


        const row = {

            theta:
                inputs[0]?.value
                || "0",

            d:
                inputs[1]?.value
                || "0",

            a:
                inputs[2]?.value
                || "0",

            alpha:
                inputs[3]?.value
                || "0",

            type:
                select?.value
                || "FIXED"

        };


        if (
            tr.dataset.min !== ""
        ) {

            row.min =
                Number(
                    tr.dataset.min
                );
        }


        if (
            tr.dataset.max !== ""
        ) {

            row.max =
                Number(
                    tr.dataset.max
                );
        }


        rows.push(
            row
        );
    }


    return rows;
}


// ============================================================
// READ PARAMETER / HOME INPUTS
// ============================================================

function readCurrentValues() {

    const values = {};


    document
        .querySelectorAll(
            "[data-symbol]"
        )
        .forEach(
            input => {

                const value =
                    Number(
                        input.value
                    );


                values[
                    input.dataset.symbol
                ] =
                    Number.isFinite(
                        value
                    )
                        ? value
                        : 0;

            }
        );


    return values;
}


// ============================================================
// DH CHANGE
// ============================================================

let parameterRefreshTimer =
    null;


function onDHChanged() {

    robotBuilt =
        false;


    disableLinearJog();


    clearTimeout(
        parameterRefreshTimer
    );


    parameterRefreshTimer =
        setTimeout(
            refreshParameters,
            250
        );
}


// ============================================================
// DEFAULT VALUES
// ============================================================

async function refreshParameters() {

    const dhTable =
        readDHTable();


    if (
        dhTable.length === 0
    ) {

        parameterList.textContent =
            "DH tablosu boş.";


        homeList.textContent =
            "DH tablosu boş.";


        return false;
    }


    try {

        const data =
            await apiRequest(
                "/api/default-values",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            dh_table:
                                dhTable,

                            values: {}

                        })

                }
            );


        createParameterInputs(
            data.values
        );


        statusText.textContent =
            "DH hazır.";


        return true;

    }

    catch (error) {

        console.error(
            error
        );


        statusText.textContent =
            "DH hatası";


        parameterList.textContent =
            "Parametreler alınamadı.";


        homeList.textContent =
            "Home değerleri alınamadı.";


        return false;
    }
}


// ============================================================
// CREATE PARAMETER / HOME INPUTS
// ============================================================

function createParameterInputs(
    values
) {

    const previous =
        readCurrentValues();


    parameterList.innerHTML =
        "";


    homeList.innerHTML =
        "";


    const names =
        Object.keys(
            values
            ?? {}
        ).sort(
            naturalSymbolSort
        );


    for (
        const name
        of names
    ) {

        const value =
            previous[name]
            ??
            values[name];


        if (
            /^q\d+$/.test(
                name
            )
        ) {

            createValueRow(
                homeList,
                `${name}(0)`,
                name,
                value,
                ""
            );

        }

        else {

            createValueRow(
                parameterList,
                name,
                name,
                value,
                "mm"
            );
        }
    }


    if (
        names.length === 0
    ) {

        parameterList.textContent =
            "Geometrik parametre yok.";
    }
}


// ============================================================
// VALUE ROW
// ============================================================

function createValueRow(
    parent,
    labelText,
    symbol,
    value,
    unitText
) {

    const row =
        document.createElement(
            "div"
        );


    row.className =
        /^q\d+$/.test(
            symbol
        )
            ? "home-row"
            : "parameter-row";


    const label =
        document.createElement(
            "label"
        );


    label.textContent =
        labelText;


    const input =
        document.createElement(
            "input"
        );


    input.type =
        "number";


    input.value =
        value ?? 0;


    input.dataset.symbol =
        symbol;


    input.addEventListener(
        "input",
        scheduleBuild
    );


    const unit =
        document.createElement(
            "span"
        );


    unit.textContent =
        unitText;


    row.append(
        label,
        input,
        unit
    );


    parent.appendChild(
        row
    );
}


// ============================================================
// NATURAL SYMBOL SORT
// ============================================================

function naturalSymbolSort(
    a,
    b
) {

    return a.localeCompare(
        b,
        undefined,
        {
            numeric: true
        }
    );
}


// ============================================================
// BUILD TIMER
// ============================================================

let buildTimer =
    null;


function scheduleBuild() {

    clearTimeout(
        buildTimer
    );


    buildTimer =
        setTimeout(
            buildRobot,
            250
        );
}


// ============================================================
// BUILD ROBOT
// ============================================================

async function buildRobot() {

    const dhTable =
        readDHTable();


    if (
        dhTable.length === 0
    ) {

        return false;
    }


    const values =
        readCurrentValues();


    try {

        statusText.textContent =
            "Robot hesaplanıyor...";


        const data =
            await apiRequest(
                "/api/build",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            dh_table:
                                dhTable,

                            values:
                                values

                        })

                }
            );


        currentDHTable =
            dhTable;


        currentRobotValues = {
            ...data.values
        };


        drawRobot(
            data.frames
        );


        // ====================================================
        // AUTO REFRAME ONLY ON FIRST BUILD
        // ====================================================

        if (
            !firstFrameDone
        ) {

            reframeViewer(
                data.frames
            );


            firstFrameDone =
                true;
        }


        updateTCP(
            data.tcp.position
        );


        buildJointJogPanel();


        enableLinearJog();


        robotBuilt =
            true;


        statusText.textContent =
            `Robot hazır — ${data.frame_count} frame`;


        return true;

    }

    catch (error) {

        console.error(
            error
        );


        robotBuilt =
            false;


        disableLinearJog();


        statusText.textContent =
            "Build hatası";


        return false;
    }
}


// ============================================================
// TCP DISPLAY
// ============================================================

function updateTCP(
    position
) {

    if (
        !position
        ||
        position.length < 3
    ) {

        return;
    }


    tcpX.textContent =
        Number(
            position[0]
        ).toFixed(2);


    tcpY.textContent =
        Number(
            position[1]
        ).toFixed(2);


    tcpZ.textContent =
        Number(
            position[2]
        ).toFixed(2);
}


// ============================================================
// CLEAR ROBOT
// ============================================================

function clearRobot() {

    while (
        robotGroup.children.length > 0
    ) {

        const child =
            robotGroup.children[0];


        robotGroup.remove(
            child
        );


        disposeObject(
            child
        );
    }
}


// ============================================================
// DISPOSE THREE OBJECT
// ============================================================

function disposeObject(
    object
) {

    object.traverse(
        child => {

            if (
                child.geometry
            ) {

                child.geometry.dispose();
            }


            if (
                child.material
            ) {

                if (
                    Array.isArray(
                        child.material
                    )
                ) {

                    child.material.forEach(
                        material =>
                            material.dispose()
                    );

                }

                else {

                    child.material.dispose();
                }
            }
        }
    );
}


// ============================================================
// CREATE LINK
// ============================================================

function createLink(
    p1,
    p2
) {

    const start =
        new THREE.Vector3(
            ...p1
        );


    const end =
        new THREE.Vector3(
            ...p2
        );


    const direction =
        new THREE.Vector3()
            .subVectors(
                end,
                start
            );


    const length =
        direction.length();


    if (
        length < 0.0001
    ) {

        return null;
    }


    const geometry =
        new THREE.CylinderGeometry(
            12,
            12,
            length,
            20
        );


    const material =
        new THREE.MeshStandardMaterial({

            color:
                0x2f86c7,

            metalness:
                0.2,

            roughness:
                0.55

        });


    const cylinder =
        new THREE.Mesh(
            geometry,
            material
        );


    const midpoint =
        new THREE.Vector3()
            .addVectors(
                start,
                end
            )
            .multiplyScalar(
                0.5
            );


    cylinder.position.copy(
        midpoint
    );


    cylinder.quaternion
        .setFromUnitVectors(

            new THREE.Vector3(
                0,
                1,
                0
            ),

            direction
                .clone()
                .normalize()

        );


    return cylinder;
}


// ============================================================
// FRAME AXES
// ============================================================

function createFrameAxes(
    frame
) {

    const group =
        new THREE.Group();


    group.position.set(
        frame.position[0],
        frame.position[1],
        frame.position[2]
    );


    const R =
        frame.rotation;


    const matrix =
        new THREE.Matrix4();


    matrix.set(

        R[0][0],
        R[0][1],
        R[0][2],
        0,

        R[1][0],
        R[1][1],
        R[1][2],
        0,

        R[2][0],
        R[2][1],
        R[2][2],
        0,

        0,
        0,
        0,
        1

    );


    group.setRotationFromMatrix(
        matrix
    );


    group.add(

        new THREE.AxesHelper(
            45
        )

    );


    return group;
}


// ============================================================
// DRAW ROBOT
// ============================================================

function drawRobot(
    frames
) {

    if (
        !Array.isArray(
            frames
        )
    ) {

        return;
    }


    currentFrames =
        frames;


    clearRobot();


    // ========================================================
    // LINKS
    // ========================================================

    for (
        let i = 0;
        i < frames.length - 1;
        i++
    ) {

        const link =
            createLink(

                frames[i].position,

                frames[i + 1].position

            );


        if (
            link
        ) {

            robotGroup.add(
                link
            );
        }
    }


    // ========================================================
    // JOINTS + FRAMES
    // ========================================================

    for (
        const frame
        of frames
    ) {

        const joint =
            new THREE.Mesh(

                new THREE.SphereGeometry(
                    17,
                    20,
                    20
                ),

                new THREE.MeshStandardMaterial({
                    color:
                        0xd8d8d8
                })

            );


        joint.position.set(
            ...frame.position
        );


        robotGroup.add(
            joint
        );


        robotGroup.add(

            createFrameAxes(
                frame
            )

        );
    }


    // Kamera burada değiştirilmez.
}


// ============================================================
// REFRAME VIEWER
// ============================================================

function reframeViewer(
    frames
) {

    if (
        !frames
        ||
        frames.length === 0
    ) {

        return;
    }


    const points =
        frames.map(

            frame =>
                new THREE.Vector3(
                    ...frame.position
                )

        );


    const box =
        new THREE.Box3()
            .setFromPoints(
                points
            );


    const center =
        box.getCenter(
            new THREE.Vector3()
        );


    const size =
        box.getSize(
            new THREE.Vector3()
        );


    let maxSize =
        Math.max(
            size.x,
            size.y,
            size.z
        );


    if (
        maxSize < 100
    ) {

        maxSize =
            100;
    }


    const distance =
        maxSize * 2.2;


    camera.position.set(

        center.x + distance,

        center.y - distance,

        center.z + distance * 0.7

    );


    camera.up.set(
        0,
        0,
        1
    );


    controls.target.copy(
        center
    );


    controls.update();
}


// ============================================================
// ENABLE / DISABLE LINEAR JOG
// ============================================================

function enableLinearJog() {

    linearJogButtons.forEach(
        button => {

            button.disabled =
                false;

        }
    );
}


function disableLinearJog() {

    linearJogButtons.forEach(
        button => {

            button.disabled =
                true;

        }
    );
}


// ============================================================
// FIND JOINT INFO
// ============================================================

function extractJointInfo() {

    const joints = [];


    for (
        const row
        of currentDHTable
    ) {

        if (
            row.type !== "R"
            &&
            row.type !== "P"
        ) {

            continue;
        }


        const text = [

            row.theta,

            row.d,

            row.a,

            row.alpha

        ].join(" ");


        const match =
            text.match(
                /\bq\d+\b/
            );


        if (
            !match
        ) {

            continue;
        }


        const name =
            match[0];


        let min =
            row.min;


        let max =
            row.max;


        if (
            min === undefined
            ||
            min === null
        ) {

            min =
                row.type === "R"
                    ? -180
                    : -1000;
        }


        if (
            max === undefined
            ||
            max === null
        ) {

            max =
                row.type === "R"
                    ? 180
                    : 1000;
        }


        joints.push({

            name,

            type:
                row.type,

            min,

            max

        });
    }


    joints.sort(

        (a, b) =>

            Number(
                a.name.slice(1)
            )

            -

            Number(
                b.name.slice(1)
            )

    );


    return joints;
}


// ============================================================
// SIMPLE JOINT JOG PANEL
// ============================================================

function buildJointJogPanel() {

    jointJogList.innerHTML =
        "";


    currentJointInfo =
        extractJointInfo();


    if (
        currentJointInfo.length === 0
    ) {

        jointJogList.textContent =
            "Hareketli joint bulunamadı.";


        return;
    }


    for (
        const joint
        of currentJointInfo
    ) {

        const row =
            document.createElement(
                "div"
            );


        row.className =
            "joint-jog-row";


        const currentValue =
            Number(

                currentRobotValues[
                    joint.name
                ]
                ??
                0

            );


        const unit =
            joint.type === "R"
                ? "°"
                : "mm";


        row.innerHTML = `

            <span class="joint-name">
                ${joint.name}
            </span>

            <button
                class="joint-minus"
                data-joint="${joint.name}"
            >
                &lt;
            </button>

            <span
                id="joint-current-${joint.name}"
                class="joint-value"
                data-unit="${unit}"
            >
                ${currentValue.toFixed(2)} ${unit}
            </span>

            <button
                class="joint-plus"
                data-joint="${joint.name}"
            >
                &gt;
            </button>

        `;


        jointJogList.appendChild(
            row
        );
    }


    bindJointJogButtons();
}


// ============================================================
// UPDATE JOINT DISPLAY
//
// Burada önceki kodda .value kullanılıyordu.
//
// Ancak joint-current artık INPUT değil SPAN.
//
// Bu nedenle textContent kullanıyoruz.
// ============================================================

function updateJointJogValues(
    qValues
) {

    for (
        const [name, value]
        of Object.entries(
            qValues
        )
    ) {

        const display =
            document.getElementById(
                `joint-current-${name}`
            );


        if (
            display
        ) {

            const unit =
                display.dataset.unit
                ??
                "";


            display.textContent =
                `${Number(value).toFixed(2)} ${unit}`;

        }
    }
}


// ============================================================
// LINEAR JOG
// ============================================================

async function performLinearJog(
    axis,
    direction
) {

    if (
        !robotBuilt
    ) {

        return;
    }


    const step =
        Math.abs(

            Number(
                linearStepInput.value
            )

        );


    if (
        !Number.isFinite(
            step
        )
        ||
        step <= 0
    ) {

        alert(
            "Geçerli bir Linear Jog step değeri gir."
        );


        return;
    }


    try {

        statusText.textContent =
            `${axis} Jog...`;


        const data =
            await apiRequest(
                "/api/jog/linear",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            dh_table:
                                currentDHTable,

                            values:
                                currentRobotValues,

                            axis:
                                axis,

                            direction:
                                direction,

                            step:
                                step

                        })

                }
            );


        if (
            !data.success
        ) {

            statusText.textContent =
                `Jog başarısız — hata ${Number(
                    data.position_error
                ).toFixed(2)} mm`;


            return;
        }


        for (
            const [name, value]
            of Object.entries(
                data.q
            )
        ) {

            currentRobotValues[
                name
            ] =
                value;
        }


        drawRobot(
            data.frames
        );


        updateTCP(
            data.tcp
        );


        updateJointJogValues(
            data.q
        );


        statusText.textContent =

            `${axis} ` +

            `${data.distance >= 0 ? "+" : ""}` +

            `${Number(data.distance).toFixed(2)} mm`;

    }

    catch (error) {

        console.error(
            error
        );


        statusText.textContent =
            "Linear Jog hatası";


        alert(
            error.message
        );
    }
}


// ============================================================
// JOINT JOG
// ============================================================

async function performJointJog(
    jointName,
    direction
) {

    if (
        !robotBuilt
    ) {

        return;
    }


    const revoluteStep =
        Math.abs(

            Number(
                revoluteStepInput.value
            )

        );


    const prismaticStep =
        Math.abs(

            Number(
                prismaticStepInput.value
            )

        );


    if (
        !Number.isFinite(
            revoluteStep
        )
        ||
        !Number.isFinite(
            prismaticStep
        )
    ) {

        alert(
            "Geçerli Joint Jog step değerleri gir."
        );


        return;
    }


    try {

        statusText.textContent =
            `${jointName} Jog...`;


        const data =
            await apiRequest(
                "/api/jog/joint",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            dh_table:
                                currentDHTable,

                            values:
                                currentRobotValues,

                            joint_name:
                                jointName,

                            direction:
                                direction,

                            revolute_step:
                                revoluteStep,

                            prismatic_step:
                                prismaticStep

                        })

                }
            );


        for (
            const [name, value]
            of Object.entries(
                data.q
            )
        ) {

            currentRobotValues[
                name
            ] =
                value;
        }


        drawRobot(
            data.frames
        );


        updateTCP(
            data.tcp
        );


        updateJointJogValues(
            data.q
        );


        statusText.textContent =
            `${jointName} jog`;

    }

    catch (error) {

        console.error(
            error
        );


        statusText.textContent =
            "Joint Jog hatası";


        alert(
            error.message
        );
    }
}

// ============================================================
// HOLD TO JOG
// ============================================================

let jogHoldActive = false;

let jogHoldToken = 0;


// Basılı tutmaya başla
async function startHoldJog(
    jogFunction
) {

    // Başka bir hold devam ediyorsa iptal et
    stopHoldJog();


    jogHoldActive =
        true;


    // Her yeni hold için yeni token
    const myToken =
        ++jogHoldToken;


    // ========================================================
    // İLK JOG
    //
    // Butona basıldığı anda hareket etsin.
    // ========================================================

    await jogFunction();


    // İlk tek tıklamayla uzun basışı ayırmak için
    // ufak bir bekleme
    await sleep(
        120
    );


    // ========================================================
    // CONTINUOUS JOG
    // ========================================================

    while (
        jogHoldActive
        &&
        myToken === jogHoldToken
    ) {

        /*
            KRİTİK:

            await kullandığımız için önceki API request
            bitmeden yeni request gönderilmiyor.

            Bu yüzden backend request kuyruğu oluşmaz.
        */

        await jogFunction();


        // Çok agresif request göndermeyelim
        await sleep(
            40
        );
    }
}


// Basılı tutmayı bırak
function stopHoldJog() {

    jogHoldActive =
        false;


    // Eski loop'un geçersiz kalmasını sağlar
    jogHoldToken++;
}

// ============================================================
// JOINT BUTTON BIND
// ============================================================


// ============================================================
// JOINT BUTTON BIND
// ============================================================

function bindJointJogButtons() {

    // ========================================================
    // JOINT -
    // ========================================================

    document
        .querySelectorAll(
            ".joint-minus"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "pointerdown",
                    event => {

                        event.preventDefault();


                        button.setPointerCapture(
                            event.pointerId
                        );


                        startHoldJog(
                            () =>
                                performJointJog(

                                    button.dataset.joint,

                                    -1

                                )
                        );

                    }
                );


                button.addEventListener(
                    "pointerup",
                    stopHoldJog
                );


                button.addEventListener(
                    "pointercancel",
                    stopHoldJog
                );


                button.addEventListener(
                    "lostpointercapture",
                    stopHoldJog
                );

            }
        );


    // ========================================================
    // JOINT +
    // ========================================================

    document
        .querySelectorAll(
            ".joint-plus"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "pointerdown",
                    event => {

                        event.preventDefault();


                        button.setPointerCapture(
                            event.pointerId
                        );


                        startHoldJog(
                            () =>
                                performJointJog(

                                    button.dataset.joint,

                                    1

                                )
                        );

                    }
                );


                button.addEventListener(
                    "pointerup",
                    stopHoldJog
                );


                button.addEventListener(
                    "pointercancel",
                    stopHoldJog
                );


                button.addEventListener(
                    "lostpointercapture",
                    stopHoldJog
                );

            }
        );
}

// ============================================================
// LINEAR BUTTON EVENTS
// ============================================================

linearJogButtons.forEach(

    button => {

        button.addEventListener(
            "click",
            () => {

                performLinearJog(

                    button.dataset.axis,

                    Number(
                        button.dataset.direction
                    )

                );
            }
        );
    }

);


// ============================================================
// PRESET LIBRARY
// ============================================================

async function loadPresetLibrary() {

    try {

        robotLibrary.textContent =
            "Presetler yükleniyor...";


        const data =
            await apiRequest(
                "/api/presets",
                {},
                3
            );


        robotLibrary.innerHTML =
            "";


        if (
            !data?.presets
            ||
            data.presets.length === 0
        ) {

            robotLibrary.textContent =
                "Preset bulunamadı.";


            return false;
        }


        for (
            const preset
            of data.presets
        ) {

            const button =
                document.createElement(
                    "button"
                );


            button.className =
                "preset-button";


            button.textContent =
                preset.dof

                    ? `${preset.name} (${preset.dof} DOF)`

                    : preset.name;


            button.addEventListener(
                "click",
                () => {

                    loadPreset(
                        preset.id
                    );

                }
            );


            robotLibrary.appendChild(
                button
            );
        }


        return true;

    }

    catch (error) {

        console.error(
            error
        );


        robotLibrary.textContent =
            "Presetler yüklenemedi.";


        return false;
    }
}


// ============================================================
// LOAD PRESET
// ============================================================

async function loadPreset(
    presetId
) {

    try {

        statusText.textContent =
            "Preset yükleniyor...";


        const preset =
            await apiRequest(
                `/api/presets/${encodeURIComponent(presetId)}`,
                {},
                2
            );


        if (
            !preset?.dh_table
        ) {

            throw new Error(
                "Preset DH tablosu bulunamadı."
            );
        }


        // ====================================================
        // DH
        // ====================================================

        dhBody.innerHTML =
            "";


        for (
            const row
            of preset.dh_table
        ) {

            addDHRow(
                row
            );
        }


        // ====================================================
        // PARAMETER / HOME INPUTS
        // ====================================================

        const refreshed =
            await refreshParameters();


        if (
            !refreshed
        ) {

            throw new Error(
                "Preset parametreleri oluşturulamadı."
            );
        }


        // ====================================================
        // PRESET PARAMETERS
        // ====================================================

        if (
            preset.parameters
        ) {

            for (
                const [name, value]
                of Object.entries(
                    preset.parameters
                )
            ) {

                const input =
                    document.querySelector(
                        `[data-symbol="${name}"]`
                    );


                if (
                    input
                ) {

                    input.value =
                        value;
                }
            }
        }


        // ====================================================
        // HOME
        // ====================================================

        if (
            preset.home
        ) {

            for (
                const [name, value]
                of Object.entries(
                    preset.home
                )
            ) {

                const input =
                    document.querySelector(
                        `[data-symbol="${name}"]`
                    );


                if (
                    input
                ) {

                    input.value =
                        value;
                }
            }
        }


        // ====================================================
        // BUILD ONCE
        // ====================================================

        const built =
            await buildRobot();


        if (
            built
        ) {

            statusText.textContent =
                `${preset.name} yüklendi`;
        }

    }

    catch (error) {

        console.error(
            error
        );


        statusText.textContent =
            "Preset yükleme hatası";


        alert(
            error.message
        );
    }
}


// ============================================================
// ADD ROW
// ============================================================

addRowButton.addEventListener(
    "click",
    async () => {

        addDHRow();


        await refreshParameters();

    }
);


// ============================================================
// REMOVE ROW
// ============================================================

removeRowButton.addEventListener(
    "click",
    async () => {

        const rows =
            dhBody.querySelectorAll(
                "tr"
            );


        if (
            rows.length > 0
        ) {

            rows[
                rows.length - 1
            ].remove();


            await refreshParameters();
        }

    }
);


// ============================================================
// BUILD BUTTON
// ============================================================

buildButton.addEventListener(
    "click",
    buildRobot
);


// ============================================================
// REFRAME BUTTON
// ============================================================

reframeButton.addEventListener(
    "click",
    () => {

        reframeViewer(
            currentFrames
        );

    }
);


// ============================================================
// DEMO ROBOT
// ============================================================

async function createDemoRobot() {

    if (
        dhBody.children.length > 0
    ) {

        return;
    }


    addDHRow({

        theta:
            "q1",

        d:
            "0",

        a:
            "L1",

        alpha:
            "0",

        type:
            "R",

        min:
            -180,

        max:
            180

    });


    addDHRow({

        theta:
            "q2",

        d:
            "0",

        a:
            "L2",

        alpha:
            "0",

        type:
            "R",

        min:
            -180,

        max:
            180

    });


    addDHRow({

        theta:
            "q3",

        d:
            "0",

        a:
            "L3",

        alpha:
            "0",

        type:
            "R",

        min:
            -180,

        max:
            180

    });


    await refreshParameters();
}


// ============================================================
// START APPLICATION
// ============================================================

async function startApplication() {

    disableLinearJog();


    robotLibrary.textContent =
        "Backend bekleniyor...";


    // ========================================================
    // 1) BACKEND READY
    // ========================================================

    const backendReady =
        await waitForBackend();


    if (
        !backendReady
    ) {

        robotLibrary.textContent =
            "Backend bağlantısı kurulamadı.";


        return;
    }


    // ========================================================
    // 2) LOAD PRESETS
    // ========================================================

    await loadPresetLibrary();


    // ========================================================
    // 3) CREATE DEFAULT DEMO
    // ========================================================

    await createDemoRobot();


    statusText.textContent =
        "Backend bağlı";
}


// ============================================================
// START
// ============================================================

startApplication();