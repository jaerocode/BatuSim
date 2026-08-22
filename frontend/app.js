import * as THREE from "three";

import {
    OrbitControls
} from "three/addons/controls/OrbitControls.js";


// ============================================================
// DOM
// ============================================================

// ============================================================
// AI DIRECTOR DOM
// ============================================================

const aiDirectorInput =
    document.getElementById(
        "ai-director-input"
    );


const aiGenerateButton =
    document.getElementById(
        "ai-generate-button"
    );


const aiRunButton =
    document.getElementById(
        "ai-run-button"
    );


const aiDirectorStatus =
    document.getElementById(
        "ai-director-status"
    );

const directorResetButton =
    document.getElementById(
        "director-reset-button"
    );

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
// DIRECTOR DOM
// ============================================================

const builderTabButton =
    document.getElementById(
        "builder-tab-button"
    );

const directorTabButton =
    document.getElementById(
        "director-tab-button"
    );

const builderMode =
    document.getElementById(
        "builder-mode"
    );

const directorMode =
    document.getElementById(
        "director-mode"
    );


const directorProgramList =
    document.getElementById(
        "director-program-list"
    );

const directorProgramPlaceholder =
    document.getElementById(
        "director-program-placeholder"
    );

const directorCommandButtons =
    document.querySelectorAll(
        ".director-command-button"
    );


const directorClearButton =
    document.getElementById(
        "director-clear-button"
    );

const directorRunButton =
    document.getElementById(
        "director-run-button"
    );

const directorStopButton =
    document.getElementById(
        "director-stop-button"
    );


const directorSpeedInput =
    document.getElementById(
        "director-speed"
    );

const directorSpeedValue =
    document.getElementById(
        "director-speed-value"
    );


const directorStatus =
    document.getElementById(
        "director-status"
    );

const directorProgress =
    document.getElementById(
        "director-progress"
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
// DIRECTOR STATE
// ============================================================

let directorProgram = [];

let directorTrajectory = [];

let directorStartValues =
    null;

let directorRunning = false;

let directorAnimationId = null;

let directorCurrentIndex = 0;

let directorLastTimestamp = null;

let directorAccumulator = 0;


// 100% hızda saniyede kaç trajectory point oynatılacak.
const DIRECTOR_BASE_POINTS_PER_SECOND = 30;


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
// ============================================================

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
                        cache:
                            "no-store",

                        ...options
                    }
                );


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
                    *
                    (attempt + 1)
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
                *
                (attempt + 1)
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

                `Backend henüz hazır değil ` +
                `(${attempt}/${maxAttempts})`,

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
        antialias:
            true
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
// DIRECTOR TRAJECTORY GROUP
//
// Robot her redraw olduğunda robotGroup temizleniyor.
//
// Trajectory ise ayrı group'ta tutulduğu için
// drawRobot() trajectory'yi silmez.
// ============================================================

const trajectoryGroup =
    new THREE.Group();


scene.add(
    trajectoryGroup
);


let directorTrajectoryLine =
    null;




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
        data.min
        ??
        "";


    tr.dataset.max =
        data.max
        ??
        "";


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
            ??
            "0";


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
        ??
        "FIXED";


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
                ||
                "0",

            d:
                inputs[1]?.value
                ||
                "0",

            a:
                inputs[2]?.value
                ||
                "0",

            alpha:
                inputs[3]?.value
                ||
                "0",

            type:
                select?.value
                ||
                "FIXED"

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


    stopHoldJog();


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
            ??
            {}
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
        value
        ??
        0;


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
            numeric:
                true
        }
    );
}


// ============================================================
// BUILD TIMER
// ============================================================

let buildTimer =
    null;


function scheduleBuild() {

    stopHoldJog();


    clearTimeout(
        buildTimer
    );


    statusText.textContent =
        "Değişiklik bekleniyor...";


    // Kullanıcının L1/L2 vb. değerini yazmayı
    // bitirmesini biraz bekle.
    buildTimer =
        setTimeout(
            buildRobot,
            800
        );
}




// ============================================================
// BUILD ROBOT
// ============================================================

async function buildRobot() {

    stopHoldJog();


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


// ============================================================
// JOINT INFO MUST BE READY BEFORE DRAW
// ============================================================

currentJointInfo =
    extractJointInfo();


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
// CLEAR DIRECTOR TRAJECTORY
// ============================================================

function clearDirectorTrajectory() {

    while (
        trajectoryGroup.children.length > 0
    ) {

        const child =
            trajectoryGroup.children[0];


        trajectoryGroup.remove(
            child
        );


        disposeObject(
            child
        );
    }


    directorTrajectoryLine =
        null;
}


// ============================================================
// CREATE DIRECTOR TRAJECTORY
//
// Backend'in tcp_path:
//
// [
//     [x, y, z],
//     [x, y, z],
//     ...
// ]
//
// tek seferde geometry'ye çevrilir.
//
// Animasyon sırasında setDrawRange() kullanılarak
// robot gittikçe çizgi uzatılır.
// ============================================================

// ============================================================
// CREATE DIRECTOR TRAJECTORY
// ============================================================

// ============================================================
// RESET ROBOT TO HOME POSITION
// ============================================================

async function resetDirectorToHome() {

    stopHoldJog();


    stopDirectorAnimation(
        false
    );


    clearDirectorTrajectory();


    if (
        !robotBuilt
    ) {

        setDirectorStatus(
            "Önce robot oluştur.",
            "error"
        );


        return;
    }


    try {

        setDirectorStatus(
            "Resetting to Home...",
            "running"
        );


        statusText.textContent =
            "Home Position yükleniyor...";


        // ====================================================
        // HOME + PARAMETERS
        //
        // Sağ paneldeki inputlardan okuyoruz.
        //
        // Bunların içinde:
        // L0 L1 L2...
        // q1 q2 q3...
        //
        // hepsi var.
        // ====================================================

        const homeValues =
            readCurrentValues();


        // ====================================================
        // BUILD HOME CONFIGURATION
        // ====================================================

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
                                currentDHTable,

                            values:
                                homeValues

                        })

                }
            );


        // ====================================================
        // CURRENT STATE = HOME
        // ====================================================

        currentRobotValues = {
            ...data.values
        };


        // ====================================================
        // UPDATE VISUAL
        // ====================================================

        drawRobot(
            data.frames
        );


        // ====================================================
        // TCP
        // ====================================================

        updateTCP(
            data.tcp.position
        );


        // ====================================================
        // JOINT DISPLAY
        // ====================================================

        updateJointJogValues(
            currentRobotValues
        );


        // ====================================================
        // DIRECTOR STATE
        // ====================================================

        directorTrajectory =
            [];


        directorCurrentIndex =
            0;


        directorProgress.textContent =
            "0 / 0";


        setDirectorStatus(
            "Home Position",
            "success"
        );


        statusText.textContent =
            "Robot Home Position'a döndü";

    }

    catch (error) {

        console.error(
            "Home reset error:",
            error
        );


        setDirectorStatus(
            "Reset Error",
            "error"
        );


        statusText.textContent =
            "Home reset hatası";
    }
}

function createDirectorTrajectory(
    tcpPath
) {

    clearDirectorTrajectory();


    if (
        !Array.isArray(tcpPath)
        ||
        tcpPath.length < 2
    ) {

        console.warn(
            "Trajectory çizilemedi. TCP path:",
            tcpPath
        );

        return;
    }


    const points =
        tcpPath.map(

            position =>
                new THREE.Vector3(

                    Number(position[0]),
                    Number(position[1]),
                    Number(position[2])

                )

        );


    console.log(
        "Trajectory points:",
        points
    );


    const geometry =
        new THREE.BufferGeometry()
            .setFromPoints(
                points
            );


    const material =
        new THREE.LineBasicMaterial({

            color: 0x00ff00,   // yeşil,

            depthTest: false,

            depthWrite: false,

            linewidth: 4

        });


    directorTrajectoryLine =
        new THREE.Line(
            geometry,
            material
        );


    directorTrajectoryLine.renderOrder =
        999;


    directorTrajectoryLine.frustumCulled =
        false;


    geometry.setDrawRange(
        0,
        Math.min(
            2,
            points.length
        )
    );


    trajectoryGroup.add(
        directorTrajectoryLine
    );


    console.log(
        "Trajectory line created:",
        directorTrajectoryLine
    );
}
// ============================================================
// RIGHT PANEL MODE
// ============================================================

function setRightPanelMode(
    mode
) {

    const directorSelected =
        mode === "director";


    builderMode.classList.toggle(

        "active",

        !directorSelected

    );


    directorMode.classList.toggle(

        "active",

        directorSelected

    );


    builderTabButton.classList.toggle(

        "active",

        !directorSelected

    );


    directorTabButton.classList.toggle(

        "active",

        directorSelected

    );


    // Director'a geçerken robot yoksa kullanıcıya belirt.
    if (
        directorSelected
        &&
        !robotBuilt
    ) {

        setDirectorStatus(

            "Önce bir robot oluştur.",

            "error"

        );
    }
}

// ============================================================
// DIRECTOR STATUS
// ============================================================

function setDirectorStatus(
    text,
    state = "ready"
) {

    directorStatus.textContent =
        text;


    directorStatus.className =
        `director-status ${state}`;
}

// ============================================================
// ADD DIRECTOR COMMAND
// ============================================================

function addDirectorCommand(
    commandData
) {

    directorProgram.push({

        type:
            commandData.type,

        axis:
            commandData.axis,

        label:
            commandData.label,

        unit:
            commandData.unit,

        value:
            0

    });


    renderDirectorProgram();


    setDirectorStatus(
        "Program edited",
        "ready"
    );
}


// ============================================================
// REMOVE DIRECTOR COMMAND
// ============================================================

function removeDirectorCommand(
    index
) {

    directorProgram.splice(
        index,
        1
    );


    renderDirectorProgram();
}

function setAIDirectorStatus(
    text,
    state = ""
) {

    aiDirectorStatus.textContent =
        text;


    aiDirectorStatus.className =
        `ai-director-status ${state}`;
}

async function interpretAICommand() {

    const text =
        aiDirectorInput.value.trim();


    if (
        !text
    ) {

        setAIDirectorStatus(
            "Bir komut yaz.",
            "error"
        );

        return;
    }


    try {

        setAIDirectorStatus(
            "Komut yorumlanıyor...",
            "running"
        );


        const data =
            await apiRequest(
                "/api/tasks/interpret",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            text:
                                text

                        })

                }
            );


        console.log(
            "AI Task IR:",
            data.task_ir
        );


        const pretty =
            JSON.stringify(
                data.task_ir,
                null,
                2
            );


        setAIDirectorStatus(
            `Understood: ${pretty}`,
            "success"
        );

    }

    catch (error) {

        console.error(
            error
        );


        setAIDirectorStatus(
            error.message,
            "error"
        );
    }
}

async function runAITask() {

    if (
        directorRunning
    ) {

        return;
    }


    if (
        !robotBuilt
    ) {

        setAIDirectorStatus(
            "Önce robot oluştur.",
            "error"
        );

        return;
    }


    const text =
        aiDirectorInput.value.trim();


    if (
        !text
    ) {

        setAIDirectorStatus(
            "Bir komut yaz.",
            "error"
        );

        return;
    }


    stopHoldJog();


    clearDirectorTrajectory();


    setDirectorRunningState(
        true
    );


    setAIDirectorStatus(
        "Robot programı oluşturuluyor...",
        "running"
    );


    directorProgress.textContent =
        "AI Planning...";


    statusText.textContent =
        "AI Director trajectory hesaplıyor...";


    try {

        const data =
            await apiRequest(
                "/api/tasks/plan",
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

                            text:
                                text,

                            linear_step_mm:
                                5,

                            rotation_step_deg:
                                2,

                            revolute_step_deg:
                                2,

                            prismatic_step_mm:
                                5

                        })

                },
                1
            );


        if (
            !data?.success
        ) {

            setDirectorRunningState(
                false
            );


            showDirectorError(
                data
            );


            setAIDirectorStatus(
                data.message
                ??
                "Robot programı oluşturulamadı.",
                "error"
            );


            return;
        }


        if (
            !Array.isArray(
                data.trajectory
            )
            ||
            data.trajectory.length === 0
        ) {

            throw new Error(
                "AI Director trajectory üretmedi."
            );
        }


        console.log(
            "Task IR:",
            data.task_ir
        );


        console.log(
            "Generated program:",
            data.generated_program
        );


        // ====================================================
        // TCP PATH
        // ====================================================

        const tcpPath =
            data.trajectory
                .map(
                    point =>
                        point.tcp
                )
                .filter(

                    tcp =>
                        Array.isArray(
                            tcp
                        )
                        &&
                        tcp.length >= 3

                );


        if (
            tcpPath.length >= 2
        ) {

            createDirectorTrajectory(
                tcpPath
            );
        }


        setAIDirectorStatus(
            "Program understood and validated.",
            "success"
        );


        // ====================================================
        // ANIMATION
        // ====================================================

        startDirectorAnimation(
            data.trajectory
        );

    }

    catch (error) {

        console.error(
            "AI Director error:",
            error
        );


        setDirectorRunningState(
            false
        );


        setAIDirectorStatus(
            error.message,
            "error"
        );


        directorProgress.textContent =
            "Error";


        statusText.textContent =
            "AI Director hatası";
    }
}

// ============================================================
// CLEAR DIRECTOR PROGRAM
// ============================================================

function clearDirectorProgram() {

    if (
        directorRunning
    ) {

        return;
    }


    directorProgram = [];


    renderDirectorProgram();


    clearDirectorTrajectory();


    directorProgress.textContent =
        "0 / 0";


    setDirectorStatus(
        "Ready",
        "ready"
    );
}


// ============================================================
// RENDER DIRECTOR PROGRAM
// ============================================================

function renderDirectorProgram() {

    directorProgramList.innerHTML =
        "";


    if (
        directorProgram.length === 0
    ) {

        const placeholder =
            document.createElement(
                "div"
            );


        placeholder.id =
            "director-program-placeholder";


        placeholder.textContent =
            "Program boş. Aşağıdaki komutlardan birini seç.";


        directorProgramList.appendChild(
            placeholder
        );


        return;
    }


    directorProgram.forEach(

        (
            command,
            index
        ) => {

            const row =
                document.createElement(
                    "div"
                );


            row.className =
                "director-program-row";


            // =================================================
            // LINE NUMBER
            // =================================================

            const lineNumber =
                document.createElement(
                    "div"
                );


            lineNumber.className =
                "director-program-number";


            lineNumber.textContent =
                index + 1;


            // =================================================
            // LABEL
            // =================================================

            const label =
                document.createElement(
                    "div"
                );


            label.className =
                "director-program-label";


            label.textContent =
                command.label;


            // =================================================
            // VALUE
            // =================================================

            const input =
                document.createElement(
                    "input"
                );


            input.className =
                "director-program-value";


            input.type =
                "number";


            input.step =
                command.unit === "deg"
                    ? "1"
                    : "5";


            input.value =
                command.value;


            input.addEventListener(
                "input",
                () => {

                    const value =
                        Number(
                            input.value
                        );


                    command.value =
                        Number.isFinite(
                            value
                        )
                            ? value
                            : 0;


                    clearDirectorTrajectory();


                    setDirectorStatus(
                        "Program edited",
                        "ready"
                    );

                }
            );


            // =================================================
            // UNIT
            // =================================================

            const unit =
                document.createElement(
                    "div"
                );


            unit.className =
                "director-program-unit";


            unit.textContent =
                command.unit;


            // =================================================
            // REMOVE
            // =================================================

            const removeButton =
                document.createElement(
                    "button"
                );


            removeButton.className =
                "director-program-remove";


            removeButton.type =
                "button";


            removeButton.textContent =
                "×";


            removeButton.addEventListener(
                "click",
                () => {

                    removeDirectorCommand(
                        index
                    );

                }
            );


            row.append(

                lineNumber,

                label,

                input,

                unit,

                removeButton

            );


            directorProgramList.appendChild(
                row
            );

        }

    );
}

// ============================================================
// BUILD DIRECTOR COMMAND PAYLOAD
// ============================================================

function buildDirectorCommands() {

    return directorProgram.map(

        command => ({

            type:
                command.type,

            axis:
                command.axis,

            value:
                Number(
                    command.value
                )

        })

    );
}

// ============================================================
// LOCK DIRECTOR / JOG CONTROLS
// ============================================================

function setDirectorRunningState(
    running
) {

    directorRunning =
        running;


    directorRunButton.disabled =
        running;


    directorStopButton.disabled =
        !running;


    directorClearButton.disabled =
        running;


    builderTabButton.disabled =
        running;


    directorTabButton.disabled =
        running;


    directorCommandButtons.forEach(
        button => {

            button.disabled =
                running;

        }
    );


    document
        .querySelectorAll(
            ".director-program-value, .director-program-remove"
        )
        .forEach(
            element => {

                element.disabled =
                    running;

            }
        );


    linearJogButtons.forEach(
        button => {

            button.disabled =
                running
                ||
                !robotBuilt;

        }
    );


    document
        .querySelectorAll(
            ".joint-minus, .joint-plus"
        )
        .forEach(
            button => {

                button.disabled =
                    running;

            }
        );
}

// ============================================================
// SYNC DIRECTOR POINT TO PHYSICAL ROBOT
// ============================================================

async function sendDirectorPointToHardware(
    point
) {

    if (
        !point
        ||
        !Array.isArray(point.q_vector)
        ||
        point.q_vector.length !== 3
    ) {

        console.warn(
            "Hardware için geçersiz q_vector:",
            point
        );

        return false;
    }


    try {

        const data =
            await apiRequest(
                "/api/hardware/q",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            q_vector:
                                point.q_vector

                        })

                },

                0
            );


        console.log(
            "Physical robot:",
            data.q,
            "servo:",
            data.servo_angles
        );


        return true;

    }

    catch (error) {

        console.error(
            "Physical robot sync error:",
            error
        );


        return false;
    }
}

// ============================================================
// APPLY DIRECTOR TRAJECTORY POINT
// ============================================================

function applyDirectorTrajectoryPoint(
    point
) {

    if (
        !point
    ) {

        return;
    }


    // ========================================================
    // ROBOT VALUES
    // ========================================================

    if (
        point.q
    ) {

        for (
            const [name, value]
            of Object.entries(
                point.q
            )
        ) {

            currentRobotValues[
                name
            ] =
                Number(
                    value
                );
        }
    }


    // ========================================================
    // ROBOT GEOMETRY
    // ========================================================

    if (
        point.frames
    ) {

        drawRobot(
            point.frames
        );
    }


    // ========================================================
    // TCP
    // ========================================================

    if (
        point.tcp
    ) {

        updateTCP(
            point.tcp
        );
    }


    // ========================================================
    // JOINT VALUES
    // ========================================================

    if (
        point.q
    ) {

        updateJointJogValues(
            point.q
        );
    }
}

// ============================================================
// START DIRECTOR ANIMATION
// ============================================================

function startDirectorAnimation(
    trajectory
) {

    if (
        !Array.isArray(
            trajectory
        )
        ||
        trajectory.length === 0
    ) {

        setDirectorStatus(
            "Trajectory boş.",
            "error"
        );


        return;
    }


    directorTrajectory =
        trajectory;


    directorCurrentIndex =
        0;


    directorLastTimestamp =
        null;


    directorAccumulator =
        0;


    setDirectorRunningState(
        true
    );


    setDirectorStatus(
        "Running...",
        "running"
    );


    // İlk frame simülasyonda göster.
    applyDirectorTrajectoryPoint(
        directorTrajectory[0]
    );


    // İlk frame fiziksel robota da gönder.
    sendDirectorPointToHardware(
        directorTrajectory[0]
    );


    directorCurrentIndex =
        1;


    updateDirectorTrajectoryTrace(
        1
    );


    directorProgress.textContent =
        `1 / ${directorTrajectory.length}`;


    directorAnimationId =
        requestAnimationFrame(
            directorAnimationLoop
        );
}


// ============================================================
// DIRECTOR ANIMATION LOOP
// ============================================================

function directorAnimationLoop(
    timestamp
) {

    if (
        !directorRunning
    ) {

        return;
    }


    if (
        directorLastTimestamp === null
    ) {

        directorLastTimestamp =
            timestamp;
    }


    const deltaSeconds =

        (
            timestamp
            -
            directorLastTimestamp
        )

        /
        1000;


    directorLastTimestamp =
        timestamp;


    const speedPercent =
        Number(
            directorSpeedInput.value
        );


    const speedFactor =

        Math.max(
            0.1,
            speedPercent / 100
        );


    const pointsPerSecond =

        DIRECTOR_BASE_POINTS_PER_SECOND

        *
        speedFactor;


    directorAccumulator +=

        deltaSeconds

        *
        pointsPerSecond;


    // ========================================================
    // ADVANCE TRAJECTORY
    // ========================================================

    while (
        directorAccumulator >= 1
        &&
        directorCurrentIndex
        <
        directorTrajectory.length
    ) {

        const point =
            directorTrajectory[
                directorCurrentIndex
            ];


        // ========================================================
        // DIGITAL TWIN
        //
        // Aynı trajectory point:
        //
        //      Simulation
        //          +
        //      Physical Robot
        //
        // tarafından kullanılır.
        // ========================================================

        applyDirectorTrajectoryPoint(
            point
        );


        // Fiziksel robota aynı q gönder.
        sendDirectorPointToHardware(
            point
        );


        directorCurrentIndex++;


        directorAccumulator -=
            1;


        updateDirectorTrajectoryTrace(
            directorCurrentIndex
        );


        directorProgress.textContent =

            `${directorCurrentIndex} / ` +
            `${directorTrajectory.length}`;

    }


    // ========================================================
    // FINISHED
    // ========================================================

    if (
        directorCurrentIndex
        >=
        directorTrajectory.length
    ) {

        finishDirectorAnimation();


        return;
    }


    directorAnimationId =
        requestAnimationFrame(
            directorAnimationLoop
        );
}


// ============================================================
// FINISH DIRECTOR ANIMATION
// ============================================================

function finishDirectorAnimation() {

    if (
        directorAnimationId !== null
    ) {

        cancelAnimationFrame(
            directorAnimationId
        );


        directorAnimationId =
            null;
    }


    setDirectorRunningState(
        false
    );


    updateDirectorTrajectoryTrace(
        directorTrajectory.length
    );


    directorProgress.textContent =

        `${directorTrajectory.length} / ` +
        `${directorTrajectory.length}`;


    setDirectorStatus(
        "Program completed",
        "success"
    );


    statusText.textContent =
        "Director program tamamlandı";
}

// ============================================================
// STOP DIRECTOR ANIMATION
// ============================================================

function stopDirectorAnimation(
    showStatus = true
) {

    if (
        directorAnimationId !== null
    ) {

        cancelAnimationFrame(
            directorAnimationId
        );


        directorAnimationId =
            null;
    }


    const wasRunning =
        directorRunning;


    setDirectorRunningState(
        false
    );


    directorLastTimestamp =
        null;


    directorAccumulator =
        0;


    if (
        showStatus
        &&
        wasRunning
    ) {

        setDirectorStatus(
            "Program stopped",
            "ready"
        );


        statusText.textContent =
            "Director program durduruldu";
    }
}

// ============================================================
// RESET DIRECTOR
// ============================================================

async function resetDirectorProgram() {

    // Çalışan animasyon varsa durdur.
    stopDirectorAnimation(
        false
    );


    clearDirectorTrajectory();


    if (
        !directorStartValues
    ) {

        setDirectorStatus(
            "Reset pozisyonu bulunamadı.",
            "error"
        );

        return;
    }


    try {

        setDirectorStatus(
            "Resetting...",
            "running"
        );


        statusText.textContent =
            "Robot başlangıç pozisyonuna dönüyor...";


        // Başlangıç joint değerlerini geri yükle.
        currentRobotValues = {
            ...directorStartValues
        };


        // FK'yı yeniden hesaplat.
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
                                currentDHTable,

                            values:
                                currentRobotValues

                        })

                }
            );


        currentRobotValues = {
            ...data.values
        };


        drawRobot(
            data.frames
        );


        updateTCP(
            data.tcp.position
        );


        updateJointJogValues(
            currentRobotValues
        );


        directorTrajectory = [];

        directorCurrentIndex = 0;

        directorProgress.textContent =
            "0 / 0";


        setDirectorStatus(
            "Reset complete",
            "success"
        );


        statusText.textContent =
            "Robot başlangıç pozisyonuna döndü";

    }

    catch (error) {

        console.error(
            "Director reset error:",
            error
        );


        setDirectorStatus(
            "Reset Error",
            "error"
        );


        statusText.textContent =
            "Reset hatası";
    }
}

// ============================================================
// DIRECTOR ERROR DISPLAY
// ============================================================

function showDirectorError(
    data
) {

    const errorType =
        data?.error_type
        ??
        "DIRECTOR_ERROR";


    let title =
        "Director Error";


    if (
        errorType ===
        "JOINT_LIMIT_ERROR"
    ) {

        title =
            "Joint Limit Error";
    }


    else if (
        errorType ===
        "REACH_ERROR"
    ) {

        title =
            "Reach Error";
    }


    else if (
        errorType ===
        "SINGULARITY_ERROR"
    ) {

        title =
            "Singularity Error";
    }


    const commandNumber =

        Number.isFinite(
            Number(
                data?.command_index
            )
        )

            ? Number(
                data.command_index
            ) + 1

            : null;


    let message =
        title;


    if (
        commandNumber !== null
        &&
        commandNumber > 0
    ) {

        message +=
            ` — Line ${commandNumber}`;
    }


    if (
        data?.message
    ) {

        message +=
            `: ${data.message}`;
    }


    setDirectorStatus(
        message,
        "error"
    );


    statusText.textContent =
        title;
}

// ============================================================
// RUN DIRECTOR PROGRAM
// ============================================================

async function runDirectorProgram() {

    if (
        directorRunning
    ) {
        return;
    }


    // ========================================================
    // ROBOT CHECK
    // ========================================================

    if (
        !robotBuilt
    ) {

        setDirectorStatus(
            "Önce robot oluştur.",
            "error"
        );

        return;
    }


    // ========================================================
    // PROGRAM CHECK
    // ========================================================

    if (
        directorProgram.length === 0
    ) {

        setDirectorStatus(
            "Program boş.",
            "error"
        );

        return;
    }


    // ========================================================
    // PREPARE
    // ========================================================

stopHoldJog();


// Program başlamadan önce robotun
// başlangıç joint konumunu kaydet.
directorStartValues = {
    ...currentRobotValues
};


clearDirectorTrajectory();


clearDirectorTrajectory();

    clearDirectorTrajectory();


    setDirectorRunningState(
        true
    );


    setDirectorStatus(
        "Program planning...",
        "running"
    );


    directorProgress.textContent =
        "Planning...";


    statusText.textContent =
        "Director trajectory hesaplanıyor...";


    try {

        // ====================================================
        // REQUEST DIRECTOR PLAN
        // ====================================================

        const data =
            await apiRequest(
                "/api/director/plan",
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

                            commands:
                                buildDirectorCommands(),

                            linear_step_mm:
                                5,

                            rotation_step_deg:
                                2,

                            revolute_step_deg:
                                2,

                            prismatic_step_mm:
                                5

                        })

                },
                1
            );


        // ====================================================
        // PLANNER ERROR
        // ====================================================

        if (
            !data?.success
        ) {

            setDirectorRunningState(
                false
            );


            directorProgress.textContent =
                "Validation failed";


            showDirectorError(
                data
            );


            return;
        }


        // ====================================================
        // TRAJECTORY CHECK
        // ====================================================

        if (
            !Array.isArray(
                data.trajectory
            )
            ||
            data.trajectory.length === 0
        ) {

            throw new Error(
                "Backend trajectory üretmedi."
            );
        }


        // ====================================================
        // EXTRACT TCP PATH FROM TRAJECTORY
        // ====================================================

        const tcpPath =
            data.trajectory
                .map(
                    point =>
                        point.tcp
                )
                .filter(

                    tcp =>
                        Array.isArray(
                            tcp
                        )
                        &&
                        tcp.length >= 3

                );


        console.log(
            "Director TCP Path:",
            tcpPath
        );


        // ====================================================
        // TRAJECTORY CHECK
        // ====================================================

        if (
            tcpPath.length < 2
        ) {

            console.warn(
                "Trajectory çizmek için yeterli TCP noktası yok.",
                tcpPath
            );

        }

        else {

            // ================================================
            // CREATE TRAJECTORY LINE
            // ================================================

            createDirectorTrajectory(
                tcpPath
            );

        }


        // ====================================================
        // START ANIMATION
        // ====================================================

        startDirectorAnimation(
            data.trajectory
        );

    }

    catch (error) {

        console.error(
            "Director error:",
            error
        );


        setDirectorRunningState(
            false
        );


        setDirectorStatus(
            error.message,
            "error"
        );


        directorProgress.textContent =
            "Error";


        statusText.textContent =
            "Director hatası";
    }
}



// ============================================================
// UPDATE TRAJECTORY TRACE
// ============================================================

function updateDirectorTrajectoryTrace(
    pointCount
) {

    if (
        !directorTrajectoryLine
    ) {

        return;
    }


    const totalPoints =
        directorTrajectoryLine
            .geometry
            .attributes
            .position
            .count;


    const visiblePoints =
        Math.min(

            totalPoints,

            Math.max(
                2,
                pointCount
            )

        );


    directorTrajectoryLine
        .geometry
        .setDrawRange(
            0,
            visiblePoints
        );
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
// GET FRAME Z AXIS
//
// Standard DH:
// Joint i hareket ekseni z_(i-1)
//
// Yani joint görselleştirmesinde parent frame'in
// lokal Z eksenini kullanıyoruz.
// ============================================================

function getFrameZAxis(
    frame
) {

    const R =
        frame.rotation;


    return new THREE.Vector3(

        R[0][2],

        R[1][2],

        R[2][2]

    ).normalize();
}


// ============================================================
// GET JOINT NAME FROM DH ROW
// ============================================================

function getJointNameFromDHRow(
    dhRow
) {

    const text = [

        dhRow.theta ?? "0",

        dhRow.d ?? "0",

        dhRow.a ?? "0",

        dhRow.alpha ?? "0"

    ].join(" ");


    const match =
        text.match(
            /\bq\d+\b/
        );


    return match
        ? match[0]
        : null;
}


// ============================================================
// EVALUATE DH EXPRESSION
//
// Örnek:
//
// q1
// q1 + L0
// L0 + q1
// q1 + L3
// 2*q1 + L0
//
// gibi ifadeleri sayısal olarak çözer.
// ============================================================

function evaluateDHExpression(
    expression,
    overrideValues = {}
) {

    let numericExpression =
        String(
            expression ?? "0"
        );


    const values = {

        ...currentRobotValues,

        ...overrideValues

    };


    const names =
        Object.keys(
            values
        ).sort(

            (a, b) =>
                b.length - a.length

        );


    for (
        const name
        of names
    ) {

        const value =
            Number(
                values[name]
            );


        if (
            !Number.isFinite(
                value
            )
        ) {

            continue;
        }


        numericExpression =
            numericExpression.replace(

                new RegExp(
                    `\\b${name}\\b`,
                    "g"
                ),

                `(${value})`

            );
    }


    // Sadece matematiksel ifadeye izin ver
    if (
        !/^[0-9eE+\-*/().\s]+$/.test(
            numericExpression
        )
    ) {

        console.warn(

            "DH ifadesi çözülemedi:",

            expression,

            "->",

            numericExpression

        );


        return null;
    }


    try {

        const result =
            Function(

                `"use strict"; ` +
                `return (${numericExpression});`

            )();


        if (
            Number.isFinite(
                result
            )
        ) {

            return Number(
                result
            );
        }

    }

    catch (error) {

        console.warn(

            "DH expression error:",

            expression,

            error

        );
    }


    return null;
}


// ============================================================
// PRISMATIC DISTANCE
//
// Kritik nokta:
//
// d = q1
//       -> q1
//
// d = q1 + L0
//       -> L0 + q1
//
// Dolayısıyla sabit offset kaybolmuyor.
// ============================================================

function getPrismaticDistance(
    dhRow,
    jointName,
    qValue
) {

    const result =
        evaluateDHExpression(

            dhRow.d,

            {

                [jointName]:
                    qValue

            }

        );


    if (
        result !== null
    ) {

        return result;
    }


    // Fallback
    return Number(
        qValue
    ) || 0;
}

// ============================================================
// PRISMATIC VISUAL STATE
//
// Örnek:
//
// d = L0 + q1
//
// dMin     = L0 + qMin
// dCurrent = L0 + qCurrent
// dMax     = L0 + qMax
//
// Sabit fiziksel gövde:
//      dMin
//
// Güncel stroke:
//      dCurrent - dMin
//
// Maksimum stroke:
//      dMax - dMin
// ============================================================

function getPrismaticVisualState(
    dhRow,
    jointName,
    qCurrent
) {

    const qMin =
        Number.isFinite(
            Number(
                dhRow.min
            )
        )
            ? Number(
                dhRow.min
            )
            : 0;


    const qMax =
        Number.isFinite(
            Number(
                dhRow.max
            )
        )
            ? Number(
                dhRow.max
            )
            : qCurrent;


    const dMin =
        getPrismaticDistance(

            dhRow,

            jointName,

            qMin

        );


    const dCurrent =
        getPrismaticDistance(

            dhRow,

            jointName,

            qCurrent

        );


    const dMax =
        getPrismaticDistance(

            dhRow,

            jointName,

            qMax

        );


    return {

        qMin,

        qMax,

        dMin,

        dCurrent,

        dMax,

        currentStroke:
            dCurrent - dMin,

        maxStroke:
            dMax - dMin

    };
}


// ============================================================
// CREATE BOX BETWEEN TWO POINTS
//
// Prizmatik gövde ve teleskopik stroke için ortak helper.
// ============================================================

function createBoxBetweenPoints(
    startPoint,
    endPoint,
    thickness,
    materialOptions = {}
) {

    const start =
        startPoint.clone();


    const end =
        endPoint.clone();


    const direction =
        new THREE.Vector3()
            .subVectors(
                end,
                start
            );


    const length =
        direction.length();


    if (
        length < 0.001
    ) {

        return null;
    }


    const midpoint =
        new THREE.Vector3()
            .addVectors(
                start,
                end
            )
            .multiplyScalar(
                0.5
            );


    const geometry =
        new THREE.BoxGeometry(

            thickness,

            length,

            thickness

        );


    const material =
        new THREE.MeshStandardMaterial({

            color:
                0x5d5d5d,

            metalness:
                0.30,

            roughness:
                0.5,

            ...materialOptions

        });


    const body =
        new THREE.Mesh(

            geometry,

            material

        );


    body.position.copy(
        midpoint
    );


    // BoxGeometry'nin uzun ekseni Y.
    // Bunu gerçek eksene çevir.
    body.quaternion
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


    return body;
}


// ============================================================
// CREATE PRISMATIC FIXED BODY
//
// KRİTİK:
//
// Fiziksel minimum link uzunluğu:
//
//      d(qMin)
//
// Örn:
//
//      d = L0 + q1
//
// ise:
//
//      physicalLength = L0 + q1_min
//
// qMax burada kullanılmaz.
// ============================================================

function createPrismaticFixedBody(
    parentFrame,
    dhRow,
    jointName,
    qCurrent,
    thickness = 26
) {

    const state =
        getPrismaticVisualState(

            dhRow,

            jointName,

            qCurrent

        );


    const parentPosition =
        new THREE.Vector3(
            ...parentFrame.position
        );


    const axis =
        getFrameZAxis(
            parentFrame
        );


    const bodyEnd =
        parentPosition
            .clone()
            .add(

                axis
                    .clone()
                    .multiplyScalar(
                        state.dMin
                    )

            );


    return createBoxBetweenPoints(

        parentPosition,

        bodyEnd,

        thickness,

        {

            color:
                0x555555

        }

    );
}


// ============================================================
// CREATE PRISMATIC EXTENSION
//
// Yalnızca:
//
//      dCurrent - dMin
//
// kadar uzar.
//
// q = qMin olduğunda görünmez.
//
// q arttıkça sabit gövdenin ucundan çıkar.
// ============================================================

function createPrismaticExtension(
    parentFrame,
    dhRow,
    jointName,
    qCurrent,
    thickness = 16
) {

    const state =
        getPrismaticVisualState(

            dhRow,

            jointName,

            qCurrent

        );


    const parentPosition =
        new THREE.Vector3(
            ...parentFrame.position
        );


    const axis =
        getFrameZAxis(
            parentFrame
        );


    // Sabit gövdenin bittiği nokta
    const minimumPoint =
        parentPosition
            .clone()
            .add(

                axis
                    .clone()
                    .multiplyScalar(
                        state.dMin
                    )

            );


    // Güncel slider/frame noktası
    const currentPoint =
        parentPosition
            .clone()
            .add(

                axis
                    .clone()
                    .multiplyScalar(
                        state.dCurrent
                    )

            );


    return createBoxBetweenPoints(

        minimumPoint,

        currentPoint,

        thickness,

        {

            color:
                0x858585

        }

    );
}


// ============================================================
// GET PRISMATIC SLIDER POSITION
// ============================================================

function getPrismaticSliderPosition(
    parentFrame,
    dhRow,
    jointName,
    qCurrent
) {

    const parentPosition =
        new THREE.Vector3(
            ...parentFrame.position
        );


    const axis =
        getFrameZAxis(
            parentFrame
        );


    const distance =
        getPrismaticDistance(

            dhRow,

            jointName,

            qCurrent

        );


    return parentPosition
        .clone()
        .add(

            axis
                .clone()
                .multiplyScalar(
                    distance
                )

        );
}


// ============================================================
// CREATE PRISMATIC SLIDER
// ============================================================

function createPrismaticSlider(
    parentFrame,
    dhRow,
    jointName,
    qCurrent,
    size = 34
) {

    const position =
        getPrismaticSliderPosition(

            parentFrame,

            dhRow,

            jointName,

            qCurrent

        );


    const geometry =
        new THREE.BoxGeometry(

            size,

            size,

            size

        );


    const material =
        new THREE.MeshStandardMaterial({

            color:
                0xd8d8d8,

            metalness:
                0.15,

            roughness:
                0.5

        });


    const slider =
        new THREE.Mesh(

            geometry,

            material

        );


    slider.position.copy(
        position
    );


    // Parent frame orientation'ıyla hizala

    const R =
        parentFrame.rotation;


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


    slider.setRotationFromMatrix(
        matrix
    );


    return slider;
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
// CREATE REVOLUTE JOINT
//
// Silindirin merkez ekseni = gerçek revolute ekseni.
// ============================================================

function createRevoluteJoint(
    parentFrame,
    radius = 18,
    height = 34
) {

    const geometry =
        new THREE.CylinderGeometry(

            radius,

            radius,

            height,

            24

        );


    const material =
        new THREE.MeshStandardMaterial({

            color:
                0xd8d8d8,

            metalness:
                0.15,

            roughness:
                0.5

        });


    const joint =
        new THREE.Mesh(

            geometry,

            material

        );


    joint.position.set(
        ...parentFrame.position
    );


    const axis =
        getFrameZAxis(
            parentFrame
        );


    // Three.js Cylinder uzun ekseni Y.
    //
    // Y -> joint Z axis

    joint.quaternion
        .setFromUnitVectors(

            new THREE.Vector3(
                0,
                1,
                0
            ),

            axis

        );


    return joint;
}


// ============================================================
// DRAW ROBOT
//
// REVOLUTE:
//
// parentFrame
//      ◎──────────── childFrame
//
// PRISMATIC:
//
// parent
//   │
//   ████████████████░░░░░■
//   |<--- dMin ---->|     ↑
//         sabit         slider
//
//   ░ kısmı:
//      dCurrent - dMin
//
// qMin/qMax link boyu olarak kullanılmaz.
// ============================================================

function drawRobot(
    frames
) {

    if (
        !Array.isArray(
            frames
        )
        ||
        frames.length === 0
    ) {

        return;
    }


    currentFrames =
        frames;


    clearRobot();


    // ========================================================
    // EACH DH ROW
    // ========================================================

    for (
        let i = 0;
        i < frames.length - 1;
        i++
    ) {

        const parentFrame =
            frames[i];


        const childFrame =
            frames[
                i + 1
            ];


        const dhRow =
            currentDHTable[i];


        if (
            !dhRow
        ) {

            continue;
        }


        // ====================================================
        // PRISMATIC
        // ====================================================

        if (
            dhRow.type === "P"
        ) {

            const jointName =
                getJointNameFromDHRow(
                    dhRow
                );


            if (
                !jointName
            ) {

                console.warn(

                    "Prismatic q sembolü bulunamadı:",

                    dhRow

                );


                continue;
            }


            const qCurrent =
                Number(

                    currentRobotValues[
                        jointName
                    ]
                    ??
                    0

                );


            // =================================================
            // 1) FIXED PHYSICAL BODY
            //
            // Uzunluğu:
            //
            //      d(qMin)
            //
            // Örnek:
            //
            //      L0 + qMin
            // =================================================

            const fixedBody =
                createPrismaticFixedBody(

                    parentFrame,

                    dhRow,

                    jointName,

                    qCurrent

                );


            if (
                fixedBody
            ) {

                robotGroup.add(
                    fixedBody
                );
            }


            // =================================================
            // 2) TELESCOPIC EXTENSION
            //
            // Uzunluğu:
            //
            //      dCurrent - dMin
            // =================================================

            const extension =
                createPrismaticExtension(

                    parentFrame,

                    dhRow,

                    jointName,

                    qCurrent

                );


            if (
                extension
            ) {

                robotGroup.add(
                    extension
                );
            }


            // =================================================
            // 3) MOVING PRISMATIC CUBE / CARRIAGE
            // =================================================

            const slider =
                createPrismaticSlider(

                    parentFrame,

                    dhRow,

                    jointName,

                    qCurrent

                );


            robotGroup.add(
                slider
            );


            // =================================================
            // 4) CURRENT PRISMATIC POSITION
            // =================================================

            const sliderPosition =
                getPrismaticSliderPosition(

                    parentFrame,

                    dhRow,

                    jointName,

                    qCurrent

                );


            // =================================================
            // 5) DH "a" OFFSET AFTER PRISMATIC MOTION
            //
            // Eğer a != 0 ise gerçek child frame,
            // slider noktasından yana kaymıştır.
            //
            // Bu sabit linktir.
            // =================================================

            const rigidLink =
                createLink(

                    sliderPosition.toArray(),

                    childFrame.position

                );


            if (
                rigidLink
            ) {

                robotGroup.add(
                    rigidLink
                );
            }


            // Prismatic için normal parent-child link yok.
            continue;
        }


        // ====================================================
        // REVOLUTE
        // ====================================================

        if (
            dhRow.type === "R"
        ) {

            const revoluteJoint =
                createRevoluteJoint(
                    parentFrame
                );


            robotGroup.add(
                revoluteJoint
            );
        }


        // ====================================================
        // NORMAL RIGID LINK
        // ====================================================

        const link =
            createLink(

                parentFrame.position,

                childFrame.position

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
    // FRAME AXES
    // ========================================================

    for (
        const frame
        of frames
    ) {

        robotGroup.add(

            createFrameAxes(
                frame
            )

        );
    }


    // Kamera burada resetlenmez.
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
        maxSize
        *
        2.2;


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

        return false;
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

        stopHoldJog();


        alert(
            "Geçerli bir Linear Jog step değeri gir."
        );


        return false;
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

            stopHoldJog();


            statusText.textContent =

                `Jog başarısız — hata ` +

                `${Number(
                    data.position_error
                ).toFixed(2)} mm`;


            return false;
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


        return true;

    }

    catch (error) {

        console.error(
            error
        );


        stopHoldJog();


        statusText.textContent =
            "Linear Jog hatası";


        alert(
            error.message
        );


        return false;
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

        return false;
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

        stopHoldJog();


        alert(
            "Geçerli Joint Jog step değerleri gir."
        );


        return false;
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


        return true;

    }

    catch (error) {

        console.error(
            error
        );


        stopHoldJog();


        statusText.textContent =
            "Joint Jog hatası";


        alert(
            error.message
        );


        return false;
    }
}


// ============================================================
// HOLD TO JOG
//
// Aynı anda yalnızca bir backend request çalışır.
//
// Akış:
//
// pointerdown
//      ↓
// jog
//      ↓
// cevap bekle
//      ↓
// hala basılı mı?
//      ↓
// sonraki jog
//
// Böylece Render'a üst üste request yığılmaz.
// ============================================================

let jogHoldActive =
    false;


let jogHoldToken =
    0;


async function startHoldJog(
    jogFunction
) {

    stopHoldJog();


    jogHoldActive =
        true;


    const myToken =
        ++jogHoldToken;


    // ========================================================
    // FIRST STEP IMMEDIATELY
    // ========================================================

    const firstSuccess =
        await jogFunction();


    if (
        !firstSuccess
    ) {

        stopHoldJog();

        return;
    }


    // Tek tık ile hold arasında küçük gecikme
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

        const success =
            await jogFunction();


        if (
            !success
        ) {

            stopHoldJog();

            break;
        }


        // Backend'i gereksiz dövmemek için
        // kısa bekleme
        await sleep(
            40
        );
    }
}


function stopHoldJog() {

    jogHoldActive =
        false;


    jogHoldToken++;
}


// ============================================================
// JOINT BUTTON BIND
// ============================================================

function bindJointJogButtons() {

    // ========================================================
    // JOINT MINUS
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
    // JOINT PLUS
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
            "pointerdown",
            event => {

                event.preventDefault();


                button.setPointerCapture(
                    event.pointerId
                );


                startHoldJog(
                    () =>
                        performLinearJog(

                            button.dataset.axis,

                            Number(
                                button.dataset.direction
                            )

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


// ============================================================
// GLOBAL JOG STOP SAFETY
// ============================================================

window.addEventListener(
    "pointerup",
    stopHoldJog
);


window.addEventListener(
    "pointercancel",
    stopHoldJog
);


window.addEventListener(
    "blur",
    stopHoldJog
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

                    stopHoldJog();


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

    stopHoldJog();


    stopDirectorAnimation(
        false
    );


    clearDirectorTrajectory();
    try {

        statusText.textContent =
            "Preset yükleniyor...";


        const preset =
            await apiRequest(

                `/api/presets/${encodeURIComponent(
                    presetId
                )}`,

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

        stopHoldJog();


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

        stopHoldJog();


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
    () => {

        stopHoldJog();

        buildRobot();

    }
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
// DIRECTOR EVENT BINDINGS
// ============================================================

// Builder tab
builderTabButton.addEventListener(
    "click",
    () => {

        setRightPanelMode(
            "builder"
        );

    }
);


// Director tab
directorTabButton.addEventListener(
    "click",
    () => {

        setRightPanelMode(
            "director"
        );

    }
);


// ============================================================
// COMMAND BUTTONS
// ============================================================

directorCommandButtons.forEach(
    button => {

        button.addEventListener(
            "click",
            () => {

                addDirectorCommand({

                    type:
                        button.dataset
                            .directorCommand,

                    axis:
                        button.dataset.axis,

                    label:
                        button.dataset.label,

                    unit:
                        button.dataset.unit

                });

            }
        );

    }
);


// ============================================================
// CLEAR
// ============================================================

directorClearButton.addEventListener(
    "click",
    clearDirectorProgram
);


// ============================================================
// RUN
// ============================================================

directorRunButton.addEventListener(
    "click",
    runDirectorProgram
);


// ============================================================
// STOP
// ============================================================

directorStopButton.addEventListener(
    "click",
    () => {

        stopDirectorAnimation(
            true
        );

    }
);


// ============================================================
// AI DIRECTOR EVENT BINDINGS
// ============================================================

// Generate Program
if (aiGenerateButton) {

    aiGenerateButton.addEventListener(
        "click",
        () => {

            console.log(
                "AI Generate clicked"
            );

            interpretAICommand();

        }
    );

}


// Generate & Run
if (aiRunButton) {

    aiRunButton.addEventListener(
        "click",
        () => {

            console.log(
                "AI Generate & Run clicked"
            );

            runAITask();

        }
    );

}


// CTRL + ENTER = Generate & Run
if (aiDirectorInput) {

    aiDirectorInput.addEventListener(
        "keydown",
        event => {

            if (
                event.ctrlKey
                &&
                event.key === "Enter"
            ) {

                event.preventDefault();

                console.log(
                    "AI Ctrl+Enter"
                );

                runAITask();

            }

        }
    );

}


// ============================================================
// RESET
// ============================================================

if (directorResetButton) {

    directorResetButton.addEventListener(
        "click",
        resetDirectorToHome
    );

}


// ============================================================
// SPEED
// ============================================================

directorSpeedInput.addEventListener(
    "input",
    () => {

        directorSpeedValue.textContent =

            `${directorSpeedInput.value}%`;

    }
);


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