/**
 * Test utility functions for splat-transform tests.
 */

import { Column, DataTable } from '../../src/lib/index.js';

/**
 * Creates a minimal DataTable with standard Gaussian splat columns.
 * Values are deterministic based on index for reproducible tests.
 *
 * @param {number} count - Number of splats to create
 * @param {object} [options] - Options for customizing the data
 * @param {boolean} [options.includeSH=false] - Include spherical harmonics columns
 * @param {number} [options.shBands=1] - Number of SH bands (1=9 coeffs, 2=24, 3=45)
 * @returns {DataTable} A DataTable with Gaussian splat data
 */
function createTestDataTable(count, options = {}) {
    const { includeSH = false, shBands = 1 } = options;

    // Constants for encoding
    const SH_C0 = 0.28209479177387814;
    const packClr = (c) => (c - 0.5) / SH_C0;
    const packOpacity = (opacity) => {
        if (opacity <= 0) return -20;
        if (opacity >= 1) return 20;
        return -Math.log(1 / opacity - 1);
    };

    // Create base columns
    const columns = [
        new Column('x', new Float32Array(count)),
        new Column('y', new Float32Array(count)),
        new Column('z', new Float32Array(count)),
        new Column('scale_0', new Float32Array(count)),
        new Column('scale_1', new Float32Array(count)),
        new Column('scale_2', new Float32Array(count)),
        new Column('f_dc_0', new Float32Array(count)),
        new Column('f_dc_1', new Float32Array(count)),
        new Column('f_dc_2', new Float32Array(count)),
        new Column('opacity', new Float32Array(count)),
        new Column('rot_0', new Float32Array(count)),
        new Column('rot_1', new Float32Array(count)),
        new Column('rot_2', new Float32Array(count)),
        new Column('rot_3', new Float32Array(count))
    ];

    // Grid dimensions for positioning
    const gridSize = Math.ceil(Math.sqrt(count));
    const spacing = 1.0;
    const scale = 0.1;

    for (let i = 0; i < count; i++) {
        const gx = i % gridSize;
        const gz = Math.floor(i / gridSize);

        // Position: grid layout
        columns[0].data[i] = (gx - gridSize / 2) * spacing; // x
        columns[1].data[i] = 0; // y
        columns[2].data[i] = (gz - gridSize / 2) * spacing; // z

        // Scale: log-encoded
        columns[3].data[i] = Math.log(scale);
        columns[4].data[i] = Math.log(scale);
        columns[5].data[i] = Math.log(scale);

        // Color: varying based on position
        const r = (gx + 1) / (gridSize + 1);
        const g = (gz + 1) / (gridSize + 1);
        const b = 0.5;
        columns[6].data[i] = packClr(r);
        columns[7].data[i] = packClr(g);
        columns[8].data[i] = packClr(b);

        // Opacity: sigmoid-encoded
        columns[9].data[i] = packOpacity(0.9);

        // Rotation: identity quaternion (rot_0 = w)
        columns[10].data[i] = 1; // rot_0 (w)
        columns[11].data[i] = 0; // rot_1 (x)
        columns[12].data[i] = 0; // rot_2 (y)
        columns[13].data[i] = 0; // rot_3 (z)
    }

    // Add spherical harmonics if requested
    if (includeSH) {
        const shCoeffs = [0, 3, 8, 15][shBands];
        for (let c = 0; c < shCoeffs * 3; c++) {
            const shColumn = new Column(`f_rest_${c}`, new Float32Array(count));
            // Small deterministic values
            for (let i = 0; i < count; i++) {
                shColumn.data[i] = ((c + i) % 10 - 5) * 0.01;
            }
            columns.push(shColumn);
        }
    }

    return new DataTable(columns);
}

/**
 * Creates a 4x4 grid of splats (16 total) for minimal testing.
 * @param {object} [options] - Options passed to createTestDataTable
 * @returns {DataTable} A DataTable with 16 Gaussian splats
 */
function createMinimalTestData(options = {}) {
    return createTestDataTable(16, options);
}

/**
 * Adds the spacetime columns a .sogst file needs to a splat DataTable.
 *
 * t_center is spread across the clip so the splats bucket into several temporal
 * segments, and every `persistentEvery`-th splat is given a wide t_sigma so it
 * lands in the persistent group. Velocities vary per axis so a swapped axis in
 * the writer shows up as a field mismatch rather than a wash.
 *
 * @param {DataTable} dataTable - Table to extend, modified in place
 * @param {object} [options] - Options
 * @param {number} [options.timeMax=1.0] - Clip length in seconds
 * @param {boolean} [options.includeAccel=false] - Also add ax, ay, az
 * @param {number} [options.persistentEvery=5] - Every nth splat is long-lived
 * @param {number[]} [options.gap] - [start, end] seconds left with no splats, so
 * the encoder has to emit empty segments. No real capture produces one.
 * @returns {DataTable} The same DataTable, for chaining
 */
function addSpacetimeColumns(dataTable, options = {}) {
    const { timeMax = 1.0, includeAccel = false, persistentEvery = 5, gap } = options;
    const count = dataTable.numRows;

    const make = name => new Column(name, new Float32Array(count));
    const vx = make('vx'), vy = make('vy'), vz = make('vz');
    const tCenter = make('t_center'), tSigma = make('t_sigma');

    for (let i = 0; i < count; i++) {
        // distinct per-axis velocities, both signs
        vx.data[i] = (i % 7) * 0.1 - 0.3;
        vy.data[i] = (i % 5) * -0.2 + 0.4;
        vz.data[i] = (i % 3) * 0.05;

        const t = count > 1 ? (i / (count - 1)) * timeMax : 0;

        // squeeze every t_center into the intervals either side of the gap, so
        // the segments covering the gap come out empty
        if (gap) {
            const [gapStart, gapEnd] = gap;
            const kept = timeMax - (gapEnd - gapStart);
            const scaled = (t / timeMax) * kept;
            tCenter.data[i] = scaled < gapStart ? scaled : scaled + (gapEnd - gapStart);
        } else {
            tCenter.data[i] = t;
        }

        // short-lived by default; every nth splat spans enough of the clip to
        // be classified persistent
        tSigma.data[i] = (i % persistentEvery === 0) ? 0.25 : 0.005 + (i % 3) * 0.002;
    }

    for (const column of [vx, vy, vz, tCenter, tSigma]) {
        dataTable.addColumn(column);
    }

    if (includeAccel) {
        const ax = make('ax'), ay = make('ay'), az = make('az');
        for (let i = 0; i < count; i++) {
            ax.data[i] = (i % 4) * 0.02 - 0.03;
            ay.data[i] = (i % 6) * -0.01;
            az.data[i] = (i % 5) * 0.015 + 0.01;
        }
        for (const column of [ax, ay, az]) {
            dataTable.addColumn(column);
        }
    }

    return dataTable;
}

/**
 * Encodes a DataTable to PLY binary format.
 * @param {DataTable} dataTable - The data to encode
 * @param {string[]} comments - Header comments (without the leading `comment `)
 * @returns {Uint8Array} PLY file as binary data
 */
function encodePlyBinary(dataTable, comments = []) {
    const columns = dataTable.columns;
    const numRows = dataTable.numRows;

    // Build header
    const columnTypeToPlyType = (type) => {
        switch (type) {
            case 'float32': return 'float';
            case 'float64': return 'double';
            case 'int8': return 'char';
            case 'uint8': return 'uchar';
            case 'int16': return 'short';
            case 'uint16': return 'ushort';
            case 'int32': return 'int';
            case 'uint32': return 'uint';
        }
    };

    const headerLines = [
        'ply',
        'format binary_little_endian 1.0',
        ...comments.map(c => `comment ${c}`),
        `element vertex ${numRows}`,
        ...columns.map(c => `property ${columnTypeToPlyType(c.dataType)} ${c.name}`),
        'end_header'
    ];
    const headerStr = headerLines.join('\n') + '\n';
    const headerBytes = new TextEncoder().encode(headerStr);

    // Calculate row size
    const sizes = columns.map(c => c.data.BYTES_PER_ELEMENT);
    const rowSize = sizes.reduce((a, b) => a + b, 0);

    // Create output buffer
    const dataSize = numRows * rowSize;
    const result = new Uint8Array(headerBytes.length + dataSize);

    // Write header
    result.set(headerBytes, 0);

    // Write data
    const buffers = columns.map(c => new Uint8Array(c.data.buffer, c.data.byteOffset, c.data.byteLength));
    let offset = headerBytes.length;

    for (let row = 0; row < numRows; row++) {
        for (let col = 0; col < columns.length; col++) {
            const size = sizes[col];
            const colOffset = row * size;
            result.set(buffers[col].subarray(colOffset, colOffset + size), offset);
            offset += size;
        }
    }

    return result;
}

export { createTestDataTable, createMinimalTestData, addSpacetimeColumns, encodePlyBinary };
