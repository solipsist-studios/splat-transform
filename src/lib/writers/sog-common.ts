import { DataTable } from '../data-table';

/**
 * Per-column [min, max] over the rows referenced by `indices`.
 *
 * @param dataTable - Table to scan.
 * @param columnNames - Columns to measure.
 * @param indices - Row indices to include.
 * @returns One [min, max] pair per column name, in order.
 * @ignore
 */
const calcMinMax = (dataTable: DataTable, columnNames: string[], indices: Uint32Array) => {
    const columns = columnNames.map(name => dataTable.getColumnByName(name));
    const minMax = columnNames.map(() => [Infinity, -Infinity]);
    const row = {};

    for (let i = 0; i < indices.length; ++i) {
        const r = dataTable.getRow(indices[i], row, columns);

        for (let j = 0; j < columnNames.length; ++j) {
            const value = r[columnNames[j]];
            if (value < minMax[j][0]) minMax[j][0] = value;
            if (value > minMax[j][1]) minMax[j][1] = value;
        }
    }

    return minMax;
};

/**
 * The SOG means transform: sign(x) * ln(1 + |x|). Monotonic, so it can be
 * applied to the endpoints of a range to get the range in log space.
 * @ignore
 */
const logTransform = (value: number) => {
    return Math.sign(value) * Math.log(Math.abs(value) + 1);
};

/**
 * Near-square texture dimensions holding `n` row-major texels.
 *
 * @param n - Number of texels required. Must be > 0.
 * @param align - Round both axes up to this multiple. SOG uses 4; the SOGST
 * spec pins the dimensions to exactly `ceil(sqrt(n))` by `ceil(n / width)`, so
 * it passes 1.
 * @returns The texture width and height.
 * @ignore
 */
const texDims = (n: number, align = 4) => {
    const width = Math.ceil(Math.sqrt(n) / align) * align;
    const height = Math.ceil(n / width / align) * align;
    return { width, height };
};

/**
 * The shared means/motion quantizer: log-transform each axis, then normalize
 * over the per-axis range to 16 bits and split into low and high byte planes.
 *
 * Both planes are RGBA in `indices` order, so texel i of the output holds row
 * `indices[i]` — every plane built this way stays in sync with the others.
 *
 * @param dataTable - Table holding the source columns.
 * @param columnNames - The three axis columns, in x, y, z order.
 * @param indices - Row order to emit.
 * @returns The two byte planes and the log-space mins/maxs for meta.json.
 * @ignore
 */
const computeSplit16Planes = (dataTable: DataTable, columnNames: string[], indices: Uint32Array) => {
    const minMax = calcMinMax(dataTable, columnNames, indices).map(v => v.map(logTransform));
    const columns = columnNames.map(name => dataTable.getColumnByName(name).data);
    const numRows = indices.length;

    const lo = new Uint8Array(numRows * 4);
    const hi = new Uint8Array(numRows * 4);

    for (let i = 0; i < numRows; ++i) {
        const idx = indices[i];

        for (let c = 0; c < 3; ++c) {
            const [min, max] = minMax[c];
            const t = logTransform(columns[c][idx]);

            // a zero-width range uses a span of 1.0, so it quantizes to 0
            // rather than dividing by zero; the decode is constant at min
            const span = max > min ? max - min : 1.0;
            const v = Math.max(0, Math.min(65535, Math.round(65535 * (t - min) / span)));

            lo[i * 4 + c] = v & 0xff;
            hi[i * 4 + c] = (v >> 8) & 0xff;
        }

        lo[i * 4 + 3] = 0xff;
        hi[i * 4 + 3] = 0xff;
    }

    return {
        lo,
        hi,
        mins: minMax.map(v => v[0]),
        maxs: minMax.map(v => v[1])
    };
};

export { calcMinMax, computeSplit16Planes, logTransform, texDims };
