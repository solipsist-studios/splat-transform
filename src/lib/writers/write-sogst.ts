import { basename } from 'pathe';

import { requireSogstClip, type SogstClip } from './sogst-clip';
import { logWrittenFile } from './utils';
import { Column, DataTable, sortMortonOrder, type TypedArray } from '../data-table';
import { type FileSystem, writeStoredZip, type StoredZipEntry } from '../io/write';
import { kmeansInterleaved } from '../spatial';
import type { DeviceCreator } from '../types';
import { logger, sigmoid } from '../utils';
import { version } from '../version';
import { runEncodeWebp, runQuantize1dColumns } from '../workers';

// Temporal segment length in seconds. Splats are bucketed by t_center into
// segments of this length so a player can cull by time.
const DEFAULT_SEGMENT_DURATION = 0.1;

// Half-width of a splat's active interval, in temporal standard deviations.
const DEFAULT_K_SIGMA = 3.8;

// A splat whose active interval spans more than this many segment durations is
// 'persistent' and always drawn. See computeOrder for why this matters.
const DEFAULT_PERSISTENT_SPAN_MULT = 3.0;

// Guard against a mistyped segment duration; see computeOrder.
const MAX_SEGMENTS = 65536;

// A ZIP local file header is fixed at 30 bytes by the format itself (PKWARE
// APPNOTE 4.3.7): signature 4, version 2, flags 2, method 2, modtime 2,
// moddate 2, crc32 4, compressed size 4, uncompressed size 4, name length 2,
// extra length 2. It is not a budget with headroom — an entry's header is
// exactly this plus its filename, and only because writeStoredZip emits no
// extra field and no data descriptor (spec §2). Adding either would grow every
// header and silently invalidate reveal_bytes / geometry_bytes.
//
// Nothing rests on this constant being right by inspection: writeSogst
// re-derives both offsets from the header offsets writeStoredZip actually
// wrote and throws on a mismatch (see the end of this file), and the tests
// assert method 0, extra length 0 and no data descriptor on every entry.
const ZIP_LOCAL_HEADER_SIZE = 30;

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

const motionNames = ['vx', 'vy', 'vz'];
const accelNames = ['ax', 'ay', 'az'];
const trbfNames = ['t_center', 't_sigma'];


/**
 * A single temporal segment: the time range its members are actually active
 * over, and the contiguous index range they occupy in the file.
 */
type Segment = {
    t0: number;
    t1: number;
    range: [number, number];
};

/**
 * The temporal segment table written to `meta.segments`.
 */
type Segments = {
    duration: number;
    k_sigma: number;
    persistent_span_mult: number;
    persistent: [number, number];
    list: Segment[];
};

/**
 * Options for writing a SOGST (SOG spacetime) file.
 */
type WriteSogstOptions = {
    /** Path to the output .sogst file. */
    filename: string;
    /** The splat data to write. Must carry vx, vy, vz, t_center and t_sigma. */
    dataTable: DataTable;
    /** Number of k-means iterations for higher-order SH compression. */
    iterations: number;
    /**
     * Clip-level scalars. `timeMin`, `timeMax` and `fps` are required and have
     * no defaults — see requireSogstClip for why.
     */
    clip: Partial<SogstClip>;
    /** Temporal segment length in seconds. 0 disables segmentation. Default 0.1. */
    segmentDuration?: number;
    /** Active-interval half-width in temporal standard deviations. Default 3.8. */
    kSigma?: number;
    /** Persistent-splat threshold, in segment durations. Default 3.0. */
    persistentSpanMult?: number;
    /** Optional function to create a GPU device for SH clustering. */
    createDevice?: DeviceCreator;
};

/**
 * Computes the splat ordering and the temporal segment table.
 *
 * Splats whose active interval `[t_center ∓ kSigma·|t_sigma|]` spans more than
 * `persistentSpanMult` segment durations are 'persistent' and placed first.
 * The rest bucket by `t_center` into fixed-length segments. A player then draws
 * `[0, persistentEnd)` plus the single contiguous index span covering every
 * segment whose `[t0, t1]` contains the current time.
 *
 * Pulling long-lived splats out of the buckets is what makes that span tight:
 * a segment's `[t0, t1]` is the union of its members' active intervals, so one
 * long-lived splat left in an early bucket would keep that segment active for
 * most of the clip and drag the contiguous span across the whole file.
 *
 * Each group is Morton-sorted against its own bounding box, not a global one —
 * a global box would produce a valid file that simply compresses worse.
 *
 * @param dataTable - The splat data.
 * @param timeMin - Clip start time in seconds.
 * @param timeMax - Clip end time in seconds.
 * @param segmentDuration - Segment length in seconds; <= 0 disables segmentation.
 * @param kSigma - Active-interval half-width in standard deviations.
 * @param persistentSpanMult - Persistent threshold, in segment durations.
 * @returns The row ordering and the segment table (null when disabled).
 */
const computeOrder = (
    dataTable: DataTable,
    timeMin: number,
    timeMax: number,
    segmentDuration: number,
    kSigma: number,
    persistentSpanMult: number
): { indices: Uint32Array; segments: Segments | null } => {
    const numRows = dataTable.numRows;
    const indices = new Uint32Array(numRows);
    for (let i = 0; i < numRows; ++i) {
        indices[i] = i;
    }

    if (!(segmentDuration > 0)) {
        sortMortonOrder(dataTable, indices);
        return { indices, segments: null };
    }

    const tCenter = dataTable.getColumnByName('t_center').data;
    const tSigma = dataTable.getColumnByName('t_sigma').data;

    const numSegments = Math.max(1, Math.ceil((timeMax - timeMin) / segmentDuration));

    // every segment gets an entry in meta.segments.list whether or not it holds
    // any splats, so a mistyped duration would otherwise produce a valid file
    // with a megabytes-long segment table
    if (numSegments > MAX_SEGMENTS) {
        throw new Error(`sogst segmentation would need ${numSegments} segments for a ${timeMax - timeMin}s clip at ${segmentDuration}s each (limit ${MAX_SEGMENTS}). Use a longer segment duration.`);
    }

    // classify every splat as persistent or bucketed, and record its active interval
    const persistent = new Uint8Array(numRows);
    const bucket = new Int32Array(numRows);
    const lo = new Float64Array(numRows);
    const hi = new Float64Array(numRows);
    const counts = new Uint32Array(numSegments);

    let numPersistent = 0;

    for (let i = 0; i < numRows; ++i) {
        const halfSpan = kSigma * Math.abs(tSigma[i]);
        lo[i] = tCenter[i] - halfSpan;
        hi[i] = tCenter[i] + halfSpan;

        if (hi[i] - lo[i] > persistentSpanMult * segmentDuration) {
            persistent[i] = 1;
            numPersistent++;
        } else {
            const b = Math.max(0, Math.min(numSegments - 1, Math.floor((tCenter[i] - timeMin) / segmentDuration)));
            bucket[i] = b;
            counts[b]++;
        }
    }

    // counting sort: persistent splats first, then one contiguous run per segment
    const cursors = new Uint32Array(numSegments);
    let offset = numPersistent;
    for (let s = 0; s < numSegments; ++s) {
        cursors[s] = offset;
        offset += counts[s];
    }

    const list: Segment[] = [];
    for (let s = 0; s < numSegments; ++s) {
        list.push({ t0: Infinity, t1: -Infinity, range: [cursors[s], cursors[s] + counts[s]] });
    }

    let persistentCursor = 0;
    for (let i = 0; i < numRows; ++i) {
        if (persistent[i]) {
            indices[persistentCursor++] = i;
        } else {
            const segment = list[bucket[i]];
            indices[cursors[bucket[i]]++] = i;
            if (lo[i] < segment.t0) segment.t0 = lo[i];
            if (hi[i] > segment.t1) segment.t1 = hi[i];
        }
    }

    // Morton-sort each group against its own subset bounding box
    if (numPersistent > 0) {
        sortMortonOrder(dataTable, indices.subarray(0, numPersistent));
    }

    for (let s = 0; s < numSegments; ++s) {
        const [start, end] = list[s].range;
        if (end > start) {
            sortMortonOrder(dataTable, indices.subarray(start, end));
        } else {
            // an empty segment gets its nominal window and a zero-length range
            list[s].t0 = timeMin + s * segmentDuration;
            list[s].t1 = list[s].t0 + segmentDuration;
        }
    }

    return {
        indices,
        segments: {
            duration: segmentDuration,
            k_sigma: kSigma,
            persistent_span_mult: persistentSpanMult,
            persistent: [0, numPersistent],
            list
        }
    };
};

/**
 * Per-column [min, max] over the rows referenced by `indices`.
 *
 * @param dataTable - Table to scan.
 * @param columnNames - Columns to measure.
 * @param indices - Row indices to include.
 * @returns One [min, max] pair per column name, in order.
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
 * The means transform: sign(x) * ln(1 + |x|). Monotonic, so it can be applied
 * to the endpoints of a range to get the range in log space.
 *
 * @param value - The value to transform.
 * @returns The log-space value.
 */
const logTransform = (value: number) => {
    return Math.sign(value) * Math.log(Math.abs(value) + 1);
};

/**
 * Near-square texture dimensions holding `n` row-major texels, with both axes
 * rounded up to a multiple of 4 to match the SOG writer. The spec pins the
 * dimensions to `ceil(sqrt(n))` by `ceil(n / width)` but permits the roundup.
 *
 * @param n - Number of texels required. Must be > 0.
 * @returns The texture width and height.
 */
const texDims = (n: number) => {
    const width = Math.ceil(Math.sqrt(n) / 4) * 4;
    const height = Math.ceil(n / width / 4) * 4;
    return { width, height };
};

/**
 * The means/motion quantizer: log-transform each axis, then normalize over the
 * per-axis range to 16 bits and split into low and high byte planes.
 *
 * Both planes are RGBA in `indices` order, so texel i of the output holds row
 * `indices[i]` — every plane built this way stays in sync with the others.
 *
 * Values are rounded, which the spec requires and which is load-bearing rather
 * than cosmetic: a source PLY that has already been through a 16-bit encode
 * lands exactly on the quantization grid, and float round-off puts about half
 * of those values a hair below their integer. Rounding reproduces them
 * exactly; truncating drops every one of them a full LSB. Measured on the
 * capture fixtures that is a ~1300x difference in RMS error, against 2x for
 * continuous source data. `write-sog.ts` truncates instead, to stay
 * byte-identical with its own pre-3.2 output — which is why this cannot be
 * shared with it.
 *
 * @param dataTable - Table holding the source columns.
 * @param columnNames - The three axis columns, in x, y, z order.
 * @param indices - Row order to emit.
 * @returns The two byte planes and the log-space mins/maxs for meta.json.
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

/**
 * Quantizes a set of DataTable columns to a shared 256-entry codebook.
 *
 * @param dataTable - Table holding the source columns.
 * @param columnNames - Columns to quantize together.
 * @returns The codebook centroids and one uint8 label column per input.
 */
const quantizeColumns = (dataTable: DataTable, columnNames: string[]) => {
    return runQuantize1dColumns(columnNames.map(name => ({
        name,
        data: dataTable.getColumnByName(name).data
    })));
};

/**
 * Packs up to three uint8 label columns into an RGBA plane in `indices` order.
 *
 * @param labels - Label columns, one per output channel.
 * @param indices - Row order to emit.
 * @param alpha - Value for the alpha channel.
 * @returns The packed plane.
 */
const packLabels = (labels: { name: string, data: Uint8Array }[], indices: Uint32Array, alpha: number): Uint8Array => {
    const numRows = indices.length;
    const columns = labels.map(c => c.data);
    const plane = new Uint8Array(numRows * 4);

    for (let i = 0; i < numRows; ++i) {
        const idx = indices[i];
        plane[i * 4 + 0] = columns[0][idx];
        plane[i * 4 + 1] = columns.length > 1 ? columns[1][idx] : 0;
        plane[i * 4 + 2] = columns.length > 2 ? columns[2][idx] : 0;
        plane[i * 4 + 3] = alpha;
    }

    return plane;
};

/**
 * Checks that a table can be written as SOGST, and reports what it holds.
 *
 * @param dataTable - Table to check.
 * @param motionDegree - Degree advertised by the input's `sogst.*` comments, if any.
 * @returns Whether acceleration is present, the SH band count, and the t_sigma column.
 * @throws Error describing the first problem found.
 */
const validateSogstInput = (dataTable: DataTable, motionDegree: number | undefined) => {
    // temporal attributes are the whole point of the format — refuse without them
    const missing = [...motionNames, ...trbfNames].filter(name => !dataTable.hasColumn(name));
    if (missing.length > 0) {
        throw new Error(`sogst output requires temporal columns, missing: ${missing.join(', ')}. Expected a PLY carrying ${[...motionNames, ...trbfNames].join(', ')}.`);
    }

    // accel is all-or-nothing, and its presence — not the advisory comment — is
    // what sets motion.degree
    const accelPresent = accelNames.filter(name => dataTable.hasColumn(name));
    if (accelPresent.length > 0 && accelPresent.length < accelNames.length) {
        throw new Error(`sogst output requires all of ${accelNames.join(', ')} or none, found only: ${accelPresent.join(', ')}.`);
    }
    const hasAccel = accelPresent.length === accelNames.length;

    if (motionDegree !== undefined && motionDegree !== (hasAccel ? 2 : 1)) {
        throw new Error(`sogst.motion_degree is ${motionDegree} but the data ${hasAccel ? 'has' : 'has no'} ax, ay, az columns.`);
    }

    const shBands = { '9': 1, '24': 2, '-1': 3 }[shNames.findIndex(v => !dataTable.hasColumn(v))] ?? 0;

    // t_sigma is a standard deviation in seconds and divides the temporal
    // exponent, so a zero or negative value is not representable
    const tSigmaColumn = dataTable.getColumnByName('t_sigma').data;
    for (let i = 0; i < dataTable.numRows; ++i) {
        if (!(tSigmaColumn[i] > 0)) {
            throw new Error(`sogst output requires t_sigma > 0 for every splat; row ${i} has ${tSigmaColumn[i]}.`);
        }
    }

    return { hasAccel, shBands, tSigmaColumn };
};

/**
 * Packs rotations using SOG's largest-three quaternion encoding: normalize,
 * drop the largest-magnitude component (recoverable from the other three), and
 * record which one was dropped in the alpha byte.
 *
 * @param dataTable - Table holding the rot_0..3 columns.
 * @param indices - Row order to emit.
 * @returns The packed RGBA plane.
 */
const packQuats = (dataTable: DataTable, indices: Uint32Array): Uint8Array => {
    const numRows = indices.length;
    const plane = new Uint8Array(numRows * 4);
    const quatColumns = ['rot_0', 'rot_1', 'rot_2', 'rot_3'].map(name => dataTable.getColumnByName(name).data);
    const q = [0, 0, 0, 0];
    const sqrt2 = Math.sqrt(2);

    for (let i = 0; i < numRows; ++i) {
        const idx = indices[i];

        for (let j = 0; j < 4; ++j) {
            q[j] = quatColumns[j][idx];
        }

        const l = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        q.forEach((v, j) => {
            q[j] = v / l;
        });

        // find max component
        const maxComp = q.reduce((v, _, j) => (Math.abs(q[j]) > Math.abs(q[v]) ? j : v), 0);

        // invert if max component is negative, then scale to fit [-1, 1]
        const sign = q[maxComp] < 0 ? -1 : 1;
        q.forEach((v, j) => {
            q[j] = v * sign * sqrt2;
        });

        const keep = [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2]
        ][maxComp];

        // q has already been scaled by sqrt(2), so this is
        // round((s / sqrt(2) + 0.5) * 255) with s the raw component
        plane[i * 4 + 0] = Math.round(255 * (q[keep[0]] * 0.5 + 0.5));
        plane[i * 4 + 1] = Math.round(255 * (q[keep[1]] * 0.5 + 0.5));
        plane[i * 4 + 2] = Math.round(255 * (q[keep[2]] * 0.5 + 0.5));

        // mode byte: 252 + the index of the dropped component in (w,x,y,z)
        plane[i * 4 + 3] = 252 + maxComp;
    }

    return plane;
};

/**
 * Packs the temporal radial basis function: t_center and t_sigma, each
 * quantized to its own 256-entry codebook and packed into the R and G channels.
 *
 * @param dataTable - Table holding the t_center and t_sigma columns.
 * @param indices - Row order to emit.
 * @param tSigmaColumn - The t_sigma column data, already validated as positive.
 * @returns The packed plane and both codebooks, in linear (not log) space.
 */
const packTrbf = async (dataTable: DataTable, indices: Uint32Array, tSigmaColumn: TypedArray) => {
    const numRows = indices.length;
    const center = await quantizeColumns(dataTable, ['t_center']);

    // sigma is heavy-tailed, so cluster in the log domain: a linear codebook
    // spends almost all 256 entries on the tail. The codebook is mapped back to
    // linear space, so decoders are unaffected.
    const logSigma = new Float32Array(numRows);
    for (let i = 0; i < numRows; ++i) {
        logSigma[i] = Math.log(tSigmaColumn[i]);
    }
    const sigma = await runQuantize1dColumns([{ name: 't_sigma', data: logSigma }]);

    const plane = new Uint8Array(numRows * 4);
    const centerLabels = center.labels[0].data;
    const sigmaLabels = sigma.labels[0].data;
    for (let i = 0; i < numRows; ++i) {
        const idx = indices[i];
        plane[i * 4 + 0] = centerLabels[idx];
        plane[i * 4 + 1] = sigmaLabels[idx];
        plane[i * 4 + 3] = 255;
    }

    return {
        plane,
        centerCodebook: Array.from(center.centroids),
        sigmaCodebook: Array.from(sigma.centroids, v => Math.exp(v))
    };
};

/**
 * Vector-quantizes the higher-order spherical harmonics into a palette shared
 * by every group: k-means over the per-splat coefficient vectors, then the
 * centroid palette itself quantized to a uint8 codebook.
 *
 * @param dataTable - Table holding the f_rest_* columns.
 * @param indices - Row order to emit.
 * @param shBands - Number of SH bands present (1-3).
 * @param iterations - k-means iteration count.
 * @param createDevice - Optional GPU device factory; k-means is the only GPU user.
 * @param bar - Progress bar, ticked twice.
 * @returns The centroid texture, its dimensions, the per-splat label plane, and
 * the shN metadata block.
 */
const packSh = async (
    dataTable: DataTable,
    indices: Uint32Array,
    shBands: number,
    iterations: number,
    createDevice: DeviceCreator | undefined,
    bar: ReturnType<typeof logger.bar>
) => {
    const numRows = indices.length;
    const shCoeffs = [0, 3, 8, 15][shBands];
    const shColumnNames = shNames.slice(0, shCoeffs * 3);
    const restCount = shColumnNames.length;

    // the clusterer takes one interleaved buffer, not columns
    const shCols = shColumnNames.map(name => dataTable.getColumnByName(name).data);
    const shRest = new Float32Array(numRows * restCount);
    for (let i = 0; i < numRows; ++i) {
        for (let j = 0; j < restCount; ++j) {
            shRest[i * restCount + j] = shCols[j][i];
        }
    }

    const paletteSize = Math.min(64, 2 ** Math.floor(Math.log2(numRows / 1024))) * 1024;

    // create the GPU device lazily — only SH clustering needs it
    const gpuDevice = createDevice ? await createDevice() : undefined;

    bar.tick();
    const { centroids, labels } = await kmeansInterleaved(shRest, numRows, restCount, paletteSize, iterations, gpuDevice);
    const numCentroids = centroids.length / restCount;

    bar.tick();
    // de-interleave the (small) centroid palette into columns for the quantizer
    const cbCols: { name: string, data: Float32Array }[] = [];
    for (let j = 0; j < restCount; ++j) {
        const col = new Float32Array(numCentroids);
        for (let i = 0; i < numCentroids; ++i) {
            col[i] = centroids[i * restCount + j];
        }
        cbCols.push({ name: shColumnNames[j], data: col });
    }
    const codebook = await runQuantize1dColumns(cbCols);
    const cbLabels = codebook.labels.map(c => c.data);

    // centroid texture: 64 palette entries per row, each shCoeffs texels wide
    const centroidsDims = { width: 64 * shCoeffs, height: Math.ceil(numCentroids / 64) };
    const centroidsBuf = new Uint8Array(centroidsDims.width * centroidsDims.height * 4);
    for (let i = 0; i < numCentroids; ++i) {
        for (let j = 0; j < shCoeffs; ++j) {
            centroidsBuf[i * shCoeffs * 4 + j * 4 + 0] = cbLabels[shCoeffs * 0 + j][i];
            centroidsBuf[i * shCoeffs * 4 + j * 4 + 1] = cbLabels[shCoeffs * 1 + j][i];
            centroidsBuf[i * shCoeffs * 4 + j * 4 + 2] = cbLabels[shCoeffs * 2 + j][i];
            centroidsBuf[i * shCoeffs * 4 + j * 4 + 3] = 0xff;
        }
    }

    const labelsPlane = new Uint8Array(numRows * 4);
    for (let i = 0; i < numRows; ++i) {
        const label = labels[indices[i]];
        labelsPlane[i * 4 + 0] = 0xff & label;
        labelsPlane[i * 4 + 1] = 0xff & (label >> 8);
        labelsPlane[i * 4 + 3] = 0xff;
    }

    return {
        centroidsBuf,
        centroidsDims,
        labelsPlane,
        shN: {
            count: numCentroids,
            bands: shBands,
            codebook: Array.from(codebook.centroids),
            files: ['shN_centroids.webp', 'shN_labels.webp']
        }
    };
};

type Split16Planes = ReturnType<typeof computeSplit16Planes>;
type QuantizedColumns = Awaited<ReturnType<typeof quantizeColumns>>;
type ShPack = Awaited<ReturnType<typeof packSh>>;
type TrbfPack = Awaited<ReturnType<typeof packTrbf>>;
type PendingEntry = { name: string, data: Promise<Uint8Array> };
type WrittenEntry = { name: string, data: Uint8Array };

/**
 * The `streams` block: which groups exist and where the two streaming markers
 * fall. The byte offsets start at 0 and are filled in by resolveStreamOffsets.
 */
type Streams = {
    persistent: string | null;
    segments: (string | null)[];
    sh_deferred: boolean;
    reveal_bytes: number;
    geometry_bytes: number;
};

/**
 * The stored size of one archive entry: its local header, its name, its data.
 *
 * @param name - Entry name.
 * @param data - Entry contents.
 * @returns The number of bytes the entry occupies in the archive.
 */
const entrySize = (name: string, data: Uint8Array) => {
    return ZIP_LOCAL_HEADER_SIZE + name.length + data.length;
};

/**
 * Lays out the archive: slices every plane per group, starts the WebP encodes,
 * and records which entry each streaming marker falls on.
 *
 * With segments this is the streamed layout, SH deferred behind geometry:
 * `[meta | persistent/* | seg_NNN/* | shN_centroids | *\/shN_labels]`. A player
 * starts DC-only playback once the first segment has landed and layers the
 * view-dependent SH in as the trailing entries arrive. Without segments it is
 * one whole-clip texture set at the archive root and there are no markers.
 *
 * @param planes - The geometry planes, as [entry name, whole-clip plane] pairs.
 * @param sh - Packed spherical harmonics, or null when the asset has none.
 * @param segments - The segment table, or null for a monolithic archive.
 * @param numRows - Total splat count.
 * @returns The pending entries in archive order, the index of the last entry
 * covered by each marker, and the `streams` block (null when unsegmented).
 */
const buildArchive = (
    planes: [string, Uint8Array][],
    sh: ShPack | null,
    segments: Segments | null,
    numRows: number
) => {
    // encode a plane's [a, b) slice into its own near-square texture. Padding
    // texels beyond the slice stay zero — never read, and they compress best.
    //
    // Encoding runs on the worker pool: a segmented clip emits one texture set
    // per group, which is 120-280 encodes for the capture fixtures, and they
    // are independent of one another. `buf` is freshly allocated here, so
    // handing its buffer to the worker as a transfer detaches nothing shared.
    const encodeSlice = (plane: Uint8Array, a: number, b: number) => {
        // same 4-aligned near-square dimensions as the SOG writer — the spec
        // requires only splat i at texel i, and permits the roundup
        const { width, height } = texDims(b - a);
        const buf = new Uint8Array(width * height * 4);
        buf.set(plane.subarray(a * 4, b * 4));
        return runEncodeWebp(buf, width, height);
    };

    // the geometry textures of one index range, in archive order
    const groupTextures = (prefix: string, a: number, b: number) => {
        return planes.map(([name, plane]) => ({
            name: prefix ? `${prefix}/${name}` : name,
            data: encodeSlice(plane, a, b)
        }));
    };

    const pending: PendingEntry[] = [];
    let revealThrough = -1;
    let geometryThrough = -1;
    let streams: Streams | null = null;

    if (segments) {
        const persistentEnd = segments.persistent[1];
        const prefixes = segments.list.map((segment, i) => (segment.range[1] > segment.range[0] ? `seg_${String(i).padStart(3, '0')}` : null));
        const revealPrefix = prefixes.find(prefix => prefix !== null) ?? null;

        const groups: [string, number, number][] = [];
        if (persistentEnd > 0) {
            groups.push(['persistent', 0, persistentEnd]);
        }
        segments.list.forEach((segment, i) => {
            if (prefixes[i]) {
                groups.push([prefixes[i], segment.range[0], segment.range[1]]);
            }
        });

        const labelEntries: PendingEntry[] = [];

        for (const [prefix, a, b] of groups) {
            pending.push(...groupTextures(prefix, a, b));

            if (sh) {
                labelEntries.push({
                    name: `${prefix}/shN_labels.webp`,
                    data: encodeSlice(sh.labelsPlane, a, b)
                });
            }

            if (prefix === 'persistent' || prefix === revealPrefix) {
                revealThrough = pending.length - 1;
            }
        }

        geometryThrough = pending.length - 1;

        if (sh) {
            pending.push({
                name: 'shN_centroids.webp',
                data: runEncodeWebp(sh.centroidsBuf, sh.centroidsDims.width, sh.centroidsDims.height)
            });
        }
        pending.push(...labelEntries);

        streams = {
            persistent: persistentEnd > 0 ? 'persistent' : null,
            segments: prefixes,
            sh_deferred: labelEntries.length > 0,
            reveal_bytes: 0,
            geometry_bytes: 0
        };
    } else {
        pending.push(...groupTextures('', 0, numRows));

        if (sh) {
            pending.push({
                name: 'shN_centroids.webp',
                data: runEncodeWebp(sh.centroidsBuf, sh.centroidsDims.width, sh.centroidsDims.height)
            });
            pending.push({ name: 'shN_labels.webp', data: encodeSlice(sh.labelsPlane, 0, numRows) });
        }
    }

    return { pending, revealThrough, geometryThrough, streams };
};

/**
 * Assembles meta.json.
 *
 * @param parts - Everything the manifest reports on.
 * @param parts.numRows - Total splat count.
 * @param parts.timeMin - Clip start, in seconds.
 * @param parts.timeMax - Clip end, in seconds.
 * @param parts.fps - Playback rate.
 * @param parts.cov2dScale - Optional 2D covariance scale pair.
 * @param parts.means - Packed positions.
 * @param parts.scales - Quantized log scales.
 * @param parts.colors - Quantized SH DC coefficients.
 * @param parts.motion - Packed velocities.
 * @param parts.accel - Packed accelerations, or null at degree 1.
 * @param parts.trbf - Packed temporal centre and sigma.
 * @param parts.sh - Packed higher-order harmonics, or null when absent.
 * @param parts.segments - The segment table, or null for a monolithic archive.
 * @param parts.streams - The stream block, or null for a monolithic archive.
 * @returns The manifest object, ready to serialize.
 */
const buildSogstMeta = (parts: {
    numRows: number,
    timeMin: number,
    timeMax: number,
    fps: number,
    cov2dScale?: SogstClip['cov2dScale'],
    means: Split16Planes,
    scales: QuantizedColumns,
    colors: QuantizedColumns,
    motion: Split16Planes,
    accel: Split16Planes | null,
    trbf: TrbfPack,
    sh: ShPack | null,
    segments: Segments | null,
    streams: Streams | null
}): any => {
    const { numRows, timeMin, timeMax, fps, cov2dScale, means, scales, colors, motion, accel, trbf, sh, segments, streams } = parts;

    return {
        // container version 1. The development-era formats this grew out of were
        // never released, so they were deleted rather than deprecated: there is
        // no version 2 or 3 to be compatible with. `format` is required — a
        // reader rejects an archive without it rather than guessing.
        version: 1,
        format: 'sogst',
        asset: {
            generator: `splat-transform v${version}`
        },
        count: numRows,
        time: {
            min: timeMin,
            max: timeMax,
            fps: fps
        },
        means: {
            mins: means.mins,
            maxs: means.maxs,
            files: ['means_l.webp', 'means_u.webp']
        },
        scales: {
            codebook: Array.from(scales.centroids),
            files: ['scales.webp']
        },
        quats: {
            files: ['quats.webp']
        },
        sh0: {
            codebook: Array.from(colors.centroids),
            files: ['sh0.webp']
        },
        ...(sh ? { shN: sh.shN } : {}),
        motion: {
            degree: accel ? 2 : 1,
            mins: motion.mins,
            maxs: motion.maxs,
            files: ['motion_l.webp', 'motion_u.webp']
        },
        ...(accel ? {
            accel: {
                mins: accel.mins,
                maxs: accel.maxs,
                files: ['accel_l.webp', 'accel_u.webp']
            }
        } : {}),
        trbf: {
            center: { codebook: trbf.centerCodebook },
            sigma: { codebook: trbf.sigmaCodebook },
            files: ['trbf.webp']
        },
        ...(cov2dScale ? { cov2d_scale: cov2dScale } : {}),
        ...(segments ? { segments } : {}),
        ...(streams ? { streams } : {})
    };
};

/**
 * Fills in `streams.reveal_bytes` / `geometry_bytes` and serializes the
 * manifest.
 *
 * The offsets are byte positions inside the very JSON whose length they change,
 * so this iterates to a fixed point. `streams` is the same object the manifest
 * holds, so mutating it updates the manifest in place. reveal_bytes ends the
 * last entry needed for first paint; geometry_bytes ends the last geometry
 * entry, which a player uses with measured bandwidth to decide when gap-free
 * playback becomes possible.
 *
 * @param meta - The manifest, holding `streams` by reference.
 * @param streams - The stream block to fill in, or null when unsegmented.
 * @param entries - The archive entries following meta.json, in order.
 * @param revealThrough - Index of the last entry covered by reveal_bytes.
 * @param geometryThrough - Index of the last entry covered by geometry_bytes.
 * @returns The serialized manifest.
 */
const resolveStreamOffsets = (
    meta: any,
    streams: Streams | null,
    entries: WrittenEntry[],
    revealThrough: number,
    geometryThrough: number
): Uint8Array => {
    const textEncoder = new TextEncoder();
    let metaJson = textEncoder.encode(JSON.stringify(meta));

    if (streams) {
        for (let pass = 0; pass < 6; ++pass) {
            const previous = [streams.reveal_bytes, streams.geometry_bytes];

            let pos = entrySize('meta.json', metaJson);
            for (let i = 0; i < entries.length; ++i) {
                pos += entrySize(entries[i].name, entries[i].data);
                if (i === revealThrough) streams.reveal_bytes = pos;
                if (i === geometryThrough) streams.geometry_bytes = pos;
            }

            metaJson = textEncoder.encode(JSON.stringify(meta));

            if (previous[0] === streams.reveal_bytes && previous[1] === streams.geometry_bytes) {
                break;
            }
        }
    }

    return metaJson;
};

/**
 * Verifies the analytic offsets against the real layout: the entry following
 * each marker must start exactly at the stored byte offset. This is the only
 * thing standing between an off-by-one and a player revealing a half-loaded
 * frame.
 *
 * When a marker falls on the last entry there is no following entry — the
 * offset is the end of the data section, where the central directory begins.
 * That is the ordinary case for an asset with no SH, so skipping the check
 * there would leave every static asset unverified.
 *
 * @param streams - The stream block that was written, or null when unsegmented.
 * @param entries - The archive entries following meta.json, in order.
 * @param archive - Every entry written, meta.json first.
 * @param headerOffsets - Local header offset of each archive entry, as written.
 * @param revealThrough - Index of the last entry covered by reveal_bytes.
 * @param geometryThrough - Index of the last entry covered by geometry_bytes.
 * @throws Error naming the offset that disagrees and what it landed on.
 */
const verifyStreamOffsets = (
    streams: Streams | null,
    entries: WrittenEntry[],
    archive: StoredZipEntry[],
    headerOffsets: number[],
    revealThrough: number,
    geometryThrough: number
) => {
    if (!streams) {
        return;
    }

    const last = archive[archive.length - 1];
    const endOfData = headerOffsets[headerOffsets.length - 1] + entrySize(last.name, last.data);

    const checks: ['reveal_bytes' | 'geometry_bytes', number][] = [
        ['reveal_bytes', revealThrough],
        ['geometry_bytes', geometryThrough]
    ];

    for (const [key, index] of checks) {
        if (index < 0) {
            continue;
        }

        // headerOffsets[0] is meta.json, so entries[n] is at n + 1
        const follows = index + 1 < entries.length;
        const actual = follows ? headerOffsets[index + 2] : endOfData;
        if (actual !== streams[key]) {
            const what = follows ? `the local header of '${entries[index + 1].name}'` : 'the central directory';
            throw new Error(`writeSogst: ${key} ${streams[key]} does not match ${actual}, ${what}`);
        }
    }
};

/**
 * Writes Gaussian splat data to the SOGST (SOG spacetime) format.
 *
 * SOGST is a SOG container extended to spacetime: the same WebP texture groups
 * and k-means codebooks, plus a per-splat linear velocity, temporal centre and
 * temporal standard deviation, so each Gaussian moves and fades:
 *
 * ```
 * mean(t)  = xyz + v·(t − t_center)  [+ a·(t − t_center)²  when degree 2]
 * alpha(t) = sigmoid(opacity) · exp(−0.5·((t − t_center)/t_sigma)²)
 * ```
 *
 * The temporal factor is deliberately unnormalised — alpha is a weight, not a
 * density.
 *
 * With segmentation on (the default) the archive uses a streamed layout: one
 * texture set per group, written in play order, with the higher-order SH
 * deferred to the tail so geometry arrives first.
 *
 * @param options - Options including filename, data and compression settings.
 * @param fs - File system for writing the output file.
 * @ignore
 */
const writeSogst = async (options: WriteSogstOptions, fs: FileSystem) => {
    const { filename, dataTable, iterations, createDevice } = options;
    const segmentDuration = options.segmentDuration ?? DEFAULT_SEGMENT_DURATION;
    const kSigma = options.kSigma ?? DEFAULT_K_SIGMA;
    const persistentSpanMult = options.persistentSpanMult ?? DEFAULT_PERSISTENT_SPAN_MULT;

    const numRows = dataTable.numRows;

    // clip scalars have no defaults — an assumed frame rate renders perfectly
    // and plays at the wrong speed
    const { timeMin, timeMax, fps, motionDegree, cov2dScale } = requireSogstClip(options.clip ?? {});

    const { hasAccel, shBands, tSigmaColumn } = validateSogstInput(dataTable, motionDegree);

    const totalSteps = 8 + (hasAccel ? 1 : 0) + (shBands > 0 ? 2 : 0);
    const bar = logger.bar('encoding', totalSteps);

    const { indices, segments } = computeOrder(dataTable, timeMin, timeMax, segmentDuration, kSigma, persistentSpanMult);

    // Every plane below is RGBA in `indices` order, so texel i holds row
    // indices[i]. A group is then just a slice — and every plane stays in sync
    // with the others because they all share this one permutation.

    bar.tick();
    const means = computeSplit16Planes(dataTable, ['x', 'y', 'z'], indices);

    bar.tick();
    const quatsPlane = packQuats(dataTable, indices);

    bar.tick();
    const scales = await quantizeColumns(dataTable, ['scale_0', 'scale_1', 'scale_2']);
    const scalesPlane = packLabels(scales.labels, indices, 255);

    bar.tick();
    const colors = await quantizeColumns(dataTable, ['f_dc_0', 'f_dc_1', 'f_dc_2']);
    const sh0Plane = packLabels(colors.labels, indices, 255);
    const opacity = dataTable.getColumnByName('opacity').data;
    for (let i = 0; i < numRows; ++i) {
        sh0Plane[i * 4 + 3] = Math.max(0, Math.min(255, Math.round(sigmoid(opacity[indices[i]]) * 255)));
    }

    bar.tick();
    const motion = computeSplit16Planes(dataTable, motionNames, indices);

    let accel = null;
    if (hasAccel) {
        bar.tick();
        accel = computeSplit16Planes(dataTable, accelNames, indices);
    }

    bar.tick();
    const trbf = await packTrbf(dataTable, indices, tSigmaColumn);

    // higher-order spherical harmonics: vector-quantized, shared across groups
    const sh = shBands > 0 ? await packSh(dataTable, indices, shBands, iterations, createDevice, bar) : null;

    // -- texture encoding ---------------------------------------------------

    bar.tick();

    const planes: [string, Uint8Array][] = [
        ['means_l.webp', means.lo],
        ['means_u.webp', means.hi],
        ['quats.webp', quatsPlane],
        ['scales.webp', scalesPlane],
        ['sh0.webp', sh0Plane],
        ['motion_l.webp', motion.lo],
        ['motion_u.webp', motion.hi],
        ['trbf.webp', trbf.plane]
    ];

    if (accel) {
        planes.push(['accel_l.webp', accel.lo], ['accel_u.webp', accel.hi]);
    }

    const { pending, revealThrough, geometryThrough, streams } = buildArchive(planes, sh, segments, numRows);

    bar.tick();

    // Archive order is `pending` order; awaiting as a batch keeps it while
    // letting the pool encode out of order.
    const encoded = await Promise.all(pending.map(entry => entry.data));
    const entries = pending.map((entry, i) => ({ name: entry.name, data: encoded[i] }));

    // -- meta.json ----------------------------------------------------------

    const meta = buildSogstMeta({
        numRows,
        timeMin,
        timeMax,
        fps,
        cov2dScale,
        means,
        scales,
        colors,
        motion,
        accel,
        trbf,
        sh,
        segments,
        streams
    });

    const metaJson = resolveStreamOffsets(meta, streams, entries, revealThrough, geometryThrough);

    // -- write the archive --------------------------------------------------

    const archive: StoredZipEntry[] = [{ name: 'meta.json', data: metaJson }, ...entries];

    const outputWriter = await fs.createWriter(filename);
    const headerOffsets = await writeStoredZip(outputWriter, archive);
    await outputWriter.close();

    bar.end();
    logWrittenFile(basename(filename), outputWriter.bytesWritten);

    verifyStreamOffsets(streams, entries, archive, headerOffsets, revealThrough, geometryThrough);
};

export { writeSogst, computeOrder, type WriteSogstOptions, type Segment, type Segments };
