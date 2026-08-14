import { basename } from 'pathe';

import { computeSplit16Planes, texDims } from './sog-common';
import { requireSogstClip, type SogstClip } from './sogst-clip';
import { logWrittenFile } from './utils';
import { Column, DataTable, sortMortonOrder } from '../data-table';
import { type FileSystem, writeStoredZip, type StoredZipEntry } from '../io/write';
import { kmeans, quantize1d } from '../spatial';
import type { DeviceCreator } from '../types';
import { logger, sigmoid, WebPCodec } from '../utils';
import { version } from '../version';

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

// Entries carry no extra fields and no data descriptor, so a local header is
// exactly this plus the filename. The streaming offsets depend on it.
const ZIP_LOCAL_HEADER_SIZE = 30;

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

const motionNames = ['vx', 'vy', 'vz'];
const accelNames = ['ax', 'ay', 'az'];
const trbfNames = ['t_center', 't_sigma'];

let webPCodec: WebPCodec;

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
 * Packs up to three uint8 label columns into an RGBA plane in `indices` order.
 *
 * @param labels - Table of Uint8Array label columns.
 * @param indices - Row order to emit.
 * @param alpha - Value for the alpha channel.
 * @returns The packed plane.
 */
const packLabels = (labels: DataTable, indices: Uint32Array, alpha: number): Uint8Array => {
    const numRows = indices.length;
    const columns = labels.columns.map(c => c.data);
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
    for (let i = 0; i < numRows; ++i) {
        if (!(tSigmaColumn[i] > 0)) {
            throw new Error(`sogst output requires t_sigma > 0 for every splat; row ${i} has ${tSigmaColumn[i]}.`);
        }
    }

    const totalSteps = 8 + (hasAccel ? 1 : 0) + (shBands > 0 ? 2 : 0);
    const bar = logger.bar('encoding', totalSteps);

    const { indices, segments } = computeOrder(dataTable, timeMin, timeMax, segmentDuration, kSigma, persistentSpanMult);

    // Every plane below is RGBA in `indices` order, so texel i holds row
    // indices[i]. A group is then just a slice — and every plane stays in sync
    // with the others because they all share this one permutation.

    bar.tick();
    const means = computeSplit16Planes(dataTable, ['x', 'y', 'z'], indices);

    bar.tick();
    const quatsPlane = (() => {
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
    })();

    bar.tick();
    const scales = quantize1d(
        new DataTable(['scale_0', 'scale_1', 'scale_2'].map(name => dataTable.getColumnByName(name)))
    );
    const scalesPlane = packLabels(scales.labels, indices, 255);

    bar.tick();
    const colors = quantize1d(
        new DataTable(['f_dc_0', 'f_dc_1', 'f_dc_2'].map(name => dataTable.getColumnByName(name)))
    );
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
    const trbfCenter = quantize1d(new DataTable([dataTable.getColumnByName('t_center')]));

    // sigma is heavy-tailed, so cluster in the log domain: a linear codebook
    // spends almost all 256 entries on the tail. The codebook is mapped back to
    // linear space, so decoders are unaffected.
    const logSigma = new Float32Array(numRows);
    for (let i = 0; i < numRows; ++i) {
        logSigma[i] = Math.log(tSigmaColumn[i]);
    }
    const trbfSigma = quantize1d(new DataTable([new Column('t_sigma', logSigma)]));

    const trbfPlane = new Uint8Array(numRows * 4);
    const centerLabels = trbfCenter.labels.getColumn(0).data;
    const sigmaLabels = trbfSigma.labels.getColumn(0).data;
    for (let i = 0; i < numRows; ++i) {
        const idx = indices[i];
        trbfPlane[i * 4 + 0] = centerLabels[idx];
        trbfPlane[i * 4 + 1] = sigmaLabels[idx];
        trbfPlane[i * 4 + 3] = 255;
    }

    // higher-order spherical harmonics: vector-quantized, shared across groups
    let shN = null;
    let shCentroidsBuf: Uint8Array = null;
    let shCentroidsDims: { width: number; height: number } = null;
    let shLabelsPlane: Uint8Array = null;

    if (shBands > 0) {
        const shCoeffs = [0, 3, 8, 15][shBands];
        const shColumnNames = shNames.slice(0, shCoeffs * 3);
        const shDataTable = new DataTable(shColumnNames.map(name => dataTable.getColumnByName(name)));

        const paletteSize = Math.min(64, 2 ** Math.floor(Math.log2(numRows / 1024))) * 1024;

        // create the GPU device lazily — only SH clustering needs it
        const gpuDevice = createDevice ? await createDevice() : undefined;

        bar.tick();
        const { centroids, labels } = await kmeans(shDataTable, paletteSize, iterations, gpuDevice);

        bar.tick();
        const codebook = quantize1d(centroids);

        // centroid texture: 64 palette entries per row, each shCoeffs texels wide
        shCentroidsDims = { width: 64 * shCoeffs, height: Math.ceil(centroids.numRows / 64) };
        shCentroidsBuf = new Uint8Array(shCentroidsDims.width * shCentroidsDims.height * 4);
        const centroidsRow: any = {};
        for (let i = 0; i < centroids.numRows; ++i) {
            codebook.labels.getRow(i, centroidsRow);

            for (let j = 0; j < shCoeffs; ++j) {
                shCentroidsBuf[i * shCoeffs * 4 + j * 4 + 0] = centroidsRow[shColumnNames[shCoeffs * 0 + j]];
                shCentroidsBuf[i * shCoeffs * 4 + j * 4 + 1] = centroidsRow[shColumnNames[shCoeffs * 1 + j]];
                shCentroidsBuf[i * shCoeffs * 4 + j * 4 + 2] = centroidsRow[shColumnNames[shCoeffs * 2 + j]];
                shCentroidsBuf[i * shCoeffs * 4 + j * 4 + 3] = 0xff;
            }
        }

        shLabelsPlane = new Uint8Array(numRows * 4);
        for (let i = 0; i < numRows; ++i) {
            const label = labels[indices[i]];
            shLabelsPlane[i * 4 + 0] = 0xff & label;
            shLabelsPlane[i * 4 + 1] = 0xff & (label >> 8);
            shLabelsPlane[i * 4 + 3] = 0xff;
        }

        shN = {
            count: paletteSize,
            bands: shBands,
            codebook: Array.from(codebook.centroids.getColumn(0).data),
            files: ['shN_centroids.webp', 'shN_labels.webp']
        };
    }

    // -- texture encoding ---------------------------------------------------

    bar.tick();

    if (!webPCodec) {
        webPCodec = await WebPCodec.create();
    }

    // encode a plane's [a, b) slice into its own near-square texture. Padding
    // texels beyond the slice stay zero — never read, and they compress best.
    const encodeSlice = (plane: Uint8Array, a: number, b: number) => {
        // same 4-aligned near-square dimensions as the SOG writer — the spec
        // requires only splat i at texel i, and permits the roundup
        const { width, height } = texDims(b - a);
        const buf = new Uint8Array(width * height * 4);
        buf.set(plane.subarray(a * 4, b * 4));
        return webPCodec.encodeLosslessRGBA(buf, width, height);
    };

    // the geometry textures of one index range, in archive order
    const groupTextures = (prefix: string, a: number, b: number) => {
        const planes: [string, Uint8Array][] = [
            ['means_l.webp', means.lo],
            ['means_u.webp', means.hi],
            ['quats.webp', quatsPlane],
            ['scales.webp', scalesPlane],
            ['sh0.webp', sh0Plane],
            ['motion_l.webp', motion.lo],
            ['motion_u.webp', motion.hi],
            ['trbf.webp', trbfPlane]
        ];

        if (accel) {
            planes.push(['accel_l.webp', accel.lo], ['accel_u.webp', accel.hi]);
        }

        return planes.map(([name, plane]) => ({
            name: prefix ? `${prefix}/${name}` : name,
            data: encodeSlice(plane, a, b)
        }));
    };

    const entries: { name: string; data: Uint8Array }[] = [];
    let revealThrough = -1;
    let geometryThrough = -1;
    let streams = null;

    if (segments) {
        // Streamed layout, SH deferred behind geometry:
        //   [meta | persistent/* | seg_NNN/* | shN_centroids | */shN_labels]
        // A player starts DC-only playback once the first segment has landed
        // and layers the view-dependent SH in as the trailing entries arrive.
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

        const labelEntries: { name: string; data: Uint8Array }[] = [];

        for (const [prefix, a, b] of groups) {
            entries.push(...groupTextures(prefix, a, b));

            if (shLabelsPlane) {
                labelEntries.push({
                    name: `${prefix}/shN_labels.webp`,
                    data: encodeSlice(shLabelsPlane, a, b)
                });
            }

            if (prefix === 'persistent' || prefix === revealPrefix) {
                revealThrough = entries.length - 1;
            }
        }

        geometryThrough = entries.length - 1;

        if (shCentroidsBuf) {
            entries.push({
                name: 'shN_centroids.webp',
                data: webPCodec.encodeLosslessRGBA(shCentroidsBuf, shCentroidsDims.width, shCentroidsDims.height)
            });
        }
        entries.push(...labelEntries);

        streams = {
            persistent: persistentEnd > 0 ? 'persistent' : null,
            segments: prefixes,
            sh_deferred: labelEntries.length > 0,
            reveal_bytes: 0,
            geometry_bytes: 0
        };
    } else {
        // monolithic layout: one whole-clip texture set at the archive root
        entries.push(...groupTextures('', 0, numRows));

        if (shCentroidsBuf) {
            entries.push({
                name: 'shN_centroids.webp',
                data: webPCodec.encodeLosslessRGBA(shCentroidsBuf, shCentroidsDims.width, shCentroidsDims.height)
            });
            entries.push({ name: 'shN_labels.webp', data: encodeSlice(shLabelsPlane, 0, numRows) });
        }
    }

    bar.tick();

    // -- meta.json ----------------------------------------------------------

    const meta: any = {
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
            codebook: Array.from(scales.centroids.getColumn(0).data),
            files: ['scales.webp']
        },
        quats: {
            files: ['quats.webp']
        },
        sh0: {
            codebook: Array.from(colors.centroids.getColumn(0).data),
            files: ['sh0.webp']
        },
        ...(shN ? { shN } : {}),
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
            center: { codebook: Array.from(trbfCenter.centroids.getColumn(0).data) },
            sigma: { codebook: Array.from(trbfSigma.centroids.getColumn(0).data, v => Math.exp(v)) },
            files: ['trbf.webp']
        },
        ...(cov2dScale ? { cov2d_scale: cov2dScale } : {}),
        ...(segments ? { segments } : {}),
        ...(streams ? { streams } : {})
    };

    const textEncoder = new TextEncoder();
    const entrySize = (name: string, data: Uint8Array) => {
        return ZIP_LOCAL_HEADER_SIZE + name.length + data.length;
    };

    // reveal_bytes / geometry_bytes are byte offsets inside the very JSON whose
    // length they change, so iterate to a fixed point. reveal_bytes ends the
    // last entry needed for first paint; geometry_bytes ends the last geometry
    // entry, which a player uses with measured bandwidth to decide when
    // gap-free playback becomes possible.
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

    // -- write the archive --------------------------------------------------

    const archive: StoredZipEntry[] = [{ name: 'meta.json', data: metaJson }, ...entries];

    const outputWriter = await fs.createWriter(filename);
    const headerOffsets = await writeStoredZip(outputWriter, archive);
    await outputWriter.close();

    bar.end();
    logWrittenFile(basename(filename), outputWriter.bytesWritten);

    // Verify the analytic offsets against the real layout: the entry following
    // each marker must start exactly at the stored byte offset. This is the
    // only thing standing between an off-by-one and a player revealing a
    // half-loaded frame.
    //
    // When a marker falls on the last entry there is no following entry — the
    // offset is the end of the data section, where the central directory
    // begins. That is the ordinary case for an asset with no SH, so skipping
    // the check there would leave every static asset unverified.
    if (streams) {
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
    }
};

export { writeSogst, computeOrder, type WriteSogstOptions, type Segment, type Segments };
