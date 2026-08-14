/**
 * SOGST (SOG spacetime) writer tests.
 *
 * There is no .sogst reader in this repo, so these tests decode the archive
 * independently — a minimal ZIP_STORED parser plus a field decoder written
 * straight from the format spec. That is deliberate: a decoder that shared code
 * with the writer would agree with it about any mistake they both made.
 */

import assert from 'node:assert';
import { describe, it, before } from 'node:test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
    MemoryFileSystem,
    WebPCodec,
    writeSogst,
    parseSogstComments,
    computeSummary
} from '../src/lib/index.js';

import { createTestDataTable, addSpacetimeColumns } from './helpers/test-utils.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

const CLIP = { timeMin: 0, timeMax: 1, fps: 24 };

const SQRT2 = Math.sqrt(2);

// ---------------------------------------------------------------------------
// Independent archive reader
// ---------------------------------------------------------------------------

/**
 * Parses a ZIP archive by walking its local file headers, asserting the
 * structural guarantees the format depends on: every entry STORED, no extra
 * fields, and no data descriptor — which together are what make the local
 * header exactly 30 + len(name) bytes and the streaming offsets computable.
 */
const readStoredZip = (bytes) => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const decoder = new TextDecoder();
    const entries = [];

    let pos = 0;
    while (pos + 4 <= bytes.length && view.getUint32(pos, true) === 0x04034b50) {
        const flags = view.getUint16(pos + 6, true);
        const method = view.getUint16(pos + 8, true);
        const size = view.getUint32(pos + 18, true);
        const nameLen = view.getUint16(pos + 26, true);
        const extraLen = view.getUint16(pos + 28, true);
        const name = decoder.decode(bytes.subarray(pos + 30, pos + 30 + nameLen));

        assert.strictEqual(method, 0, `entry '${name}' must be STORED`);
        assert.strictEqual(extraLen, 0, `entry '${name}' must have no extra field`);
        assert.strictEqual(flags & 0x8, 0, `entry '${name}' must not use a data descriptor`);

        const dataStart = pos + 30 + nameLen;
        entries.push({ name, headerOffset: pos, data: bytes.subarray(dataStart, dataStart + size) });
        pos = dataStart + size;
    }

    assert(entries.length > 0, 'archive should contain at least one entry');
    return entries;
};

let codec;

const decodeTexture = (blob) => {
    const { rgba, width, height } = codec.decodeRGBA(blob);
    return { rgba, width, height };
};

// ---------------------------------------------------------------------------
// Independent field decoder (spec §2.2, §4)
// ---------------------------------------------------------------------------

const unsplit16 = (lo, hi, mins, maxs, i, c) => {
    const q = (hi[i * 4 + c] * 256 + lo[i * 4 + c]) / 65535;
    const t = mins[c] + q * (maxs[c] - mins[c]);
    return Math.sign(t) * (Math.exp(Math.abs(t)) - 1);
};

const unpackQuat = (texels, i) => {
    const mode = texels[i * 4 + 3];
    const dropped = mode - 252;
    const keep = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]][dropped];

    const comps = [0, 0, 0, 0];
    for (let j = 0; j < 3; ++j) {
        comps[keep[j]] = (texels[i * 4 + j] / 255 - 0.5) * SQRT2;
    }
    const sumSq = comps.reduce((t, v) => t + v * v, 0);
    comps[dropped] = Math.sqrt(Math.max(0, 1 - sumSq));

    return comps; // (w, x, y, z)
};

/**
 * Decodes a whole .sogst archive back into per-splat fields, reassembling the
 * streamed per-group textures into full-length planes at their index ranges.
 */
const decodeSogst = (bytes) => {
    const entries = readStoredZip(bytes);

    assert.strictEqual(entries[0].name, 'meta.json', 'meta.json must be the first entry');
    const meta = JSON.parse(new TextDecoder().decode(entries[0].data));

    const byName = new Map(entries.map(e => [e.name, e]));
    const count = meta.count;

    // gather the group index ranges
    const groups = [];
    if (meta.streams) {
        if (meta.streams.persistent) {
            groups.push([meta.streams.persistent, meta.segments.persistent]);
        }
        meta.streams.segments.forEach((prefix, i) => {
            if (prefix) groups.push([prefix, meta.segments.list[i].range]);
        });
    } else {
        groups.push([null, [0, count]]);
    }

    // reassemble each texture basename into one full-length RGBA plane
    const basenames = new Set();
    for (const { name } of entries) {
        if (!name.endsWith('.webp') || name === 'shN_centroids.webp') continue;
        basenames.add(name.includes('/') ? name.split('/')[1] : name);
    }

    const planes = {};
    const groupDims = new Map();

    for (const basename of basenames) {
        const plane = new Uint8Array(count * 4);
        for (const [prefix, [a, b]] of groups) {
            const entry = byName.get(prefix ? `${prefix}/${basename}` : basename);
            assert(entry, `missing archive entry for '${basename}' in group '${prefix}'`);

            const { rgba, width, height } = decodeTexture(entry.data);
            const m = b - a;

            // spec §2.1: splat i lives at row-major texel i, so a texture only
            // has to be big enough; every texture in a group must agree
            assert(width * height >= m, `${basename} is too small for group '${prefix}'`);
            assert(width >= Math.floor(Math.sqrt(m)), `${basename} should be near-square for group '${prefix}'`);

            const key = prefix ?? '';
            const dims = `${width}x${height}`;
            if (groupDims.has(key)) {
                assert.strictEqual(dims, groupDims.get(key), `${basename} dimensions differ from the rest of group '${prefix}'`);
            } else {
                groupDims.set(key, dims);
            }

            plane.set(rgba.subarray(0, m * 4), a * 4);
        }
        planes[basename] = plane;
    }

    const fields = {
        x: [], y: [], z: [],
        rot_0: [], rot_1: [], rot_2: [], rot_3: [],
        scale_0: [], scale_1: [], scale_2: [],
        f_dc_0: [], f_dc_1: [], f_dc_2: [],
        opacity: [],
        vx: [], vy: [], vz: [],
        t_center: [], t_sigma: []
    };
    if (meta.accel) {
        Object.assign(fields, { ax: [], ay: [], az: [] });
    }

    for (let i = 0; i < count; ++i) {
        const axes = ['x', 'y', 'z'];
        for (let c = 0; c < 3; ++c) {
            fields[axes[c]].push(unsplit16(planes['means_l.webp'], planes['means_u.webp'], meta.means.mins, meta.means.maxs, i, c));
            fields[`scale_${c}`].push(meta.scales.codebook[planes['scales.webp'][i * 4 + c]]);
            fields[`f_dc_${c}`].push(meta.sh0.codebook[planes['sh0.webp'][i * 4 + c]]);
            fields[['vx', 'vy', 'vz'][c]].push(unsplit16(planes['motion_l.webp'], planes['motion_u.webp'], meta.motion.mins, meta.motion.maxs, i, c));
            if (meta.accel) {
                fields[['ax', 'ay', 'az'][c]].push(unsplit16(planes['accel_l.webp'], planes['accel_u.webp'], meta.accel.mins, meta.accel.maxs, i, c));
            }
        }

        const quat = unpackQuat(planes['quats.webp'], i);
        for (let c = 0; c < 4; ++c) {
            fields[`rot_${c}`].push(quat[c]);
        }

        // sh0 alpha is already-activated linear opacity
        fields.opacity.push(planes['sh0.webp'][i * 4 + 3] / 255);

        fields.t_center.push(meta.trbf.center.codebook[planes['trbf.webp'][i * 4 + 0]]);
        fields.t_sigma.push(meta.trbf.sigma.codebook[planes['trbf.webp'][i * 4 + 1]]);
    }

    return { meta, entries, planes, fields };
};

// ---------------------------------------------------------------------------
// Source-to-decoded matching
// ---------------------------------------------------------------------------

/**
 * Matches every decoded splat to its source row by nearest position, and
 * asserts the matching is a bijection.
 *
 * Positions are unique in the fixture grid, so this recovers the writer's
 * permutation without reusing the writer's own ordering code. Every other field
 * is then checked against the *same* source row, which is what catches a texture
 * plane that has drifted out of sync with the others.
 */
const matchByPosition = (source, decoded, count) => {
    const sx = source.getColumnByName('x').data;
    const sy = source.getColumnByName('y').data;
    const sz = source.getColumnByName('z').data;

    const matches = new Array(count);
    const used = new Set();

    for (let i = 0; i < count; ++i) {
        let best = -1;
        let bestDist = Infinity;
        for (let j = 0; j < count; ++j) {
            const dx = decoded.x[i] - sx[j];
            const dy = decoded.y[i] - sy[j];
            const dz = decoded.z[i] - sz[j];
            const dist = dx * dx + dy * dy + dz * dz;
            if (dist < bestDist) {
                bestDist = dist;
                best = j;
            }
        }
        assert(bestDist < 1e-4, `decoded splat ${i} has no close source position (nearest ${Math.sqrt(bestDist)})`);
        assert(!used.has(best), `source row ${best} matched more than once — the ordering is not a permutation`);
        used.add(best);
        matches[i] = best;
    }

    assert.strictEqual(used.size, count, 'every source row should be present exactly once');
    return matches;
};

const assertField = (source, decoded, matches, name, tolerance) => {
    const src = source.getColumnByName(name).data;
    let worst = 0;
    let worstAt = -1;

    for (let i = 0; i < matches.length; ++i) {
        const error = Math.abs(decoded[name][i] - src[matches[i]]);
        if (error > worst) {
            worst = error;
            worstAt = i;
        }
    }

    assert(worst <= tolerance, `${name}: max error ${worst} exceeds ${tolerance} (at decoded index ${worstAt})`);
};

// ---------------------------------------------------------------------------

const makeFixture = (count = 64, options = {}) => {
    const dataTable = createTestDataTable(count, options);

    // vary the rotations — the default fixture is all-identity, which would let
    // a broken smallest-three encoding pass
    const rots = [
        [1, 0, 0, 0],
        [0.7071067811865476, 0.7071067811865476, 0, 0],
        [0, 0, 0, 1],
        [0.5, 0.5, 0.5, 0.5],
        [0, 0.8, 0.6, 0],
        [-0.6, 0, 0.8, 0]
    ];
    for (let i = 0; i < count; ++i) {
        const q = rots[i % rots.length];
        for (let c = 0; c < 4; ++c) {
            dataTable.getColumnByName(`rot_${c}`).data[i] = q[c];
        }
    }

    return addSpacetimeColumns(dataTable, options);
};

const writeToMemory = async (dataTable, options = {}) => {
    const fs = new MemoryFileSystem();
    await writeSogst({
        filename: 'test.sogst',
        dataTable,
        iterations: 2,
        clip: CLIP,
        ...options
    }, fs);

    const bytes = fs.results.get('test.sogst');
    assert(bytes && bytes.length > 0, '.sogst file should be written');
    return bytes;
};

describe('SOGST Format', () => {
    before(async () => {
        codec = await WebPCodec.create();
    });

    it('should round-trip spacetime fields through the streamed layout', async () => {
        const source = makeFixture(64);
        const { meta, fields } = decodeSogst(await writeToMemory(source));

        assert.strictEqual(meta.version, 1);
        assert.strictEqual(meta.format, 'sogst');
        assert.strictEqual(meta.count, source.numRows);
        assert.deepStrictEqual(meta.time, { min: 0, max: 1, fps: 24 });

        const matches = matchByPosition(source, fields, meta.count);

        for (const axis of ['x', 'y', 'z']) {
            assertField(source, fields, matches, axis, 1e-3);
        }
        for (const axis of ['vx', 'vy', 'vz']) {
            assertField(source, fields, matches, axis, 1e-4);
        }
        for (let c = 0; c < 3; ++c) {
            assertField(source, fields, matches, `scale_${c}`, 1e-3);
            assertField(source, fields, matches, `f_dc_${c}`, 1e-2);
        }
        for (let c = 0; c < 4; ++c) {
            assertField(source, fields, matches, `rot_${c}`, 0.01);
        }
        assertField(source, fields, matches, 't_center', 1e-3);
        assertField(source, fields, matches, 't_sigma', 1e-3);

        // opacity is stored already-activated, so compare against sigmoid(logit)
        const logits = source.getColumnByName('opacity').data;
        for (let i = 0; i < meta.count; ++i) {
            const expected = 1 / (1 + Math.exp(-logits[matches[i]]));
            assert(Math.abs(fields.opacity[i] - expected) <= 1 / 255, `opacity error at ${i}`);
        }
    });

    it('should omit accel and report degree 1 when the data has no acceleration', async () => {
        const { meta } = decodeSogst(await writeToMemory(makeFixture(32)));

        assert.strictEqual(meta.motion.degree, 1);
        assert.strictEqual(meta.accel, undefined, 'a degree-1 file must have no accel key');
    });

    it('should report degree 2 and round-trip accel when acceleration is present', async () => {
        const source = makeFixture(32, { includeAccel: true });
        const { meta, fields } = decodeSogst(await writeToMemory(source));

        assert.strictEqual(meta.motion.degree, 2);
        assert(meta.accel, 'a degree-2 file must have an accel key');
        assert.deepStrictEqual(meta.accel.files, ['accel_l.webp', 'accel_u.webp']);

        const matches = matchByPosition(source, fields, meta.count);
        for (const axis of ['ax', 'ay', 'az']) {
            assertField(source, fields, matches, axis, 1e-4);
        }
    });

    it('should produce a contiguous segment table covering every splat', async () => {
        const source = makeFixture(64);
        const { meta, fields } = decodeSogst(await writeToMemory(source));

        const { segments } = meta;
        assert(segments, 'segmentation is on by default');
        assert.strictEqual(segments.duration, 0.1);
        assert.strictEqual(segments.k_sigma, 3.8);
        assert.strictEqual(segments.persistent_span_mult, 3.0);

        const [p0, p1] = segments.persistent;
        assert.strictEqual(p0, 0, 'the persistent range starts at 0');
        assert(p1 > 0, 'the fixture should produce some persistent splats');

        // ranges are half-open, contiguous, and cover [P, count)
        let cursor = p1;
        for (const segment of segments.list) {
            const [first, last] = segment.range;
            assert.strictEqual(first, cursor, 'segment ranges must be contiguous');
            assert(last >= first, 'segment ranges must be half-open and non-negative');
            cursor = last;
        }
        assert.strictEqual(cursor, meta.count, 'segments must cover every remaining splat');

        // t0/t1 are the union of the members' active intervals, so every member
        // is fully inside its own segment's coverage
        const kSigma = segments.k_sigma;
        for (const segment of segments.list) {
            const [first, last] = segment.range;
            for (let i = first; i < last; ++i) {
                const lo = fields.t_center[i] - kSigma * fields.t_sigma[i];
                const hi = fields.t_center[i] + kSigma * fields.t_sigma[i];
                assert(lo >= segment.t0 - 1e-4 && hi <= segment.t1 + 1e-4, `splat ${i} falls outside its segment coverage`);
            }
        }

        // persistent splats are exactly those whose active span is long enough
        const threshold = segments.persistent_span_mult * segments.duration;
        for (let i = 0; i < meta.count; ++i) {
            const span = 2 * kSigma * fields.t_sigma[i];
            const isPersistent = i < p1;
            // the codebook perturbs sigma slightly, so ignore borderline cases
            if (Math.abs(span - threshold) > 1e-2) {
                assert.strictEqual(isPersistent, span > threshold, `splat ${i} is in the wrong group (span ${span})`);
            }
        }
    });

    it('should emit empty segments for a gap in the timeline', async () => {
        // no real capture produces an empty segment -- the subject is always
        // somewhere -- so this branch is otherwise only ever verified by reading
        // the code. persistentEvery is raised so the gap is not papered over by
        // long-lived splats spanning it.
        const source = makeFixture(64, { timeMax: 2, gap: [0.7, 1.3], persistentEvery: 1000 });
        const { meta, entries } = decodeSogst(await writeToMemory(source, { clip: { timeMin: 0, timeMax: 2, fps: 30 } }));

        const empty = meta.segments.list
        .map((segment, i) => ({ segment, i }))
        .filter(({ segment }) => segment.range[0] === segment.range[1]);

        assert(empty.length > 0, 'the gap should leave at least one segment empty');

        for (const { segment, i } of empty) {
            // an empty segment falls back to its nominal bucket window
            assert.strictEqual(segment.t0, meta.time.min + i * meta.segments.duration);
            assert.strictEqual(segment.t1, segment.t0 + meta.segments.duration);

            // and carries no stream prefix and no archive entries
            assert.strictEqual(meta.streams.segments[i], null, `segment ${i} should have a null prefix`);
            const prefix = `seg_${String(i).padStart(3, '0')}/`;
            assert(!entries.some(e => e.name.startsWith(prefix)), `segment ${i} should have no entries`);
        }

        // ranges stay contiguous straight through the empty ones
        let cursor = meta.segments.persistent[1];
        for (const segment of meta.segments.list) {
            assert.strictEqual(segment.range[0], cursor);
            cursor = segment.range[1];
        }
        assert.strictEqual(cursor, meta.count);
    });

    it('should give a populated segment its real extent, not its bucket bounds', async () => {
        // A populated segment's [t0, t1] is the union of its members' active
        // intervals, so a splat centred just inside a bucket has support
        // reaching back before the bucket edge. Where a long empty run precedes
        // such a segment, the empty segments carry nominal bucket bounds and the
        // populated one carries a real extent that starts EARLIER -- so
        // segments.list is not sorted by t0 on conforming files.
        //
        // Clamping the populated window to the bucket to restore monotonicity is
        // the tempting fix and it is wrong: it culls splats that are still on
        // screen, which pop. This locks in the asymmetry.
        const count = 64;
        const source = makeFixture(count, { timeMax: 2 });
        const tCenter = source.getColumnByName('t_center').data;
        const tSigma = source.getColumnByName('t_sigma').data;

        // an empty run over buckets 7..12, then a segment-13 population whose
        // sigma sits just under the persistent threshold (3.0 * 0.1 / 2 / 3.8)
        // so it stays bucketed while reaching well back past 1.2
        for (let i = 0; i < count; ++i) {
            const preGap = i < count / 2;
            tCenter[i] = preGap ? 0.05 + (i / count) * 1.2 : 1.301 + ((i - count / 2) / count) * 1.2;
            tSigma[i] = preGap ? 0.008 : 0.039;
        }

        const { meta } = decodeSogst(await writeToMemory(source, { clip: { timeMin: 0, timeMax: 2, fps: 30 } }));
        const { list, duration, k_sigma: kSigma } = meta.segments;

        const populated = list[13];
        assert(populated.range[1] > populated.range[0], 'segment 13 should be populated');
        assert.strictEqual(list[12].range[0], list[12].range[1], 'segment 12 should be empty');

        // the empty predecessor uses its nominal window
        assert.strictEqual(list[12].t0, 12 * duration);

        // and the populated segment reaches back before it
        assert(
            populated.t0 < list[12].t0,
            `segment 13 t0 (${populated.t0}) should precede empty segment 12 t0 (${list[12].t0}) -- ` +
            'a populated window must not be clamped to its bucket'
        );

        // the extent is exactly the union of the members' active intervals
        let lo = Infinity;
        let hi = -Infinity;
        for (let i = 0; i < count; ++i) {
            if (Math.floor(tCenter[i] / duration) !== 13) continue;
            lo = Math.min(lo, tCenter[i] - kSigma * tSigma[i]);
            hi = Math.max(hi, tCenter[i] + kSigma * tSigma[i]);
        }
        assert(Math.abs(populated.t0 - lo) < 1e-4, `segment 13 t0 ${populated.t0} should be the member minimum ${lo}`);
        assert(Math.abs(populated.t1 - hi) < 1e-4, `segment 13 t1 ${populated.t1} should be the member maximum ${hi}`);

        // list is indexed by segment number -- the range table is keyed to that
        // index, so any reordering silently corrupts the file
        let cursor = meta.segments.persistent[1];
        for (const segment of list) {
            assert.strictEqual(segment.range[0], cursor, 'segments.list must stay in segment-index order');
            cursor = segment.range[1];
        }
    });

    it('should handle a motion axis with zero variance', async () => {
        // a static capture has constant-zero velocity on every axis, so the
        // split-plane range is degenerate -- min === max on all three
        const source = makeFixture(32);
        for (const axis of ['vx', 'vy', 'vz']) {
            source.getColumnByName(axis).data.fill(0);
        }

        const { meta, fields } = decodeSogst(await writeToMemory(source));

        assert.deepStrictEqual(meta.motion.mins, [0, 0, 0]);
        assert.deepStrictEqual(meta.motion.maxs, [0, 0, 0]);

        for (const axis of ['vx', 'vy', 'vz']) {
            for (let i = 0; i < meta.count; ++i) {
                assert.strictEqual(fields[axis][i], 0, `${axis} should decode to exactly 0 at ${i}, not NaN`);
            }
        }
    });

    it('should land reveal_bytes and geometry_bytes on entry boundaries', async () => {
        const source = makeFixture(64, { includeSH: true, shBands: 3 });
        const { meta, entries } = decodeSogst(await writeToMemory(source));

        const { streams } = meta;
        assert(streams, 'the streamed layout is on by default');
        assert.strictEqual(streams.persistent, 'persistent');
        assert.strictEqual(streams.sh_deferred, true);

        const headerOffsets = new Set(entries.map(e => e.headerOffset));
        assert(headerOffsets.has(streams.reveal_bytes), `reveal_bytes ${streams.reveal_bytes} is not an entry boundary`);
        assert(headerOffsets.has(streams.geometry_bytes), `geometry_bytes ${streams.geometry_bytes} is not an entry boundary`);
        assert(streams.reveal_bytes <= streams.geometry_bytes, 'reveal must precede the end of geometry');

        // the deferred SH tail begins exactly at geometry_bytes
        const tail = entries.filter(e => e.headerOffset >= streams.geometry_bytes);
        assert(tail.length > 0, 'there should be a deferred SH tail');
        assert.strictEqual(tail[0].name, 'shN_centroids.webp');
        assert(tail.slice(1).every(e => e.name.endsWith('/shN_labels.webp')), 'only shN labels follow the centroids');

        // spec §4.7: the centroid texture width encodes the band count
        const coeffs = [0, 3, 8, 15][meta.shN.bands];
        const { width } = decodeTexture(entries.find(e => e.name === 'shN_centroids.webp').data);
        assert.strictEqual(width, 64 * coeffs);
    });

    it('should end geometry_bytes at the central directory when there is no SH', async () => {
        // With deferred SH, geometry_bytes is the header offset of the first
        // tail entry, and it is tempting to describe it as "the next entry's
        // local header" in general. With no SH there is no next entry: the
        // offset is the end of the last entry, which is where the central
        // directory begins. A validator built on the first description passes
        // every SH asset and fails every static one.
        const source = makeFixture(64);
        const { meta, entries } = decodeSogst(await writeToMemory(source));

        const { streams } = meta;
        assert.strictEqual(streams.sh_deferred, false, 'this fixture has no SH');
        assert.strictEqual(meta.shN, undefined);

        const last = entries[entries.length - 1];
        const endOfData = last.headerOffset + 30 + last.name.length + last.data.length;

        assert.strictEqual(streams.geometry_bytes, endOfData, 'geometry_bytes should end the data section');
        assert(!entries.some(e => e.headerOffset >= streams.geometry_bytes), 'no entry may follow geometry_bytes');
        assert(streams.reveal_bytes < streams.geometry_bytes, 'reveal must precede the end of geometry');

        // reveal_bytes still points at a real entry header — only the geometry
        // end degenerates when the tail is absent
        const headerOffsets = new Set(entries.map(e => e.headerOffset));
        assert(headerOffsets.has(streams.reveal_bytes), `reveal_bytes ${streams.reveal_bytes} is not an entry boundary`);
    });

    it('should write a monolithic archive when segmentation is disabled', async () => {
        const source = makeFixture(32);
        const { meta, entries } = decodeSogst(await writeToMemory(source, { segmentDuration: 0 }));

        assert.strictEqual(meta.segments, undefined);
        assert.strictEqual(meta.streams, undefined);
        assert(entries.slice(1).every(e => !e.name.includes('/')), 'monolithic entries live at the archive root');

        // the fields still decode, just from whole-clip textures
        const matches = matchByPosition(source, decodeSogst(await writeToMemory(source, { segmentDuration: 0 })).fields, meta.count);
        assert.strictEqual(matches.length, meta.count);
    });

    it('should carry an awkward frame rate through unrounded', async () => {
        // a real fixture clip runs at 29.969999313354492 fps over 2.9696359634399414 s.
        // Neither is 30 or 3, and an encoder that rounds either one produces a file
        // that renders perfectly and plays at the wrong speed.
        const clip = { timeMin: 0, timeMax: 2.9696359634399414, fps: 29.969999313354492 };
        const source = makeFixture(32, { timeMax: clip.timeMax });
        const { meta } = decodeSogst(await writeToMemory(source, { clip }));

        assert.strictEqual(meta.time.fps, clip.fps);
        assert.strictEqual(meta.time.max, clip.timeMax);
        assert.strictEqual(meta.time.min, clip.timeMin);
    });

    it('should refuse to guess missing clip scalars', async () => {
        const source = makeFixture(16);

        await assert.rejects(
            () => writeToMemory(source, { clip: { timeMin: 0, timeMax: 1 } }),
            /fps is missing/,
            'a missing frame rate must be an error, never a default'
        );

        await assert.rejects(
            () => writeToMemory(source, { clip: {} }),
            /timeMin is missing/
        );
    });

    it('should reject data that cannot be represented', async () => {
        const noTemporal = createTestDataTable(16);
        await assert.rejects(() => writeToMemory(noTemporal), /requires temporal columns/);

        const badSigma = makeFixture(16);
        badSigma.getColumnByName('t_sigma').data[3] = 0;
        await assert.rejects(() => writeToMemory(badSigma), /t_sigma > 0/);

        const partialAccel = makeFixture(16, { includeAccel: true });
        partialAccel.removeColumn('az');
        await assert.rejects(() => writeToMemory(partialAccel), /all of ax, ay, az or none/);

        const wrongDegree = makeFixture(16);
        await assert.rejects(
            () => writeToMemory(wrongDegree, { clip: { ...CLIP, motionDegree: 2 } }),
            /motion_degree is 2/
        );
    });

    it('should preserve per-column ranges within quantization tolerance', async () => {
        const source = makeFixture(64);
        const expected = computeSummary(source);
        const { fields, meta } = decodeSogst(await writeToMemory(source));

        assert.strictEqual(meta.count, expected.rowCount);

        const tolerances = {
            t_center: 1e-3, t_sigma: 1e-3,
            vx: 1e-4, vy: 1e-4, vz: 1e-4,
            x: 1e-3, y: 1e-3, z: 1e-3
        };

        for (const [name, tolerance] of Object.entries(tolerances)) {
            const stats = expected.columns[name];
            assert(Math.abs(Math.min(...fields[name]) - stats.min) < tolerance, `${name} min`);
            assert(Math.abs(Math.max(...fields[name]) - stats.max) < tolerance, `${name} max`);
        }
    });
});

describe('SOGST clip comments', () => {
    it('should parse the sogst.* PLY comments', () => {
        const clip = parseSogstComments([
            'sogst.version 1',
            'sogst.time_min 0.5',
            'sogst.time_max 10.25',
            'sogst.fps 24',
            'sogst.motion_degree 2',
            'sogst.cov2d_scale 1.5 2.0',
            'Generated by something else',
            'sogst.future_key whatever'
        ]);

        assert.deepStrictEqual(clip, {
            timeMin: 0.5,
            timeMax: 10.25,
            fps: 24,
            motionDegree: 2,
            cov2dScale: [1.5, 2.0]
        });
    });

    it('should ignore malformed values rather than accept them', () => {
        assert.deepStrictEqual(parseSogstComments(['sogst.fps notanumber']), {});
        assert.deepStrictEqual(parseSogstComments(['sogst.cov2d_scale 1.0']), {});
        assert.deepStrictEqual(parseSogstComments(['sogst.fps']), {});
        assert.deepStrictEqual(parseSogstComments([]), {});
        assert.deepStrictEqual(parseSogstComments(), {});
    });
});
