/**
 * Convert an OMG4 compressed 4DGS checkpoint (.xz) to the web-friendly .omg4
 * binary format.
 *
 * The .omg4 format stores pre-baked per-frame Gaussian attributes so that the
 * viewer can play back the animation without GPU-side MLP inference.
 *
 * @ignore
 */

import { xz } from '@napi-rs/lzma';
import { Parser } from 'pickleparser';

import { huffmanDecode } from './huffman';
import { tcnnMlpForward, tcnnNetworkWithEncodingForward, contractToUnisphere, normalizeQuaternions } from './mlp';
import { bytesObjectToUint8Array, numpyToFloat32Array, numpyToFloat16Raw } from './pickle-helpers';
import { logger } from '../utils/logger';

/**
 * Options for the OMG4 conversion.
 */
type ConvertOmg4Options = {
    /** Number of output frames. Default: 30 */
    numFrames: number;
    /** Frames per second. Default: 24 */
    fps: number;
    /** Time duration minimum. Default: -0.5 */
    timeMin: number;
    /** Time duration maximum. Default: 0.5 */
    timeMax: number;
};

const defaultOptions: ConvertOmg4Options = {
    numFrames: 30,
    fps: 24,
    timeMin: -0.5,
    timeMax: 0.5
};

/**
 * Decode a VQ (vector-quantised) attribute layer.
 *
 * codes   : Float32Array [K × d] cluster centres
 * indices : Uint16Array  [N]     Huffman-decoded cluster labels
 *
 * Returns Float32Array [N × d].
 * @param codes - The cluster centre codebook.
 * @param indices - The Huffman-decoded cluster labels.
 * @param K - The number of clusters.
 * @param d - The dimension of each cluster centre.
 * @returns A Float32Array [N × d] of decoded attribute values.
 * @ignore
 */
const vqDecode = (codes: Float32Array, indices: Uint16Array, K: number, d: number): Float32Array => {
    const N = indices.length;
    const out = new Float32Array(N * d);
    for (let i = 0; i < N; i++) {
        const idx = indices[i];
        for (let j = 0; j < d; j++) {
            out[i * d + j] = codes[idx * d + j];
        }
    }
    return out;
};

/**
 * Decode all VQ layers for one attribute and concatenate along the last dimension.
 * @param codeList - Array of numpy ndarray PObjects for cluster centres.
 * @param indexList - Array of Huffman-encoded index byte objects.
 * @param htableList - Array of Huffman code table objects.
 * @returns An object containing the decoded data, number of elements N, and dimension d.
 * @ignore
 */
const decodeAllLayers = (
    codeList: any[],
    indexList: any[],
    htableList: any[]
): { data: Float32Array; N: number; d: number } => {
    const parts: { data: Float32Array; d: number; N: number }[] = [];

    for (let layer = 0; layer < codeList.length; layer++) {
        const codes = numpyToFloat32Array(codeList[layer]);
        const shape = codeList[layer]['1'] as number[];  // [K, d]
        const K = shape[0];
        const d = shape[1];

        const idxBytes = bytesObjectToUint8Array(indexList[layer]);
        const htable = htableList[layer];

        const labels = huffmanDecode(idxBytes, htable);
        const decoded = vqDecode(codes, labels, K, d);
        parts.push({ data: decoded, d, N: labels.length });
    }

    if (parts.length === 1) {
        return { data: parts[0].data, N: parts[0].N, d: parts[0].d };
    }

    // Concatenate along last dimension
    const N = parts[0].N;
    const totalD = parts.reduce((sum, p) => sum + p.d, 0);
    const out = new Float32Array(N * totalD);
    for (let i = 0; i < N; i++) {
        let offset = 0;
        for (const part of parts) {
            for (let j = 0; j < part.d; j++) {
                out[i * totalD + offset + j] = part.data[i * part.d + j];
            }
            offset += part.d;
        }
    }

    return { data: out, N, d: totalD };
};

/**
 * Header constants for the .omg4 binary format.
 */
const OMG4_MAGIC = 0x34474D4F;   // "OMG4" little-endian
const OMG4_VERSION = 1;
const OMG4_HEADER_SIZE = 28;     // 7 × uint32/float32
const FLOATS_PER_SPLAT = 14;

/**
 * Convert an OMG4 compressed checkpoint (.xz) to the .omg4 binary format.
 *
 * @param xzData - Raw bytes of the .xz compressed checkpoint.
 * @param options - Conversion options (frames, fps, time range).
 * @returns The .omg4 binary data as a Uint8Array.
 */
const convertOmg4 = (
    xzData: Uint8Array,
    options?: Partial<ConvertOmg4Options>
): Uint8Array => {
    const opts = { ...defaultOptions, ...options };

    logger.log('Decompressing .xz checkpoint...');
    const decompressed = xz.decompressSync(Buffer.from(xzData));

    logger.log('Parsing checkpoint data...');
    const parser = new Parser();
    const saveDict = parser.parse<Record<string, any>>(Buffer.from(decompressed));

    // ── Decode geometry ──────────────────────────────────────────────────────
    const xyz = numpyToFloat32Array(saveDict.xyz);          // [N × 3]
    const tCenter = numpyToFloat32Array(saveDict.t);        // [N × 1]
    const N = xyz.length / 3;

    logger.log(`  ${N.toLocaleString()} Gaussians | ${opts.numFrames} frames @ ${opts.fps} fps`);

    const scaling = decodeAllLayers(
        saveDict.scale_code,
        saveDict.scale_index,
        saveDict.scale_htable
    );                                                          // [N × 3]

    const rotation = decodeAllLayers(
        saveDict.rotation_code,
        saveDict.rotation_index,
        saveDict.rotation_htable
    );                                                          // [N × 4]

    const appearance = decodeAllLayers(
        saveDict.app_code,
        saveDict.app_index,
        saveDict.app_htable
    );                                                          // [N × 6]

    // 4D attributes
    const scalingT = decodeAllLayers(
        saveDict.scaling_t_code,
        saveDict.scaling_t_index,
        saveDict.scaling_t_htable
    );                                                          // [N × 1]

    // MLP weights (flat float16 as Uint16Array)
    const mlpContParams = numpyToFloat16Raw(saveDict.MLP_cont);
    const mlpDcParams = numpyToFloat16Raw(saveDict.MLP_dc);
    const mlpOpacityParams = numpyToFloat16Raw(saveDict.MLP_opacity);

    // Static appearance features (first 3 dims)
    // appearance is [N × 6]: first 3 = features_static, next 3 = features_view

    // ── Allocate output buffer ───────────────────────────────────────────────
    const frameDataSize = 4 + N * FLOATS_PER_SPLAT * 4; // timestamp + per-splat data
    const totalSize = OMG4_HEADER_SIZE + opts.numFrames * frameDataSize;
    const outputBuffer = new ArrayBuffer(totalSize);
    const view = new DataView(outputBuffer);

    // ── Write header ─────────────────────────────────────────────────────────
    let offset = 0;
    view.setUint32(offset, OMG4_MAGIC, true); offset += 4;
    view.setUint32(offset, OMG4_VERSION, true); offset += 4;
    view.setUint32(offset, N, true); offset += 4;
    view.setUint32(offset, opts.numFrames, true); offset += 4;
    view.setFloat32(offset, opts.fps, true); offset += 4;
    view.setFloat32(offset, opts.timeMin, true); offset += 4;
    view.setFloat32(offset, opts.timeMax, true); offset += 4;

    // ── Per-frame processing ─────────────────────────────────────────────────
    for (let fi = 0; fi < opts.numFrames; fi++) {
        const timestamp = opts.timeMin + (opts.timeMax - opts.timeMin) * fi / Math.max(opts.numFrames - 1, 1);
        const timestampNorm = fi / Math.max(opts.numFrames - 1, 1);

        logger.log(`  Frame ${fi + 1}/${opts.numFrames}  t=${timestamp.toFixed(4)}`);

        // ── Temporal masking ─────────────────────────────────────────────
        // sigma_t = exp(scaling_t)
        // weight_t = exp(-0.5 * (t_center - timestamp)^2 / (sigma_t^2 + 1e-8))
        const weightT = new Float32Array(N);
        for (let i = 0; i < N; i++) {
            const sigmaT = Math.exp(scalingT.data[i]);
            const dt = tCenter[i] - timestamp;
            weightT[i] = Math.exp(-0.5 * dt * dt / (sigmaT * sigmaT + 1e-8));
        }

        // ── Position (no 4D deformation) ─────────────────────────────────
        const pos = new Float32Array(xyz);  // clone

        // ── MLP inference ────────────────────────────────────────────────
        const xyzContracted = contractToUnisphere(new Float32Array(pos), N);

        // Build [N × 4] input: [contracted_xyz, t_norm]
        const xyzt = new Float32Array(N * 4);
        for (let i = 0; i < N; i++) {
            xyzt[i * 4 + 0] = xyzContracted[i * 3 + 0];
            xyzt[i * 4 + 1] = xyzContracted[i * 3 + 1];
            xyzt[i * 4 + 2] = xyzContracted[i * 3 + 2];
            xyzt[i * 4 + 3] = timestampNorm;
        }

        const contFeat = tcnnNetworkWithEncodingForward(
            mlpContParams, xyzt, N, 4, 16, 13
        );                                                      // [N × 13]

        // space_feat = cat(cont_feat, features_static)  → [N × 16]
        const spaceFeat = new Float32Array(N * 16);
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < 13; j++) {
                spaceFeat[i * 16 + j] = contFeat[i * 13 + j];
            }
            for (let j = 0; j < 3; j++) {
                spaceFeat[i * 16 + 13 + j] = appearance.data[i * appearance.d + j];
            }
        }

        const dc = tcnnMlpForward(
            mlpDcParams, spaceFeat, N, 16, 64, 3, 'leaky_relu'
        );                                                      // [N × 3]

        const rawOpa = tcnnMlpForward(
            mlpOpacityParams, spaceFeat, N, 16, 64, 1, 'leaky_relu'
        );                                                      // [N × 1]

        // Normalise rotation quaternion
        normalizeQuaternions(rotation.data, N);

        // ── Write frame ──────────────────────────────────────────────────
        view.setFloat32(offset, timestamp, true); offset += 4;

        for (let i = 0; i < N; i++) {
            // Combine temporal weight into opacity
            const opaSigmoid = 1 / (1 + Math.exp(-rawOpa[i]));
            let opaFinal = opaSigmoid * weightT[i];
            opaFinal = Math.max(1e-6, Math.min(1 - 1e-6, opaFinal));
            const opaLogit = Math.log(opaFinal / (1 - opaFinal));

            // x, y, z
            view.setFloat32(offset, pos[i * 3 + 0], true); offset += 4;
            view.setFloat32(offset, pos[i * 3 + 1], true); offset += 4;
            view.setFloat32(offset, pos[i * 3 + 2], true); offset += 4;
            // rot_0(w), rot_1(x), rot_2(y), rot_3(z)
            view.setFloat32(offset, rotation.data[i * 4 + 0], true); offset += 4;
            view.setFloat32(offset, rotation.data[i * 4 + 1], true); offset += 4;
            view.setFloat32(offset, rotation.data[i * 4 + 2], true); offset += 4;
            view.setFloat32(offset, rotation.data[i * 4 + 3], true); offset += 4;
            // scale_0, scale_1, scale_2 (log-space)
            view.setFloat32(offset, scaling.data[i * 3 + 0], true); offset += 4;
            view.setFloat32(offset, scaling.data[i * 3 + 1], true); offset += 4;
            view.setFloat32(offset, scaling.data[i * 3 + 2], true); offset += 4;
            // opacity (logit-space)
            view.setFloat32(offset, opaLogit, true); offset += 4;
            // f_dc_0, f_dc_1, f_dc_2
            view.setFloat32(offset, dc[i * 3 + 0], true); offset += 4;
            view.setFloat32(offset, dc[i * 3 + 1], true); offset += 4;
            view.setFloat32(offset, dc[i * 3 + 2], true); offset += 4;
        }
    }

    return new Uint8Array(outputBuffer);
};

export { convertOmg4, type ConvertOmg4Options };
