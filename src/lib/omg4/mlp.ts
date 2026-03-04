/**
 * CPU-side MLP (Multi-Layer Perceptron) forward pass for OMG4 conversion.
 *
 * Replaces the tinycudann FullyFusedMLP at export time.  All operations use
 * plain Float32Arrays so that no GPU or external tensor library is needed.
 *
 * @ignore
 */

import { float16ArrayToFloat32 } from './pickle-helpers';

/**
 * Round up to the nearest multiple of 16 (weight alignment in FullyFusedMLP).
 * @param n - The number to round up.
 * @returns The nearest multiple of 16 >= n.
 * @ignore
 */
const pad16 = (n: number): number => Math.ceil(n / 16) * 16;

/**
 * Evaluate a 1-hidden-layer FullyFusedMLP on CPU.
 *
 * The weight layout matches tinycudann's FullyFusedMLP:
 *   Layer 0: weight [hid_pad × inp_pad]  (row-major, float16)
 *   Layer 1: weight [out_pad × hid_pad]
 * No biases are stored.
 *
 * @param paramsF16 - Flat float16 weights stored as Uint16Array.
 * @param x - Input matrix [N × nInput] as flat Float32Array.
 * @param N - Number of rows.
 * @param nInput - Input dimension.
 * @param nHidden - Hidden layer width.
 * @param nOutput - Output dimension.
 * @param activation - 'relu' or 'leaky_relu'.
 * @returns Output matrix [N × nOutput] as flat Float32Array.
 */
const tcnnMlpForward = (
    paramsF16: Uint16Array,
    x: Float32Array,
    N: number,
    nInput: number,
    nHidden: number,
    nOutput: number,
    activation: 'relu' | 'leaky_relu' = 'relu'
): Float32Array => {
    const inpPad = pad16(nInput);
    const hidPad = pad16(nHidden);
    const outPad = pad16(nOutput);

    // Convert float16 params to float32
    const params = float16ArrayToFloat32(paramsF16);

    const w0Size = inpPad * hidPad;
    const w1Size = hidPad * outPad;
    if (params.length < w0Size + w1Size) {
        throw new Error(`params size mismatch: got ${params.length}, need ${w0Size + w1Size}`);
    }

    // Extract weights (sliced to actual dimensions)
    // W0: [nHidden × nInput] from [hidPad × inpPad]
    // W1: [nOutput × nHidden] from [outPad × hidPad]

    // h = x @ W0.T  →  h[i,j] = sum_k x[i,k] * W0[j,k]
    const h = new Float32Array(N * nHidden);
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < nHidden; j++) {
            let sum = 0;
            for (let k = 0; k < nInput; k++) {
                // W0 stored as [hidPad rows × inpPad cols], row-major
                sum += x[i * nInput + k] * params[j * inpPad + k];
            }
            h[i * nHidden + j] = sum;
        }
    }

    // Apply activation
    if (activation === 'relu') {
        for (let i = 0; i < h.length; i++) {
            if (h[i] < 0) h[i] = 0;
        }
    } else if (activation === 'leaky_relu') {
        for (let i = 0; i < h.length; i++) {
            if (h[i] < 0) h[i] *= 0.01;
        }
    }

    // out = h @ W1.T  →  out[i,j] = sum_k h[i,k] * W1[j,k]
    const out = new Float32Array(N * nOutput);
    const w1Offset = w0Size;
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < nOutput; j++) {
            let sum = 0;
            for (let k = 0; k < nHidden; k++) {
                sum += h[i * nHidden + k] * params[w1Offset + j * hidPad + k];
            }
            out[i * nOutput + j] = sum;
        }
    }

    return out;
};

/**
 * Apply sinusoidal frequency (positional) encoding.
 *
 * For each input dimension d and frequency index k in [0, nFrequencies):
 *   sin(2^k * π * x_d), cos(2^k * π * x_d)
 *
 * @param x - Input [N × D] flat Float32Array.
 * @param N - Number of rows.
 * @param D - Input dimensions.
 * @param nFrequencies - Number of frequency octaves.
 * @returns Encoded [N × D*nFrequencies*2] flat Float32Array.
 */
const frequencyEncode = (
    x: Float32Array,
    N: number,
    D: number,
    nFrequencies: number
): Float32Array => {
    const outDim = D * nFrequencies * 2;
    const out = new Float32Array(N * outDim);

    // Precompute frequencies: 2^k * π
    const freqs = new Float32Array(nFrequencies);
    for (let k = 0; k < nFrequencies; k++) {
        freqs[k] = Math.pow(2, k) * Math.PI;
    }

    for (let i = 0; i < N; i++) {
        let outIdx = i * outDim;
        for (let d = 0; d < D; d++) {
            const val = x[i * D + d];
            for (let k = 0; k < nFrequencies; k++) {
                const angle = val * freqs[k];
                out[outIdx++] = Math.sin(angle);
                out[outIdx++] = Math.cos(angle);
            }
        }
    }

    return out;
};

/**
 * Forward pass for mlp_cont (NetworkWithInputEncoding).
 *
 * Applies sinusoidal frequency encoding to the input, then runs the MLP.
 *
 * @param paramsF16 - Flat float16 MLP weights (Uint16Array).
 * @param xyzNorm - Normalised input [N × D] as flat Float32Array.
 * @param N - Number of rows.
 * @param D - Input dimension (before encoding).
 * @param nFrequencies - Number of frequency octaves per dimension.
 * @param nOutput - MLP output dimension.
 * @returns Output [N × nOutput] as flat Float32Array.
 */
const tcnnNetworkWithEncodingForward = (
    paramsF16: Uint16Array,
    xyzNorm: Float32Array,
    N: number,
    D: number,
    nFrequencies: number = 16,
    nOutput: number = 13
): Float32Array => {
    const encoded = frequencyEncode(xyzNorm, N, D, nFrequencies);
    const nInput = D * nFrequencies * 2;
    return tcnnMlpForward(paramsF16, encoded, N, nInput, 64, nOutput, 'relu');
};

/**
 * Contract positions to the unisphere (same as OMG4 renderer).
 *
 * Points inside the unit cube [-1,1]³ are linearly mapped to [0.25, 0.75].
 * Points outside are contracted via inverse-magnitude mapping.
 *
 * @param positions - [N × 3] flat Float32Array (modified in place).
 * @param N - Number of points.
 * @returns The same array, modified in place.
 */
const contractToUnisphere = (positions: Float32Array, N: number): Float32Array => {
    for (let i = 0; i < N; i++) {
        const off = i * 3;
        // AABB [-1,1] → [0,1]
        let px = (positions[off + 0] + 1) * 0.5;
        let py = (positions[off + 1] + 1) * 0.5;
        let pz = (positions[off + 2] + 1) * 0.5;

        // [0,1] → [-1,1]
        px = px * 2 - 1;
        py = py * 2 - 1;
        pz = pz * 2 - 1;

        const mag = Math.sqrt(px * px + py * py + pz * pz);
        if (mag > 1) {
            const invMag = 1 / mag;
            const scale = (2 - invMag) * invMag;
            px *= scale;
            py *= scale;
            pz *= scale;
        }

        // [-1,1] → map to [0.25, 0.75] via /4 + 0.5
        positions[off + 0] = px / 4 + 0.5;
        positions[off + 1] = py / 4 + 0.5;
        positions[off + 2] = pz / 4 + 0.5;
    }

    return positions;
};

/**
 * Normalise each row of a [N × 4] quaternion array in place.
 * @param quats - The flat Float32Array of quaternion data.
 * @param N - The number of quaternions.
 * @returns The same array, modified in place.
 * @ignore
 */
const normalizeQuaternions = (quats: Float32Array, N: number): Float32Array => {
    for (let i = 0; i < N; i++) {
        const off = i * 4;
        const w = quats[off], x = quats[off + 1], y = quats[off + 2], z = quats[off + 3];
        const len = Math.sqrt(w * w + x * x + y * y + z * z);
        if (len > 0) {
            const inv = 1 / len;
            quats[off] *= inv;
            quats[off + 1] *= inv;
            quats[off + 2] *= inv;
            quats[off + 3] *= inv;
        }
    }
    return quats;
};

export {
    tcnnMlpForward,
    tcnnNetworkWithEncodingForward,
    frequencyEncode,
    contractToUnisphere,
    normalizeQuaternions,
    pad16
};
