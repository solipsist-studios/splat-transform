/**
 * Tests for OMG4 conversion (xz → omg4).
 *
 * Tests the individual components (pickle helpers, huffman, MLP) and the
 * end-to-end conversion pipeline.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { convertOmg4 } from '../src/lib/omg4/convert-omg4.js';
import { huffmanDecode } from '../src/lib/omg4/huffman.js';
import {
    tcnnMlpForward,
    frequencyEncode,
    contractToUnisphere,
    normalizeQuaternions,
    pad16
} from '../src/lib/omg4/mlp.js';
import {
    float16ToFloat32,
    float16ArrayToFloat32,
    bytesObjectToUint8Array
} from '../src/lib/omg4/pickle-helpers.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, 'fixtures', 'omg4');

// ── Pickle helpers ──────────────────────────────────────────────────────────

describe('OMG4 - Pickle helpers', () => {
    it('float16ToFloat32 should convert common values', () => {
        // 0x3C00 = 1.0 in float16
        assert.strictEqual(float16ToFloat32(0x3C00), 1.0);
        // 0x0000 = 0.0
        assert.strictEqual(float16ToFloat32(0x0000), 0);
        // 0xBC00 = -1.0
        assert.strictEqual(float16ToFloat32(0xBC00), -1.0);
        // 0x4000 = 2.0
        assert.strictEqual(float16ToFloat32(0x4000), 2.0);
        // 0x3800 = 0.5
        assert.strictEqual(float16ToFloat32(0x3800), 0.5);
    });

    it('float16ToFloat32 should handle special values', () => {
        // Positive infinity: 0x7C00
        assert.strictEqual(float16ToFloat32(0x7C00), Infinity);
        // Negative infinity: 0xFC00
        assert.strictEqual(float16ToFloat32(0xFC00), -Infinity);
        // NaN: 0x7C01
        assert.ok(isNaN(float16ToFloat32(0x7C01)));
    });

    it('float16ArrayToFloat32 should convert array', () => {
        const u16 = new Uint16Array([0x3C00, 0x4000, 0x3800]); // 1.0, 2.0, 0.5
        const f32 = float16ArrayToFloat32(u16);
        assert.strictEqual(f32.length, 3);
        assert.strictEqual(f32[0], 1.0);
        assert.strictEqual(f32[1], 2.0);
        assert.strictEqual(f32[2], 0.5);
    });

    it('bytesObjectToUint8Array should handle Uint8Array directly', () => {
        const input = new Uint8Array([1, 2, 3]);
        const result = bytesObjectToUint8Array(input);
        assert.deepStrictEqual(Array.from(result), [1, 2, 3]);
    });

    it('bytesObjectToUint8Array should handle PObject with latin1 args', () => {
        // Simulates pickleparser's representation of Python bytes
        const mockPObject = { args: ['\x01\x02\x03', 'latin1'] };
        const result = bytesObjectToUint8Array(mockPObject);
        assert.deepStrictEqual(Array.from(result), [1, 2, 3]);
    });
});

// ── Huffman decoder ─────────────────────────────────────────────────────────

describe('OMG4 - Huffman decoder', () => {
    it('should decode a simple huffman-encoded stream', () => {
        // Code table: symbol → (bit_length, code_value)
        // 0 → (2, 0b00), 1 → (2, 0b01), 2 → (2, 0b10), EOF → (3, 0b110)
        const codeTable = {
            0: [2, 0],
            1: [2, 1],
            2: [2, 2],
            null: [3, 6]  // EOF = 0b110
        };

        // Encode sequence [0, 1, 2] + EOF:
        // 00 01 10 110 0  →  0001 1011 0000 0000  →  0x1B 0x00
        const encoded = new Uint8Array([0x1B, 0x00]);
        const decoded = huffmanDecode(encoded, codeTable);
        assert.deepStrictEqual(Array.from(decoded), [0, 1, 2]);
    });

    it('should decode single symbol', () => {
        const codeTable = {
            42: [1, 0],
            null: [2, 2]  // EOF = 0b10
        };
        // 0 10 00000  →  0x40
        const encoded = new Uint8Array([0x40]);
        const decoded = huffmanDecode(encoded, codeTable);
        assert.deepStrictEqual(Array.from(decoded), [42]);
    });

    it('should handle empty stream (EOF immediately)', () => {
        const codeTable = {
            0: [2, 0],
            null: [1, 1]  // EOF = 0b1
        };
        // 1 0000000 → 0x80
        const encoded = new Uint8Array([0x80]);
        const decoded = huffmanDecode(encoded, codeTable);
        assert.strictEqual(decoded.length, 0);
    });
});

// ── MLP ─────────────────────────────────────────────────────────────────────

describe('OMG4 - MLP utilities', () => {
    it('pad16 should round up to multiples of 16', () => {
        assert.strictEqual(pad16(1), 16);
        assert.strictEqual(pad16(16), 16);
        assert.strictEqual(pad16(17), 32);
        assert.strictEqual(pad16(64), 64);
        assert.strictEqual(pad16(0), 0);
    });

    it('frequencyEncode should produce correct output dimensions', () => {
        const N = 2;
        const D = 3;
        const nFreq = 4;
        const input = new Float32Array([1, 2, 3, 4, 5, 6]); // [2 × 3]
        const encoded = frequencyEncode(input, N, D, nFreq);
        assert.strictEqual(encoded.length, N * D * nFreq * 2);
    });

    it('frequencyEncode should produce sin/cos pairs', () => {
        const input = new Float32Array([0.5]); // [1 × 1]
        const encoded = frequencyEncode(input, 1, 1, 2); // 2 frequencies
        // freq[0] = 2^0 * π = π, freq[1] = 2^1 * π = 2π
        // Output: [sin(0.5*π), cos(0.5*π), sin(0.5*2π), cos(0.5*2π)]
        assert.strictEqual(encoded.length, 4);
        assert.ok(Math.abs(encoded[0] - Math.sin(0.5 * Math.PI)) < 1e-6);
        assert.ok(Math.abs(encoded[1] - Math.cos(0.5 * Math.PI)) < 1e-6);
        assert.ok(Math.abs(encoded[2] - Math.sin(0.5 * 2 * Math.PI)) < 1e-6);
        assert.ok(Math.abs(encoded[3] - Math.cos(0.5 * 2 * Math.PI)) < 1e-6);
    });

    it('contractToUnisphere should map unit cube identity', () => {
        // Points inside [-1,1]^3 should map into [0.25, 0.75]
        const positions = new Float32Array([0, 0, 0]); // origin
        contractToUnisphere(positions, 1);
        assert.ok(Math.abs(positions[0] - 0.5) < 1e-6);
        assert.ok(Math.abs(positions[1] - 0.5) < 1e-6);
        assert.ok(Math.abs(positions[2] - 0.5) < 1e-6);
    });

    it('normalizeQuaternions should normalize to unit length', () => {
        const quats = new Float32Array([2, 0, 0, 0, 0, 3, 0, 0]); // 2 quats
        normalizeQuaternions(quats, 2);
        assert.ok(Math.abs(quats[0] - 1.0) < 1e-6);
        assert.ok(Math.abs(quats[4]) < 1e-6);
        assert.ok(Math.abs(quats[5] - 1.0) < 1e-6);
    });

    it('tcnnMlpForward should produce correct output dimensions', () => {
        // Simple 2-layer MLP: input=4, hidden=16, output=2
        const nInput = 4;
        const nHidden = 16;
        const nOutput = 2;
        const N = 3;

        const w0Size = pad16(nInput) * pad16(nHidden);
        const w1Size = pad16(nHidden) * pad16(nOutput);

        // Create zero weights (stored as float16 = uint16)
        const params = new Uint16Array(w0Size + w1Size);
        const input = new Float32Array(N * nInput);

        const output = tcnnMlpForward(params, input, N, nInput, nHidden, nOutput, 'relu');
        assert.strictEqual(output.length, N * nOutput);
    });
});

// ── End-to-end conversion ───────────────────────────────────────────────────

describe('OMG4 - End-to-end conversion', () => {
    it('should convert xz checkpoint to omg4 binary format', async () => {
        const xzData = await readFile(join(fixturesDir, 'test_checkpoint.xz'));

        const omg4Data = convertOmg4(
            new Uint8Array(xzData.buffer, xzData.byteOffset, xzData.byteLength),
            { numFrames: 2, fps: 24, timeMin: -0.5, timeMax: 0.5 }
        );

        assert.ok(omg4Data instanceof Uint8Array, 'Should return Uint8Array');

        // Parse and validate header
        const view = new DataView(omg4Data.buffer, omg4Data.byteOffset, omg4Data.byteLength);
        const magic = view.getUint32(0, true);
        const version = view.getUint32(4, true);
        const numSplats = view.getUint32(8, true);
        const numFrames = view.getUint32(12, true);
        const fps = view.getFloat32(16, true);
        const timeMin = view.getFloat32(20, true);
        const timeMax = view.getFloat32(24, true);

        assert.strictEqual(magic, 0x34474D4F, 'Magic should be "OMG4"');
        assert.strictEqual(version, 1, 'Version should be 1');
        assert.strictEqual(numSplats, 3, 'Should have 3 splats');
        assert.strictEqual(numFrames, 2, 'Should have 2 frames');
        assert.strictEqual(fps, 24, 'FPS should be 24');
        assert.ok(Math.abs(timeMin - (-0.5)) < 1e-6, 'timeMin should be -0.5');
        assert.ok(Math.abs(timeMax - 0.5) < 1e-6, 'timeMax should be 0.5');

        // Check total size: header(28) + numFrames * (4 + numSplats * 14 * 4)
        const HEADER_SIZE = 28;
        const FLOATS_PER_SPLAT = 14;
        const expectedSize = HEADER_SIZE + numFrames * (4 + numSplats * FLOATS_PER_SPLAT * 4);
        assert.strictEqual(omg4Data.length, expectedSize, 'Total size should match expected');

        // Verify timestamps
        let offset = HEADER_SIZE;
        const ts0 = view.getFloat32(offset, true);
        assert.ok(Math.abs(ts0 - (-0.5)) < 1e-6, 'First frame timestamp should be timeMin');

        const frameSize = 4 + numSplats * FLOATS_PER_SPLAT * 4;
        offset += frameSize;
        const ts1 = view.getFloat32(offset, true);
        assert.ok(Math.abs(ts1 - 0.5) < 1e-6, 'Last frame timestamp should be timeMax');

        // Verify per-splat data is finite (not NaN/Inf)
        offset = HEADER_SIZE + 4; // skip first timestamp
        for (let s = 0; s < numSplats; s++) {
            for (let f = 0; f < FLOATS_PER_SPLAT; f++) {
                const val = view.getFloat32(offset + (s * FLOATS_PER_SPLAT + f) * 4, true);
                assert.ok(isFinite(val), `Splat ${s} field ${f} should be finite, got ${val}`);
            }
        }
    });

    it('should handle single frame', async () => {
        const xzData = await readFile(join(fixturesDir, 'test_checkpoint.xz'));

        const omg4Data = convertOmg4(
            new Uint8Array(xzData.buffer, xzData.byteOffset, xzData.byteLength),
            { numFrames: 1, fps: 1, timeMin: 0, timeMax: 0 }
        );

        const view = new DataView(omg4Data.buffer, omg4Data.byteOffset, omg4Data.byteLength);
        assert.strictEqual(view.getUint32(12, true), 1, 'Should have 1 frame');

        // Single frame timestamp should be timeMin
        const ts = view.getFloat32(28, true);
        assert.strictEqual(ts, 0, 'Single frame timestamp should be 0');
    });
});
