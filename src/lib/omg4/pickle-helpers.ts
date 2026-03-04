/**
 * Helpers for extracting typed arrays from Python pickle objects parsed by pickleparser.
 *
 * When a Python pickle containing numpy arrays is parsed by pickleparser, numpy
 * arrays are represented as PObject instances with:
 *   - prototype.__module__ = 'numpy._core.multiarray' (or 'numpy.core.multiarray')
 *   - prototype.__name__ = '_reconstruct'
 *   - args = [ndarrayClass, [0], bytes]      (REDUCE constructor)
 *   - '0' = version, '1' = shape, '2' = dtype, '3' = fortranOrder, '4' = rawData
 *
 * @ignore
 */

/**
 * Map numpy dtype strings to TypedArray constructors and byte sizes.
 */
const dtypeMap: Record<string, { ctor: Float32ArrayConstructor | Float64ArrayConstructor | Uint8ArrayConstructor | Int8ArrayConstructor | Uint16ArrayConstructor | Int16ArrayConstructor | Uint32ArrayConstructor | Int32ArrayConstructor; bytes: number }> = {
    'f2': { ctor: Uint16Array as unknown as Float32ArrayConstructor, bytes: 2 }, // float16 → stored as uint16, converted later
    'f4': { ctor: Float32Array, bytes: 4 },
    'f8': { ctor: Float64Array, bytes: 8 },
    'u1': { ctor: Uint8Array, bytes: 1 },
    'i1': { ctor: Int8Array, bytes: 1 },
    'u2': { ctor: Uint16Array, bytes: 2 },
    'i2': { ctor: Int16Array, bytes: 2 },
    'u4': { ctor: Uint32Array, bytes: 4 },
    'i4': { ctor: Int32Array, bytes: 4 }
};

/**
 * Convert a pickleparser bytes PObject to a Uint8Array.
 *
 * Bytes objects from pickleparser have: args = [latin1EncodedString, 'latin1']
 * @param obj - The bytes PObject to convert.
 * @returns A Uint8Array of the raw bytes.
 * @ignore
 */
const bytesObjectToUint8Array = (obj: any): Uint8Array => {
    if (obj instanceof Uint8Array || obj instanceof Buffer) {
        return new Uint8Array(obj);
    }
    if (obj?.args && typeof obj.args[0] === 'string') {
        return Uint8Array.from(Buffer.from(obj.args[0], 'latin1'));
    }
    throw new Error('Cannot convert object to bytes');
};

/**
 * Check if an object is a numpy ndarray PObject from pickleparser.
 * @param obj - The object to check.
 * @returns True if the object is a numpy ndarray PObject.
 * @ignore
 */
const isNumpyArray = (obj: any): boolean => {
    if (!obj || typeof obj !== 'object') return false;
    const proto = Object.getPrototypeOf(obj);
    if (!proto) return false;
    const mod = proto.__module__ ?? '';
    const name = proto.__name__ ?? '';
    return (mod === 'numpy._core.multiarray' || mod === 'numpy.core.multiarray') &&
           name === '_reconstruct';
};

/**
 * Convert a single IEEE 754 half-precision float (stored as uint16) to float32.
 * @param h - The half-precision float value as a uint16.
 * @returns The float32 equivalent.
 * @ignore
 */
const float16ToFloat32 = (h: number): number => {
    const sign = (h >> 15) & 0x1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;

    if (exponent === 0) {
        if (mantissa === 0) {
            // Zero
            return sign ? -0 : 0;
        }
        // Subnormal
        let e = -14;
        let m = mantissa;
        while ((m & 0x400) === 0) {
            m <<= 1;
            e--;
        }
        m &= 0x3ff;
        const f32 = (sign ? -1 : 1) * Math.pow(2, e) * (1 + m / 1024);
        return f32;
    } else if (exponent === 31) {
        if (mantissa === 0) {
            return sign ? -Infinity : Infinity;
        }
        return NaN;
    }

    // Normal
    return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
};

/**
 * Convert an array of IEEE 754 half-precision (float16) values stored as
 * uint16 to a Float32Array.
 * @param u16 - The Uint16Array of half-precision float values.
 * @returns A Float32Array with the converted values.
 * @ignore
 */
const float16ArrayToFloat32 = (u16: Uint16Array): Float32Array => {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
        out[i] = float16ToFloat32(u16[i]);
    }
    return out;
};

/**
 * Extract a Float32Array from a numpy ndarray PObject.
 *
 * Handles float16 (f2) by converting to float32, and float32 (f4) directly.
 * @param obj - The numpy ndarray PObject.
 * @returns A Float32Array of the extracted data.
 * @ignore
 */
const numpyToFloat32Array = (obj: any): Float32Array => {
    if (!isNumpyArray(obj)) {
        throw new Error('Object is not a numpy ndarray');
    }

    const dtypeStr: string = obj['2']?.args?.[0] ?? '';
    const rawBytes = bytesObjectToUint8Array(obj['4']);

    if (dtypeStr === 'f4') {
        // float32 - direct view
        const aligned = new Uint8Array(rawBytes.length);
        aligned.set(rawBytes);
        return new Float32Array(aligned.buffer, 0, rawBytes.length / 4);
    } else if (dtypeStr === 'f2') {
        // float16 - convert to float32
        const aligned = new Uint8Array(rawBytes.length);
        aligned.set(rawBytes);
        const u16 = new Uint16Array(aligned.buffer, 0, rawBytes.length / 2);
        return float16ArrayToFloat32(u16);
    } else if (dtypeStr === 'f8') {
        // float64 - downcast to float32
        const aligned = new Uint8Array(rawBytes.length);
        aligned.set(rawBytes);
        const f64 = new Float64Array(aligned.buffer, 0, rawBytes.length / 8);
        const f32 = new Float32Array(f64.length);
        for (let i = 0; i < f64.length; i++) {
            f32[i] = f64[i];
        }
        return f32;
    }

    throw new Error(`Unsupported numpy dtype: ${dtypeStr}`);
};

/**
 * Extract the shape from a numpy ndarray PObject.
 * @param obj - The numpy ndarray PObject.
 * @returns An array of dimension sizes.
 * @ignore
 */
const numpyShape = (obj: any): number[] => {
    if (!isNumpyArray(obj)) {
        throw new Error('Object is not a numpy ndarray');
    }
    const shape = obj['1'];
    if (Array.isArray(shape)) {
        return shape;
    }
    // Single dimension stored as number
    return [shape];
};

/**
 * Extract a float16 numpy array as raw Uint16Array (for MLP weight storage).
 * @param obj - The numpy ndarray PObject with float16 dtype.
 * @returns A Uint16Array of raw float16 values.
 * @ignore
 */
const numpyToFloat16Raw = (obj: any): Uint16Array => {
    if (!isNumpyArray(obj)) {
        throw new Error('Object is not a numpy ndarray');
    }
    const dtypeStr: string = obj['2']?.args?.[0] ?? '';
    if (dtypeStr !== 'f2') {
        throw new Error(`Expected float16 (f2) dtype, got: ${dtypeStr}`);
    }
    const rawBytes = bytesObjectToUint8Array(obj['4']);
    const aligned = new Uint8Array(rawBytes.length);
    aligned.set(rawBytes);
    return new Uint16Array(aligned.buffer, 0, rawBytes.length / 2);
};

export {
    bytesObjectToUint8Array,
    isNumpyArray,
    numpyToFloat32Array,
    numpyShape,
    numpyToFloat16Raw,
    float16ArrayToFloat32,
    float16ToFloat32
};
