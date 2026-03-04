/**
 * Huffman decoder compatible with the Python `dahuffman` library's code table format.
 *
 * A dahuffman code_table is a dictionary mapping each symbol to a tuple
 * (bit_length, code_value).  This module reads a bitstream produced by
 * dahuffman.HuffmanCodec.encode() and returns the decoded symbol sequence.
 *
 * @ignore
 */

/**
 * A Huffman code table: symbol → [bitLength, codeValue].
 *
 * The special sentinel symbol _EOF (represented as null) marks end-of-stream.
 */
type HuffmanTable = Map<number | null, [number, number]>;

/**
 * Parse a code table from the pickleparser dict format into a Map.
 *
 * The pickle dict has integer keys (symbols) mapping to [bitLength, code] tuples.
 * The _EOF sentinel is stored under key `null` or a special marker.
 * @param obj - The pickleparser dict object containing the code table.
 * @returns A HuffmanTable Map of symbol to [bitLength, codeValue] tuples.
 * @ignore
 */
const parseHuffmanTable = (obj: Record<string | number, any>): HuffmanTable => {
    const table = new Map<number | null, [number, number]>();
    for (const key of Object.keys(obj)) {
        const value = obj[key];
        if (!Array.isArray(value) || value.length !== 2) {
            continue;
        }
        const bitLen = value[0] as number;
        const code = value[1] as number;

        if (key === 'null' || key === 'undefined') {
            // _EOF sentinel
            table.set(null, [bitLen, code]);
        } else {
            table.set(Number(key), [bitLen, code]);
        }
    }
    return table;
};

/**
 * Build a decode lookup: for each (bitLength, code) → symbol.
 * @param table - The HuffmanTable to build the lookup from.
 * @returns A Map from "bitLen:code" string keys to symbol values.
 * @ignore
 */
const buildDecodeLookup = (table: HuffmanTable): Map<string, number | null> => {
    const lookup = new Map<string, number | null>();
    for (const [symbol, [bitLen, code]] of table) {
        const key = `${bitLen}:${code}`;
        lookup.set(key, symbol);
    }
    return lookup;
};

/**
 * Decode a Huffman-encoded byte stream using the given code table.
 *
 * @param encoded - The Huffman-encoded bytes.
 * @param codeTable - Symbol → [bitLength, codeValue] mapping.
 * @returns Array of decoded uint16 symbol values.
 */
const huffmanDecode = (encoded: Uint8Array, codeTable: Record<string | number, any>): Uint16Array => {
    const table = parseHuffmanTable(codeTable);

    // Find the maximum bit length for early termination
    let maxBitLen = 0;
    for (const [, [bitLen]] of table) {
        if (bitLen > maxBitLen) maxBitLen = bitLen;
    }

    const lookup = buildDecodeLookup(table);

    const decoded: number[] = [];
    let bitPos = 0;
    const totalBits = encoded.length * 8;

    while (bitPos < totalBits) {
        let code = 0;
        let found = false;

        for (let bitLen = 1; bitLen <= maxBitLen && bitPos + bitLen <= totalBits; bitLen++) {
            // Read the next bit
            const byteIdx = Math.floor((bitPos + bitLen - 1) / 8);
            const bitIdx = 7 - ((bitPos + bitLen - 1) % 8);
            const bit = (encoded[byteIdx] >> bitIdx) & 1;
            code = (code << 1) | bit;

            const key = `${bitLen}:${code}`;
            const symbol = lookup.get(key);
            if (symbol !== undefined) {
                if (symbol === null) {
                    // EOF sentinel - stop decoding
                    return new Uint16Array(decoded);
                }
                decoded.push(symbol);
                bitPos += bitLen;
                found = true;
                break;
            }
        }

        if (!found) {
            // No valid code found - we've consumed all data
            break;
        }
    }

    return new Uint16Array(decoded);
};

export { huffmanDecode, parseHuffmanTable, type HuffmanTable };
