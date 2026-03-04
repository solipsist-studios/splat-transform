/**
 * Writer for the .omg4 binary format.
 *
 * Converts an OMG4 compressed 4DGS checkpoint (.xz) to the web-friendly .omg4
 * binary format by baking per-frame Gaussian attributes via CPU-side MLP
 * inference.
 *
 * @see {@link convertOmg4} for the core conversion logic.
 * @ignore
 */

import { type FileSystem } from '../io/write';
import { convertOmg4, type ConvertOmg4Options } from '../omg4/convert-omg4';

/**
 * Options for writing an .omg4 file.
 */
type WriteOmg4Options = {
    /** Path to the output .omg4 file. */
    filename: string;
    /** Raw bytes of the .xz compressed checkpoint. */
    xzData: Uint8Array;
    /** Number of output frames. Default: 30 */
    numFrames?: number;
    /** Frames per second. Default: 24 */
    fps?: number;
    /** Time duration minimum. Default: -0.5 */
    timeMin?: number;
    /** Time duration maximum. Default: 0.5 */
    timeMax?: number;
};

/**
 * Convert an OMG4 compressed checkpoint and write the resulting .omg4 file.
 *
 * @param options - Conversion and output options.
 * @param fs - File system abstraction for writing the output.
 */
const writeOmg4 = async (options: WriteOmg4Options, fs: FileSystem) => {
    const convOpts: Partial<ConvertOmg4Options> = {};
    if (options.numFrames !== undefined) convOpts.numFrames = options.numFrames;
    if (options.fps !== undefined) convOpts.fps = options.fps;
    if (options.timeMin !== undefined) convOpts.timeMin = options.timeMin;
    if (options.timeMax !== undefined) convOpts.timeMax = options.timeMax;

    const omg4Data = await convertOmg4(options.xzData, convOpts);

    const writer = await fs.createWriter(options.filename);
    await writer.write(omg4Data);
    await writer.close();
};

export { writeOmg4, type WriteOmg4Options };
