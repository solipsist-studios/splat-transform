/**
 * Clip-level scalars for a spacetime splat file.
 *
 * These describe the clip rather than any splat, so the interchange PLY carries
 * them in `comment sogst.*` header lines rather than as columns.
 */
type SogstClip = {
    /** Clip start time in seconds. */
    timeMin: number;
    /** Clip end time in seconds. */
    timeMax: number;
    /** Advisory playback rate, for UI frame counters and scrub granularity. */
    fps: number;
    /** Advisory motion degree. The presence of ax/ay/az is what actually decides. */
    motionDegree?: number;
    /** Screen-space 2D-covariance multiplier a player applies when rasterising. */
    cov2dScale?: [number, number];
};

const parseFloats = (value: string) => {
    return value.split(/\s+/).filter(v => v.length > 0).map(Number);
};

/**
 * Extracts the `sogst.*` clip scalars from a PLY file's comment lines.
 *
 * Unknown `sogst.*` keys are ignored rather than rejected, so a producer can
 * add advisory keys without breaking older encoders. Malformed values for keys
 * we do understand are treated as absent, which surfaces as a missing-scalar
 * error rather than a silently wrong clip.
 *
 * @param comments - PLY comment lines, with the leading `comment ` stripped.
 * @returns The scalars that were present and well-formed.
 *
 * @example
 * ```ts
 * parseSogstComments(['sogst.time_min 0.0', 'sogst.fps 24']);
 * // { timeMin: 0, fps: 24 }
 * ```
 */
const parseSogstComments = (comments: string[] = []): Partial<SogstClip> => {
    const result: Partial<SogstClip> = {};

    for (const comment of comments) {
        const trimmed = comment.trim();
        if (!trimmed.startsWith('sogst.')) {
            continue;
        }

        const separator = trimmed.search(/\s/);
        if (separator < 0) {
            continue;
        }

        const key = trimmed.slice('sogst.'.length, separator);
        const values = parseFloats(trimmed.slice(separator + 1));

        if (values.some(isNaN)) {
            continue;
        }

        switch (key) {
            case 'time_min':
                if (values.length === 1) result.timeMin = values[0];
                break;
            case 'time_max':
                if (values.length === 1) result.timeMax = values[0];
                break;
            case 'fps':
                if (values.length === 1) result.fps = values[0];
                break;
            case 'motion_degree':
                if (values.length === 1) result.motionDegree = values[0];
                break;
            case 'cov2d_scale':
                if (values.length === 2) result.cov2dScale = [values[0], values[1]];
                break;
            default:
                // unknown sogst.* keys are advisory — ignore, never reject
                break;
        }
    }

    return result;
};

/**
 * Validates that every required clip scalar is present and usable.
 *
 * There is deliberately no default. An assumed frame rate renders perfectly and
 * plays at the wrong speed, which is the hardest class of bug to notice, so a
 * missing scalar is an error rather than a fallback.
 *
 * @param clip - Scalars gathered from PLY comments and/or explicit overrides.
 * @returns The complete clip.
 * @throws Error naming every scalar that is missing or out of range.
 */
const requireSogstClip = (clip: Partial<SogstClip>): SogstClip => {
    const problems: string[] = [];

    for (const key of ['timeMin', 'timeMax', 'fps'] as const) {
        if (clip[key] === undefined) {
            problems.push(`${key} is missing`);
        } else if (!isFinite(clip[key])) {
            problems.push(`${key} is not a finite number (${clip[key]})`);
        }
    }

    if (problems.length === 0) {
        if (clip.timeMax < clip.timeMin) {
            problems.push(`timeMax (${clip.timeMax}) precedes timeMin (${clip.timeMin})`);
        }
        if (clip.fps <= 0) {
            problems.push(`fps must be > 0 (got ${clip.fps})`);
        }
    }

    if (problems.length > 0) {
        throw new Error(`sogst output requires clip scalars: ${problems.join(', ')}. Expected 'comment sogst.time_min', 'comment sogst.time_max' and 'comment sogst.fps' in the input PLY header, or explicit overrides.`);
    }

    return clip as SogstClip;
};

export { parseSogstComments, requireSogstClip, type SogstClip };
