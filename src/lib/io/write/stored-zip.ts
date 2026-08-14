import { Crc } from './crc';
import { type Writer } from './file-system';

/**
 * A complete archive entry. Unlike ZipFileSystem, the payload is known up
 * front, which is what lets the sizes go in the local header.
 */
type StoredZipEntry = {
    name: string;
    data: Uint8Array;
};

// Local file header: signature through to the (empty) extra field.
const LOCAL_HEADER_SIZE = 30;

// Central directory record, excluding the filename.
const CENTRAL_HEADER_SIZE = 46;

// End of central directory record.
const EOCD_SIZE = 22;

/**
 * Writes a ZIP archive with every entry STORED and no extra fields or data
 * descriptors.
 *
 * ZipFileSystem streams entries whose size is not known until they finish, so
 * it sets the data-descriptor flag and appends 16 bytes after each payload.
 * That is fine for a bundle a player unpacks whole, but it defeats byte-range
 * access: the local header reports a compressed size of zero, so a reader
 * scanning forward cannot tell where the next entry starts. Formats that
 * publish analytic byte offsets need each local header to be exactly
 * `30 + len(name)` bytes with the real sizes inside it, which requires the
 * payload up front.
 *
 * @param writer - Destination stream. Left open; the caller closes it.
 * @param entries - Entries to write, in archive order.
 * @returns The absolute byte offset of each entry's local header, in order.
 */
const writeStoredZip = async (writer: Writer, entries: StoredZipEntry[]): Promise<number[]> => {
    const textEncoder = new TextEncoder();

    const date = new Date();
    const dosTime = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
    const dosDate = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();

    const names = entries.map(entry => textEncoder.encode(entry.name));
    const crcs = entries.map((entry) => {
        const crc = new Crc();
        crc.update(entry.data);
        return crc.value();
    });

    const headerOffsets: number[] = [];
    let offset = 0;

    for (let i = 0; i < entries.length; ++i) {
        const { data } = entries[i];
        const name = names[i];

        const header = new Uint8Array(LOCAL_HEADER_SIZE + name.length);
        const view = new DataView(header.buffer);

        view.setUint32(0, 0x04034b50, true);
        view.setUint16(4, 20, true);            // version needed to extract = 2.0
        view.setUint16(6, 0x800, true);         // utf-8 filename; sizes are in this header
        view.setUint16(8, 0, true);             // method = 0 (store)
        view.setUint16(10, dosTime, true);
        view.setUint16(12, dosDate, true);
        view.setUint32(14, crcs[i], true);
        view.setUint32(18, data.length, true);  // compressed size
        view.setUint32(22, data.length, true);  // uncompressed size
        view.setUint16(26, name.length, true);
        view.setUint16(28, 0, true);            // extra field length = 0
        header.set(name, LOCAL_HEADER_SIZE);

        headerOffsets.push(offset);

        await writer.write(header);
        await writer.write(data);

        offset += header.length + data.length;
    }

    // central directory
    const centralOffset = offset;

    for (let i = 0; i < entries.length; ++i) {
        const { data } = entries[i];
        const name = names[i];

        const record = new Uint8Array(CENTRAL_HEADER_SIZE + name.length);
        const view = new DataView(record.buffer);

        view.setUint32(0, 0x02014b50, true);
        view.setUint16(4, 20, true);            // version made by
        view.setUint16(6, 20, true);            // version needed to extract
        view.setUint16(8, 0x800, true);
        view.setUint16(10, 0, true);            // method = 0 (store)
        view.setUint16(12, dosTime, true);
        view.setUint16(14, dosDate, true);
        view.setUint32(16, crcs[i], true);
        view.setUint32(20, data.length, true);
        view.setUint32(24, data.length, true);
        view.setUint16(28, name.length, true);
        view.setUint32(42, headerOffsets[i], true);
        record.set(name, CENTRAL_HEADER_SIZE);

        await writer.write(record);

        offset += record.length;
    }

    const eocd = new Uint8Array(EOCD_SIZE);
    const eocdView = new DataView(eocd.buffer);
    eocdView.setUint32(0, 0x06054b50, true);
    eocdView.setUint16(8, entries.length, true);
    eocdView.setUint16(10, entries.length, true);
    eocdView.setUint32(12, offset - centralOffset, true);
    eocdView.setUint32(16, centralOffset, true);

    await writer.write(eocd);

    return headerOffsets;
};

export { writeStoredZip, type StoredZipEntry };
