const SQLite = require('react-native-sqlite-storage');

import { initializeDatabase } from './database';
import { computeCosineSimilarity } from './embedding_matcher';

SQLite.enablePromise(true);

export type AttendanceRow = {
  id: string;
  personnel_id: string;
  embedding: Float32Array;
  timestamp: number;
  synced: number;
};

export type StoredEmbeddingRow = {
  id: string;
  personnel_id: string;
  embedding: Float32Array;
  timestamp: number;
  synced: number;
};

const DB_NAME = 'attendance.db';
const TABLE_NAME = 'attendances';
const MATCH_THRESHOLD = 0.75;

let dbPromise: Promise<any> | null = null;
let schemaPromise: Promise<void> | null = null;
let syncLock: Promise<void> = Promise.resolve();

function toBlob(embedding: Float32Array): ArrayBuffer {
  return embedding.buffer.slice(
    embedding.byteOffset,
    embedding.byteOffset + embedding.byteLength,
  ) as ArrayBuffer;
}

function fromBlob(blob: ArrayBuffer | number[] | string | null | undefined): Float32Array {
  if (!blob) {
    return new Float32Array();
  }

  if (blob instanceof ArrayBuffer) {
    return new Float32Array(blob.slice(0));
  }

  if (Array.isArray(blob)) {
    return new Float32Array(blob);
  }

  if (typeof blob === 'string') {
    const bufferCtor = (globalThis as any).Buffer;
    if (bufferCtor?.from) {
      const bytes = bufferCtor.from(blob, 'base64');
      return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    }

    return new Float32Array();
  }

  return new Float32Array();
}

async function openDatabase() {
  if (!dbPromise) {
    dbPromise = SQLite.openDatabase({
      name: DB_NAME,
      location: 'default',
    });
  }

  return dbPromise;
}

async function ensureSchema() {
  if (!schemaPromise) {
    schemaPromise = (async () => {
      const db = await openDatabase();
      await initializeDatabase(db);
      await db.executeSql(
        `CREATE TABLE IF NOT EXISTS ${TABLE_NAME} (
          id TEXT PRIMARY KEY,
          personnel_id TEXT,
          embedding BLOB,
          timestamp INTEGER,
          synced INTEGER DEFAULT 0
        );`,
      );
    })();
  }

  return schemaPromise;
}

export async function getDatabase() {
  await ensureSchema();
  return openDatabase();
}

export async function initAttendanceDb() {
  await ensureSchema();
}

export async function saveAttendance(row: AttendanceRow) {
  const db = await getDatabase();
  await db.executeSql(
    `INSERT OR REPLACE INTO ${TABLE_NAME} (id, personnel_id, embedding, timestamp, synced)
     VALUES (?, ?, ?, ?, ?);`,
    [row.id, row.personnel_id, toBlob(row.embedding), row.timestamp, row.synced],
  );
}

export async function getUnsynced(): Promise<AttendanceRow[]> {
  const db = await getDatabase();
  const [result] = await db.executeSql(
    `SELECT id, personnel_id, embedding, timestamp, synced
     FROM ${TABLE_NAME}
     WHERE synced = 0
     ORDER BY timestamp ASC;`,
  );

  const rows: AttendanceRow[] = [];
  for (let index = 0; index < result.rows.length; index += 1) {
    const item = result.rows.item(index);
    rows.push({
      id: item.id,
      personnel_id: item.personnel_id,
      embedding: fromBlob(item.embedding),
      timestamp: item.timestamp,
      synced: item.synced,
    });
  }

  return rows;
}

export async function getUnsyncedCount(): Promise<number> {
  const db = await getDatabase();
  const [result] = await db.executeSql(
    `SELECT COUNT(*) AS count
     FROM ${TABLE_NAME}
     WHERE synced = 0;`,
  );

  const item = result.rows.item(0) as { count?: number } | undefined;
  return Number(item?.count ?? 0);
}

export async function markPurged(ids: string[]) {
  if (ids.length === 0) {
    return;
  }

  const db = await getDatabase();
  const placeholders = ids.map(() => '?').join(', ');
  await db.executeSql(
    `DELETE FROM ${TABLE_NAME} WHERE id IN (${placeholders});`,
    ids,
  );
}

export async function loadStoredEmbeddings(): Promise<StoredEmbeddingRow[]> {
  const db = await getDatabase();
  const [result] = await db.executeSql(
    `SELECT id, personnel_id, embedding, timestamp, synced
     FROM ${TABLE_NAME}
     ORDER BY timestamp DESC;`,
  );

  const rows: StoredEmbeddingRow[] = [];
  for (let index = 0; index < result.rows.length; index += 1) {
    const item = result.rows.item(index);
    rows.push({
      id: item.id,
      personnel_id: item.personnel_id,
      embedding: fromBlob(item.embedding),
      timestamp: item.timestamp,
      synced: item.synced,
    });
  }

  return rows;
}

export function cosineSim(a: ArrayLike<number>, b: ArrayLike<number>) {
  const vectorA = a instanceof Float32Array ? a : new Float32Array(Array.from(a));
  const vectorB = b instanceof Float32Array ? b : new Float32Array(Array.from(b));

  if (vectorA.length !== vectorB.length) {
    throw new Error(
      `Dimension mismatch: Vector A (${vectorA.length}) versus Vector B (${vectorB.length})`,
    );
  }

  return computeCosineSimilarity(vectorA, vectorB);
}

export async function authenticateEmbedding(liveEmbedding: ArrayLike<number>) {
  const storedRows = await loadStoredEmbeddings();
  let bestMatch: { personnelId: string; similarity: number } | null = null;

  for (const row of storedRows) {
    const similarity = cosineSim(row.embedding, liveEmbedding);
    if (!bestMatch || similarity > bestMatch.similarity) {
      bestMatch = {
        personnelId: row.personnel_id,
        similarity,
      };
    }
  }

  if (!bestMatch || bestMatch.similarity <= MATCH_THRESHOLD) {
    return null;
  }

  return bestMatch;
}

export async function syncAndPurge(uploadToS3: (row: AttendanceRow) => Promise<void>) {
  syncLock = syncLock.catch(() => undefined).then(async () => {
    const rows = await getUnsynced();

    for (const row of rows) {
      await uploadToS3(row);
    }

    await markPurged(rows.map(row => row.id));
  });

  return syncLock;
}
