import { getDatabase } from './attendanceStore';

type EmployeeRow = {
  id: string;
  name: string;
  employee_code: string;
  embedding_blob: ArrayBuffer | number[] | string | null;
  updated_at: number;
  deleted_at: number | null;
};

type RemoteSyncResponse = {
  changes?: EmployeeRow[];
};

type SyncOptions = {
  endpoint?: string;
  fetchImpl?: typeof fetch;
};

const DEFAULT_ENDPOINT = 'https://api.datalake.com/sync/pull';
let syncLock: Promise<void> = Promise.resolve();

async function queryFirst<T>(db: any, sql: string, params: unknown[] = []): Promise<T | null> {
  const [result] = await db.executeSql(sql, params);

  if (!result.rows.length) {
    return null;
  }

  return result.rows.item(0) as T;
}

async function runInTransaction(db: any, task: () => Promise<void>) {
  await db.executeSql('BEGIN TRANSACTION;');

  try {
    await task();
    await db.executeSql('COMMIT;');
  } catch (error) {
    try {
      await db.executeSql('ROLLBACK;');
    } catch {
      // Ignore rollback failures and surface the original error.
    }

    throw error;
  }
}

async function pullRemoteChanges(db: any, options: SyncOptions = {}) {
  const cursorRow = await queryFirst<{ last_cursor: number }>(
    db,
    `SELECT IFNULL(MAX(updated_at), 0) AS last_cursor FROM employees;`,
  );
  const localCursor = cursorRow?.last_cursor ?? 0;

  const response = await (options.fetchImpl ?? fetch)(options.endpoint ?? DEFAULT_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cursor: localCursor }),
  });

  if (!response.ok) {
    throw new Error(`Pull sync failed with status ${response.status}`);
  }

  const data = (await response.json()) as RemoteSyncResponse;
  const changes = Array.isArray(data.changes) ? data.changes : [];

  await runInTransaction(db, async () => {
    for (const remoteUser of changes) {
      const localRecord = await queryFirst<{ updated_at: number }>(
        db,
        `SELECT updated_at FROM employees WHERE id = ?;`,
        [remoteUser.id],
      );

      if (!localRecord) {
        await db.executeSql(
          `INSERT INTO employees (id, name, employee_code, embedding_blob, updated_at, deleted_at)
           VALUES (?, ?, ?, ?, ?, ?);`,
          [
            remoteUser.id,
            remoteUser.name,
            remoteUser.employee_code,
            remoteUser.embedding_blob,
            remoteUser.updated_at,
            remoteUser.deleted_at,
          ],
        );
        continue;
      }

      if (remoteUser.updated_at > localRecord.updated_at) {
        await db.executeSql(
          `UPDATE employees
           SET name = ?, employee_code = ?, embedding_blob = ?, updated_at = ?, deleted_at = ?
           WHERE id = ?;`,
          [
            remoteUser.name,
            remoteUser.employee_code,
            remoteUser.embedding_blob,
            remoteUser.updated_at,
            remoteUser.deleted_at,
            remoteUser.id,
          ],
        );
      }
    }
  });
}

export async function syncEmployeePull(options: SyncOptions = {}) {
  syncLock = syncLock.catch(() => undefined).then(async () => {
    const db = await getDatabase();
    await pullRemoteChanges(db, options);
  });

  return syncLock;
}