import { getDatabase } from './attendanceStore';

type OutboxRow = {
  id: string;
  payload: string;
  idempotency_key: string;
  attempt_count: number;
};

type PushOptions = {
  endpoint?: string;
  fetchImpl?: typeof fetch;
};

const DEFAULT_ENDPOINT = 'https://api.datalake.com/sync/push';
const BATCH_SIZE = 10;
let syncLock: Promise<void> = Promise.resolve();

async function queryAll<T>(db: any, sql: string, params: unknown[] = []): Promise<T[]> {
  const [result] = await db.executeSql(sql, params);
  const rows: T[] = [];

  for (let index = 0; index < result.rows.length; index += 1) {
    rows.push(result.rows.item(index) as T);
  }

  return rows;
}

async function scheduleRetry(db: any, id: string, attemptCount: number) {
  const baseInterval = 10_000;
  const maximumDelay = 600_000;
  const backoffDelay = Math.min(maximumDelay, baseInterval * Math.pow(2, Math.max(0, attemptCount)));
  const nextAttempt = backoffDelay;
  const nextRetryAt = Date.now() + nextAttempt;

  await db.executeSql(
    `UPDATE sync_outbox
     SET attempt_count = attempt_count + 1,
         next_retry_at = ?,
         status = 'pending'
     WHERE id = ?;`,
    [nextRetryAt, id],
  );
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

async function pushLocalChanges(db: any, options: PushOptions = {}) {
  const outboxBatch = await queryAll<OutboxRow>(
    db,
    `SELECT id, payload, idempotency_key, attempt_count
     FROM sync_outbox
     WHERE status = 'pending' AND next_retry_at <= ?
     ORDER BY created_at ASC
     LIMIT ?;`,
    [Date.now(), BATCH_SIZE],
  );

  for (const item of outboxBatch) {
    try {
      const response = await (options.fetchImpl ?? fetch)(options.endpoint ?? DEFAULT_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Idempotency-Key': item.idempotency_key,
        },
        body: item.payload,
      });

      if (response.ok) {
        let parsedPayload: { id?: string } | null = null;

        try {
          parsedPayload = JSON.parse(item.payload) as { id?: string };
        } catch {
          parsedPayload = null;
        }

        await runInTransaction(db, async () => {
          await db.executeSql(`UPDATE sync_outbox SET status = 'done' WHERE id = ?;`, [item.id]);

          if (parsedPayload?.id) {
            await db.executeSql(`DELETE FROM attendance_journal WHERE id = ?;`, [parsedPayload.id]);
          }

          await db.executeSql(`DELETE FROM sync_outbox WHERE id = ?;`, [item.id]);
        });
      } else {
        await scheduleRetry(db, item.id, item.attempt_count);
      }
    } catch {
      await scheduleRetry(db, item.id, item.attempt_count);
    }
  }
}

export async function pushLocalChangesToServer(options: PushOptions = {}) {
  syncLock = syncLock.catch(() => undefined).then(async () => {
    const db = await getDatabase();
    await pushLocalChanges(db, options);
  });

  return syncLock;
}