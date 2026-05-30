export type SQLiteDatabaseLike = {
  execAsync?(sql: string): Promise<void>;
  executeSql?(sql: string, params?: unknown[]): Promise<unknown>;
};

async function runSql(db: SQLiteDatabaseLike, sql: string) {
  if (typeof db.execAsync === 'function') {
    await db.execAsync(sql);
    return;
  }

  if (typeof db.executeSql === 'function') {
    await db.executeSql(sql);
    return;
  }

  throw new Error('Unsupported SQLite database object');
}

async function runStatements(db: SQLiteDatabaseLike, statements: string[]) {
  for (const statement of statements) {
    await runSql(db, statement);
  }
}

export async function initializeDatabase(db: SQLiteDatabaseLike): Promise<void> {
  await runStatements(db, [
    `PRAGMA journal_mode = WAL;`,
    `PRAGMA synchronous = NORMAL;`,
    `CREATE TABLE IF NOT EXISTS employees (
      id TEXT PRIMARY KEY NOT NULL,
      name TEXT NOT NULL,
      employee_code TEXT UNIQUE NOT NULL,
      embedding_blob BLOB NOT NULL,
      updated_at INTEGER NOT NULL,
      deleted_at INTEGER DEFAULT NULL
    );`,
    `CREATE TABLE IF NOT EXISTS attendance_journal (
      id TEXT PRIMARY KEY NOT NULL,
      employee_id TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      liveness_score REAL NOT NULL,
      matching_score REAL NOT NULL,
      sync_status TEXT DEFAULT 'pending',
      updated_at INTEGER NOT NULL,
      deleted_at INTEGER DEFAULT NULL,
      FOREIGN KEY(employee_id) REFERENCES employees(id)
    );`,
    `CREATE TABLE IF NOT EXISTS sync_outbox (
      id TEXT PRIMARY KEY NOT NULL,
      transaction_type TEXT NOT NULL,
      payload TEXT NOT NULL,
      idempotency_key TEXT UNIQUE NOT NULL,
      status TEXT DEFAULT 'pending',
      attempt_count INTEGER DEFAULT 0,
      next_retry_at INTEGER DEFAULT 0,
      created_at INTEGER NOT NULL
    );`,
    `CREATE INDEX IF NOT EXISTS idx_employees_updated ON employees(updated_at);`,
    `CREATE INDEX IF NOT EXISTS idx_journal_sync_status ON attendance_journal(sync_status);`,
    `CREATE INDEX IF NOT EXISTS idx_outbox_scheduler ON sync_outbox(status, next_retry_at);`,
  ]);
}