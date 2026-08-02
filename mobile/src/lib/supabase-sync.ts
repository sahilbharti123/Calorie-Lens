import { requireSupabase } from '@/src/lib/supabase';
import type { AppData } from '@/src/types';

export type CloudSnapshot = {
  payload: Partial<AppData>;
  version: number;
  updatedAt: string;
};

export class CloudConflictError extends Error {
  snapshot: CloudSnapshot;

  constructor(snapshot: CloudSnapshot) {
    super('Cloud data changed on another device. Vigorly is merging it now.');
    this.name = 'CloudConflictError';
    this.snapshot = snapshot;
  }
}

type UserAppDataRow = {
  payload: Partial<AppData> | null;
  version: number | string;
  updated_at: string;
};

function snapshot(row?: UserAppDataRow | null): CloudSnapshot {
  return {
    payload: row?.payload ?? {},
    version: Number(row?.version ?? 0),
    updatedAt: row?.updated_at ?? '',
  };
}

export async function readCloudSnapshot(userId: string): Promise<CloudSnapshot> {
  const client = requireSupabase();
  const { data, error } = await client
    .from('user_app_data')
    .select('payload, version, updated_at')
    .eq('user_id', userId)
    .maybeSingle();
  if (error) throw new Error(error.message);
  return snapshot(data as UserAppDataRow | null);
}

type SaveResult = UserAppDataRow & { saved: boolean };

export async function writeCloudSnapshot(
  payload: AppData,
  expectedVersion: number,
): Promise<CloudSnapshot> {
  const client = requireSupabase();
  const { data, error } = await client.rpc('save_user_app_data', {
    p_payload: payload,
    p_expected_version: expectedVersion,
  });
  if (error) throw new Error(error.message);
  const row = (Array.isArray(data) ? data[0] : data) as SaveResult | null;
  if (!row) throw new Error('Supabase did not return the saved fitness vault.');
  const next = snapshot(row);
  if (!row.saved) throw new CloudConflictError(next);
  return next;
}
