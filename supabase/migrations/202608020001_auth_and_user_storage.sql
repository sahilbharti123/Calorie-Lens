-- Vigorly account-scoped cloud vault.
-- Supabase Auth owns credentials; this table stores fitness application data only.

create table if not exists public.user_app_data (
  user_id uuid primary key references auth.users (id) on delete cascade,
  payload jsonb not null default '{}'::jsonb,
  version bigint not null default 1 check (version > 0),
  updated_at timestamptz not null default now()
);

alter table public.user_app_data enable row level security;

revoke all on table public.user_app_data from anon;
grant select, insert, update, delete on table public.user_app_data to authenticated;

drop policy if exists "Users can read their own app data" on public.user_app_data;
create policy "Users can read their own app data"
on public.user_app_data for select
to authenticated
using ((select auth.uid()) = user_id);

drop policy if exists "Users can insert their own app data" on public.user_app_data;
create policy "Users can insert their own app data"
on public.user_app_data for insert
to authenticated
with check ((select auth.uid()) = user_id);

drop policy if exists "Users can update their own app data" on public.user_app_data;
create policy "Users can update their own app data"
on public.user_app_data for update
to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

drop policy if exists "Users can delete their own app data" on public.user_app_data;
create policy "Users can delete their own app data"
on public.user_app_data for delete
to authenticated
using ((select auth.uid()) = user_id);

-- Optimistic concurrency prevents two devices from silently overwriting each
-- other. A stale caller receives the latest row and merges before retrying.
create or replace function public.save_user_app_data(
  p_payload jsonb,
  p_expected_version bigint
)
returns table (saved boolean, payload jsonb, version bigint, updated_at timestamptz)
language plpgsql
security invoker
set search_path = ''
as $$
declare
  caller_id uuid := (select auth.uid());
begin
  if caller_id is null then
    raise exception 'Authentication required';
  end if;

  if p_expected_version = 0 then
    insert into public.user_app_data (user_id, payload, version, updated_at)
    values (caller_id, p_payload, 1, now())
    on conflict (user_id) do nothing
    returning true, user_app_data.payload, user_app_data.version, user_app_data.updated_at
    into saved, payload, version, updated_at;

    if found then
      return next;
      return;
    end if;
  end if;

  update public.user_app_data
  set payload = p_payload,
      version = public.user_app_data.version + 1,
      updated_at = now()
  where user_id = caller_id
    and public.user_app_data.version = p_expected_version
  returning true, user_app_data.payload, user_app_data.version, user_app_data.updated_at
  into saved, payload, version, updated_at;

  if found then
    return next;
    return;
  end if;

  return query
  select false, current.payload, current.version, current.updated_at
  from public.user_app_data as current
  where current.user_id = caller_id;
end;
$$;

revoke all on function public.save_user_app_data(jsonb, bigint) from public;
grant execute on function public.save_user_app_data(jsonb, bigint) to authenticated;

-- Authenticated users can remove their own Auth row. The foreign key cascade
-- removes the associated fitness vault. This function must be created by the
-- project owner in the Supabase SQL editor/migration runner.
create or replace function public.delete_own_account()
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  if (select auth.uid()) is null then
    raise exception 'Authentication required';
  end if;
  delete from auth.users where id = (select auth.uid());
end;
$$;

revoke all on function public.delete_own_account() from public;
grant execute on function public.delete_own_account() to authenticated;
