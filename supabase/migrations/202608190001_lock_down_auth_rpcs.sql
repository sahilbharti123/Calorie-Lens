-- Only signed-in users may call account-scoped vault and deletion RPCs.
--
-- Earlier dashboard grants left `anon` with EXECUTE even though both
-- functions reject a missing auth.uid(). Removing the grant closes the RPC at
-- the database boundary as well as inside the function body.

revoke execute on function public.save_user_app_data(jsonb, bigint) from public, anon;
revoke execute on function public.delete_own_account() from public, anon;

grant execute on function public.save_user_app_data(jsonb, bigint) to authenticated;
grant execute on function public.delete_own_account() to authenticated;
