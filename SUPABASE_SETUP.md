# Supabase setup for Vigorly

The mobile app now uses Supabase Auth for email/password accounts and Supabase Postgres for account-scoped fitness storage. Passwords are owned by Supabase Auth; they are never written into `user_app_data`.

## 1. Create and configure the project

1. Create a Supabase project.
2. In **Project Settings → API**, copy the project URL and publishable key into `mobile/.env`:

   ```dotenv
   EXPO_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
   EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY=sb_publishable_YOUR_KEY
   ```

3. Never put the `service_role` key in the mobile app. Expo public variables are bundled into the client.

## 2. Create the protected storage table

Run [`supabase/migrations/202608020001_auth_and_user_storage.sql`](supabase/migrations/202608020001_auth_and_user_storage.sql) in the Supabase SQL editor or with the Supabase CLI migration workflow.

The migration creates:

- one `user_app_data` row per authenticated user;
- row-level security policies tied to `auth.uid()`;
- an optimistic-concurrency save function so two devices merge instead of silently overwriting;
- a self-service account deletion function that also cascades through cloud fitness data.

## 3. Configure email links

In **Authentication → URL Configuration**, add these redirect URLs:

```text
vigorly://auth
vigorly://auth-reset
```

Keep email/password enabled in **Authentication → Providers**. Email confirmation is recommended for production. Configure custom SMTP before launch so confirmation and recovery mail comes from the Vigorly domain and has production-grade deliverability.

## 4. Run and verify

Restart Expo after changing `mobile/.env`; environment variables are embedded at bundle time. Verify this sequence on a physical device:

1. Finish onboarding and create an account.
2. Open the confirmation email on the same phone and sign in.
3. Log a weight or meal, tap **Sync now**, then sign in on a second device.
4. Request **Forgot password?**, open the email link, and save a new password.
5. Delete the account and confirm the Auth user and `user_app_data` row are both gone.

Guest mode remains available as an explicit local-only choice. When a guest later signs in, existing device data is merged into that account’s cloud vault.
