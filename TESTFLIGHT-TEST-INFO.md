# TestFlight → Test Information

Paste-ready copy for the App Store Connect form, plus the Supabase settings that
decide whether a tester can actually create an account.

---

## Beta App Description

> Vigorly turns what you say into a tracked day. Tell it "2 rotis and a bowl of dal" or "500 ml of beer" — typed or spoken, in English or Hinglish — and it works out the portion, the calories and the macros, then shows you the range it is actually confident in instead of a single number pretending to be exact.
>
> WHAT TO TRY
>
> Log meals by voice and by typing, in whatever words come naturally. Fragments are fine: "2 rotis and dal", "paneer 100 g", "chai", "a packet of chips". Then check the numbers — every entry shows its range, where the figure came from and the assumptions behind it, and you can correct any of it before it saves.
>
> Teach it a food it does not know. Give it the calories once and it remembers. Settings has a "Foods you've taught" screen where anything it learned can be renamed, corrected or removed.
>
> Also worth exercising: water, weight, steps, sleep, and workouts with routines and sessions.
>
> ACCOUNTS
>
> You do not need one. Tap "Use guest mode on this device" and the whole app works, with your data encrypted on your phone and nowhere else. Creating an account backs it up and syncs it across devices — and if you start in guest mode and sign up later, what you already logged merges into your account rather than being lost.
>
> KNOWN LIMITS IN THIS BUILD
>
> The Coach tab is a placeholder and says so. Steps and sleep can be logged but not yet corrected once entered. Food figures are estimates: around thirty are verified USDA records and show their FDC ID, and the rest are typical published figures for that kind of food, carrying a deliberately wide range. Vigorly is a fitness tracker, not medical advice.
>
> WHAT I MOST WANT TO HEAR ABOUT
>
> Anything you said that it misread, logged as the wrong food, or refused to log — please include the exact words you used, because that is what I test against. A number that just looks wrong is as useful as something that fails outright.

## Feedback Email

    sahil.bharti97@gmail.com

## Contact Information

| Field        | Value                    |
| ------------ | ------------------------ |
| First Name   | Sahil                    |
| Last Name    | Bharti                   |
| Phone number | *(yours — I don't have it)* |
| Email        | sahil.bharti97@gmail.com |

## Sign-In Information

**Uncheck "Sign-in required".**

It is genuinely not required: "Use guest mode on this device" gives full access
to every feature except cloud sync. Leaving the box ticked obliges you to hand
Apple a working demo account, which you said you did not want to create yet.

If you later want App Review to see sync working, tick it again and give them a
real account — but that is a decision for the public submission, not for
TestFlight.

---

# Letting testers create their own account

The sign-up flow already works. The blocker is on the Supabase side, and it is
worse than a rate limit.

**Supabase's built-in email service only delivers to addresses belonging to
members of your Supabase organisation. Everyone else gets "Email address not
authorized."** It is also capped at two messages an hour project-wide, and
Supabase states plainly that it is not meant for production and carries no
delivery SLA.

So as things stand, if "Confirm email" is on, a tester outside your Supabase org
signs up, never receives the confirmation link, and can never sign in — and
cannot register again either, because the address is already taken. A silent
dead end, for every one of them.

## Two ways out

**Same-day unblock — turn confirmation off.**
Authentication → Sign In / Providers → Email → turn *Confirm email* off. Sign-up
then returns a session immediately and the tester is straight into the app; no
email is involved at any point. The app already handles this correctly, so no
build change is needed. Password reset will still be broken for outside testers,
because that genuinely needs an email.

**The real fix — custom SMTP.**
Project Settings → Authentication → SMTP Settings, pointed at Resend, Postmark,
SendGrid or SES. This removes the org-only restriction and the two-an-hour cap,
and makes password reset work. You need it before the public launch regardless,
so doing it now costs nothing extra.

## Also check, either way

- Authentication → Sign In / Providers → Email → **Allow new users to sign up**
  must be on, or registration is refused outright.
- Authentication → URL Configuration → Redirect URLs must include
  `vigorly://auth` and `vigorly://auth-reset`. Without them the confirmation and
  password-reset links bounce to the site URL instead of opening the app.

## What changed in the app

A tester who never received the confirmation email had no way forward: signing in
returned "Confirm your email before signing in" and that was the end of it. That
message now brings up a **"Send the confirmation email again"** button, which
re-sends the link to the same address.
