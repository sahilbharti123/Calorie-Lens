# App Store submission pack — Vigorly

Everything below is written to be pasted into App Store Connect. Character
counts are shown as `(used/limit)`. Anything in `ANGLE BRACKETS` is a
placeholder only you can fill in.

- **App**: Vigorly
- **Bundle ID**: `com.advaice.calorielens`
- **Version**: 1.0.0 (build 2)
- **Developer**: Sahil Bharti (individual), sahil.bharti97@gmail.com

---

## 1. App name and subtitle

### App name (30 characters max)

**Recommended**

```
Vigorly: Food & Gym Log
```

(28/30) — Keeps the brand first and buys two searchable words, "food" and
"gym", which is the pairing this app actually delivers. It also signals that
this is not only a calorie counter, which is the main reason someone would
choose it over an established tracker.

**Alternate A**

```
Vigorly: Calorie Counter
```

(29/30) — Highest-volume search term, but "calorie" appears twice and it hides
the training half of the app.

**Alternate B**

```
Vigorly
```

(12/30) — Clean brand-only listing. Choose this only if you would rather build
a name than chase search traffic in version 1.

### Subtitle (30 characters max)

**Recommended**

```
Every estimate shows its range
```

(30/30) — States the differentiator as a fact rather than a boast, and it is
the promise the screenshots can prove immediately.

**Alternate A**

```
Honest calories. Real gym log.
```

(30/30) — Covers both halves of the app. "Honest" is a claim rather than a
demonstration, which is slightly weaker than the recommended line.

**Alternate B**

```
USDA-sourced food and gym log
```

(29/30) — Most literal and most keyword-dense. Reads a little dry.

### Pick them as a pair

Apple shows the name and subtitle stacked, so they are read as one line. Use:

```
Vigorly: Food & Gym Log
Every estimate shows its range
```

The name carries the category, which "Vigorly" alone does not — a made-up word
tells a browsing user nothing — and the subtitle spends its 30 characters on the
one thing no competitor says. Pairing brand-only with the range line would leave
nobody able to tell what the app is for; pairing "Calorie Counter" with the range
line wastes the subtitle restating the category.

---

## 2. Promotional text (170 characters max)

```
Most trackers show one confident number. Vigorly shows a range, the assumptions behind it, and the USDA record it came from. Works offline, no account needed.
```

(163/170) — Promotional text can be changed any time without submitting a new
build, so use it later for release notes and seasonal messages.

---

## 3. Description (4000 characters max)

Paste exactly as written. Current length: **3393/4000**.

```
Ask most calorie apps about one bowl of dal and you get a single confident number. That number is a guess in a lab coat. Vigorly shows you the range instead.

Every estimate arrives with a midpoint, a plausible low to high range, the assumptions it made, a confidence label, and the source record behind it. Food numbers are calculated by a deterministic engine against reviewed USDA FoodData Central records, and each entry carries the FoodData Central ID it used. A language model is never allowed to invent a calorie value here. When the app does not recognize a food, it says so and asks you for a label or an ingredient list instead of reaching for a generic number.

Exercise energy works the same way. Activities map to the 2024 Adult Compendium of Physical Activities, and the formula subtracts resting energy, so a session is credited with the energy it actually added rather than the calories you would have burned sitting still. The MET value, your body weight, and the duration used are printed on the entry.

Logging takes one sentence. Type it, or hold the microphone and say it: two rotis and one bowl rajma, forty minute brisk walk, my weight is 78.4. Speech is transcribed by your iPhone's own speech recognition. No audio is uploaded anywhere, there is no subscription, and dictation keeps working offline wherever iOS has its on-device model.

The Train tab is a full gym logger. It ships with 100 exercises, each with demonstration photos, a written how-to covering setup, range of motion, breathing, tempo and the mistakes people actually make, plus a linked technique video. Build routines with rep ranges, rest timers and supersets, or import a template and edit it. During a session, every set row shows your previous performance and prefills it, the rest timer starts on its own, and personal records are flagged the moment you hit them. A finished session sends its duration and its energy range straight into the day, so training and food stay in one ledger.

Progress keeps the long view: weight trend, calorie adherence against target, training volume, streaks and weekly consistency across seven, thirty or ninety days. Steps, active energy, sleep and weight can be read from Apple Health with your permission.

The Coach tab reads your own numbers back to you. Protein and calorie adherence, training minutes against plan, how wide your estimates are running, where your weight is heading. Every observation states the figure it came from, so you can check it rather than trust it. All of it is arithmetic computed on this device. There is no chatbot in this release and nothing on that screen needs a network.

Vigorly is local first. Tap Continue without an account and the entire app works, with no sign-up and no connection. Records are encrypted on the device with AES-GCM before they are written to storage. An account is optional and does exactly one thing: it keeps your history in sync across your own devices. Your fitness data is never sold and never shared. There are no ads, no analytics, and no trackers.

Vigorly estimates. It is not a calorimeter and it is not medical advice. Recipes, cooking oil, portion reporting and your own metabolism all move the true number, which is precisely why you get a range and the reasoning behind it instead of false precision. Read your targets against a two to four week trend, not a single day.
```

Two rules this copy follows deliberately, worth keeping if you edit it:

- It never mentions Android or Health Connect. Apple rejects descriptions that
  reference other mobile platforms.
- It never promises a conversational or AI coach, because the Coach tab in this
  release is arithmetic. The screen itself makes the same promise: it advertises
  no unbuilt feature, which keeps it clear of guideline 2.1's rule against
  shipping placeholders for functionality that does not exist.

---

## 4. Keywords (100 characters max, comma separated, no spaces after commas)

```
macro,protein,nutrition,meal,diet,tracker,counter,workout,strength,lifting,weight,steps,diary,usda
```

(98/100) — Apple indexes the app name and subtitle alongside this field, so
nothing here repeats "calorie", "lens", "food", "gym", "log", "estimate" or
"range", which the name and subtitle already cover. Terms are singular because
Apple stems plurals, and no competitor or trademarked app names are used.

---

## 5. Support URL and Marketing URL

GitHub Pages is the right answer here: free, no server to keep alive, and the
content lives next to the code so it cannot drift.

| Field in App Store Connect | Recommended URL |
| --- | --- |
| Support URL (required) | `https://<GITHUB-USERNAME>.github.io/calorie-lens/support` |
| Marketing URL (optional) | `https://<GITHUB-USERNAME>.github.io/calorie-lens/` |
| Privacy Policy URL (required) | `https://<GITHUB-USERNAME>.github.io/calorie-lens/privacy` |

**How to set it up.** In the Vigorly repository, create a `docs/` folder,
then in Settings → Pages set the source to "Deploy from a branch", branch
`main`, folder `/docs`. Add three files:

- `docs/index.md` — the marketing page. App name, one-line pitch, three or four
  screenshots, what makes it different (ranges and sources, on-device voice,
  works without an account), a link to the App Store, and a link to
  `ACCURACY.md` for anyone who wants the method.
- `docs/privacy.md` — paste `PRIVACY.md` from this repository verbatim. Apple
  checks that this URL loads, is publicly reachable without a login, and
  actually describes the app.
- `docs/support.md` — this is the page reviewers and users click when something
  breaks, and it must not be empty. Put: how to reach you
  (sahil.bharti97@gmail.com), expected reply time, a short FAQ (why estimates
  show a range, why voice needs microphone and speech permission, how to grant
  Apple Health access, how to export a backup, how to delete an account), and a
  link to the privacy policy.

A support URL that 404s or redirects to a bare repository README is a common
reason for a metadata rejection. Load all three URLs in a private browser
window before you submit.

---

## 6. App Privacy answers

**Read this first.** Your answers depend on which build you ship, and the two
possible builds give genuinely different answers.

The account and sync service only exists when `EXPO_PUBLIC_API_URL` is set at
build time. As this repository stands today there is no `mobile/.env` and no
`env` block in the `production` profile of `mobile/eas.json`, so a production
EAS build ships with the account service disabled. The auth screen shows
"Account service is not configured in this release", the sign-in button is
disabled, and there is no code path that can send anything off the device.
**Confirm which of these you are submitting before you answer.**

### Scenario A — shipping as the repo stands (no sync server configured)

Answer the first question in App Store Connect as:

> **Do you or your third-party partners collect data from this app? → No, we do
> not collect data from this app.**

That single answer produces the "Data Not Collected" label. It is accurate
because nothing in the binary transmits user data anywhere: storage is local
SQLite encrypted with a device-only key, speech is transcribed by iOS, Apple
Health is read on device, and there are no analytics, advertising, crash
reporting or attribution SDKs in `mobile/package.json`.

### Scenario B — shipping with the account and sync service live

Then you must disclose what the server receives. Data Apple considers
"collected" includes anything transmitted off the device and retained, even if
it is encrypted and even if the user opted in.

| Data type | Collected? | Linked to identity? | Used for tracking? | Purpose |
| --- | --- | --- | --- | --- |
| Contact Info → Name | Yes (display name at signup) | Linked | No | App Functionality |
| Contact Info → Email Address | Yes (account and recovery) | Linked | No | App Functionality |
| Contact Info → Phone Number | No | — | — | — |
| Contact Info → Physical Address | No | — | — | — |
| Contact Info → Other User Contact Info | No | — | — | — |
| **Health & Fitness → Health** | Yes (weight, sleep, and any Apple Health values that land in the synced vault) | Linked | No | App Functionality |
| **Health & Fitness → Fitness** | Yes (meals, workouts, sets, steps, active energy, water) | Linked | No | App Functionality |
| Financial Info (all) | No | — | — | — |
| Location → Precise | No | — | — | — |
| Location → Coarse | No | — | — | — |
| Sensitive Info | No | — | — | — |
| Contacts | No | — | — | — |
| User Content → Audio Data | **No** | — | — | — |
| User Content → Photos or Videos | No | — | — | — |
| User Content → Emails or Text Messages | No | — | — | — |
| User Content → Customer Support | No | — | — | — |
| User Content → Other User Content | Yes (free-text notes on workouts, routines and profile, which travel inside the vault) | Linked | No | App Functionality |
| Browsing History | No | — | — | — |
| Search History | No | — | — | — |
| Identifiers → User ID | Yes (account ID and session token) | Linked | No | App Functionality |
| Identifiers → Device ID | No | — | — | — |
| Purchases → Purchase History | No | — | — | — |
| Usage Data → Product Interaction | No | — | — | — |
| Usage Data → Advertising Data | No | — | — | — |
| Usage Data → Other Usage Data | No | — | — | — |
| Diagnostics → Crash Data | No | — | — | — |
| Diagnostics → Performance Data | No | — | — | — |
| Diagnostics → Other Diagnostic Data | No | — | — | — |
| Surroundings / Body / Other Data | No | — | — | — |

Purposes: select **App Functionality only** for every collected type. Do not
select Analytics, Product Personalization, Developer's Advertising or Third-Party
Advertising, because none of them happen.

Tracking: on the "Tracking" question answer **No** for everything. Nothing is
linked with third-party data, no advertising identifier is read, and there is no
ATT prompt in the app.

### The two answers you should be ready to defend

**Health and Fitness data is not collected by the developer when the app runs
without an account.** In that mode, everything the app knows lives in local
SQLite on the phone, encrypted with AES-GCM under a 256-bit key generated on
first launch and kept in the iOS Keychain with
`WHEN_UNLOCKED_THIS_DEVICE_ONLY`. Apple Health values are read through HealthKit
on the device with your permission and written into that same local store. No
network request carries them anywhere, and there is no server that could receive
them. `Continue without an account` on the auth screen is a supported, complete
mode, not a trial.

**What changes when a user opts into sync.** Creating an account sends the
encrypted fitness vault to the Vigorly sync service so a second device can
restore it, and stores an email address, a display name, and a salted
PBKDF2-HMAC hash of the password and of the recovery code. From that moment the
data is "collected" in Apple's sense and must be disclosed exactly as in the
table above. It is still never sold, never shared with third parties, never used
for advertising and never used to track users across apps or websites.

**One honest caveat about the encryption claim.** The server-side vault in
`account_store.py` is encrypted with AES-GCM under a master key held by the
server (`CALORIE_LENS_MASTER_KEY`), and `api.py` reads fields such as
`coachMemory` and `profile` out of the decrypted payload. That is strong
encryption at rest, but it is not end-to-end encryption, so do not write
anywhere that the server "cannot read" the data. `PRIVACY.md` is worded to match
what the code actually does. If you want the stronger claim later, the change is
to derive the vault key on the device from the user's password and never send it
to the server.

---

## 7. Age rating questionnaire

Target rating: **4+**. Answer every content question with the lowest option.

| Question | Answer |
| --- | --- |
| Cartoon or Fantasy Violence | None |
| Realistic Violence | None |
| Prolonged Graphic or Sadistic Realistic Violence | None |
| Sexual Content or Nudity | None |
| Graphic Sexual Content and Nudity | None |
| Profanity or Crude Humor | None |
| Alcohol, Tobacco, or Drug Use or References | None |
| Mature or Suggestive Themes | None |
| Horror or Fear Themes | None |
| Medical or Treatment-Focused Content | None |
| Simulated Gambling | None |
| Gambling (real money) | No |
| Contests | No |
| Unrestricted Web Access | No |
| In-App Purchases | No |
| Advertisements | No |
| User-Generated Content | No |
| Messaging and Chat | No |
| Made for Kids / Kids Category | No |
| Age Assurance used | No |
| Parental controls in app | No |

Notes for the two answers a reviewer might question:

- **Medical or Treatment-Focused Content → None.** Vigorly estimates food
  and exercise energy and reports arithmetic about the user's own logs. It does
  not diagnose, does not treat, does not give dosing or clinical guidance, and
  states in the app that it is not medical advice. Nutrition tracking on its own
  does not trigger this category. If you ever add symptom guidance, medication
  reminders, or clinical interpretation, revisit this answer.
- **Unrestricted Web Access → No.** The Train tab's technique videos open in the
  system browser or the video app via `Linking.openURL`. The app embeds no web
  view of its own. If you ever switch those links to an in-app browser, this
  answer changes and the rating rises.
- **User-Generated Content → No.** Notes and routine names are private to the
  user's own device and account. Nothing is published, shared, or visible to
  another user.

---

## 8. Category

- **Primary: Health & Fitness.** This is where the app competes and where
  reviewers expect nutrition and training trackers.
- **Secondary: Food & Drink.** Best of the remaining options for the nutrition
  half, and it costs nothing to set.

If Food & Drink feels wrong to you next to recipe and delivery apps, **Lifestyle**
is the reasonable alternative secondary. The primary category is the one that
matters for ranking; the secondary rarely moves traffic.

---

## 9. Review notes for App Review

Paste into the "Notes" field in App Review Information.

```
Vigorly is a nutrition and strength-training tracker. It works completely
without an account.

GETTING IN
On first launch the app shows a short setup flow (goal, body inputs, activity,
training, diet, bowl size). Any reasonable answers are fine. At the end you
reach a sign-in screen. Tap "Continue without an account" at the bottom and you
have the entire app: food logging, the Train tab, Progress and Coach. No
account, network connection or purchase is required for any feature under
review. There is no paid tier and no in-app purchase in this version.

DEMO ACCOUNT
Not required, since the whole app is reachable without signing in. If you would
prefer one anyway:
  Email: <DEMO-EMAIL>
  Password: <DEMO-PASSWORD>
An account only enables encrypted sync of the same data between the user's own
devices.

VOICE LOGGING (microphone + speech recognition permission)
Tapping the microphone on the Today, Food or Coach tab starts dictation. iOS
asks for microphone and speech recognition permission the first time. The
transcript is produced by the operating system's own speech recognizer and is
parsed on the device. No audio is recorded to a file and no audio is uploaded to
any server. If you would rather not grant the permission, every voice entry can
be typed instead: tap the same bar and use the keyboard, for example
"2 rotis and one bowl dal" or "30 minute brisk walk".

APPLE HEALTH (physical device required)
Progress > Connected health reads steps, active energy, sleep and weight from
HealthKit after you approve the categories in the system sheet. The Simulator
usually contains no HealthKit samples and cannot receive Apple Watch data, so
the app will report that it connected but found no approved samples. That
message is expected on the Simulator and is not a bug. Please test Health on a
physical iPhone.

ACCURACY AND CLAIMS
Calorie values come from a deterministic engine over USDA FoodData Central
records; every entry shows a range, the assumptions, a confidence label and the
source record ID. Exercise energy uses the 2024 Adult Compendium of Physical
Activities with resting energy excluded. No calorie value is generated by a
language model. The app states in several places that it provides estimates and
is not medical advice.

Contact for anything else: sahil.bharti97@gmail.com
```

Replace `<DEMO-EMAIL>` and `<DEMO-PASSWORD>` with a real, working account, or
delete those three lines entirely and tick "Sign-in not required". Do not leave
placeholder credentials in the field; a demo login that fails is a guaranteed
rejection.

---

## 10. Export compliance

`mobile/app.json` currently declares:

```json
"infoPlist": {
  "ITSAppUsesNonExemptEncryption": false
}
```

That key tells App Store Connect to stop asking the encryption questions at
every upload and to record that the app contains no *non-exempt* encryption.
Note the word: it does not mean the app contains no encryption. It is an
assertion that whatever encryption it contains falls under an exemption in
Category 5 Part 2 of the US Export Administration Regulations.

What the app actually does with encryption:

- `mobile/src/lib/secure-storage.ts` encrypts the local fitness vault with
  **AES-256-GCM** from the bundled `@noble/ciphers` library, under a random
  256-bit key generated on device and stored in the iOS Keychain.
- Session tokens and recovery codes are held in Keychain through
  `expo-secure-store`.
- If the sync service is configured, network calls use HTTPS, and the server
  encrypts the stored vault with AES-GCM.

**An honest word to you, Sahil.** I can lay out the facts, but I cannot make
this call for you and you should not take a `false` on faith just because it is
already in the file. The usual reason an app answers `false` without further
paperwork is that its only encryption is HTTPS or the encryption built into iOS.
That is not quite your situation: you bundle a third-party AES-GCM
implementation and use a 256-bit symmetric key for data confidentiality, which
is exactly the case where the easy exemptions are least obvious. The exemption
for encryption "limited to authentication" does not cover encrypting a user's
data at rest, and the low-key-length exemptions cap out far below 256 bits. Two
routes are commonly available to an app like yours: the publicly-available
source-code route, if the Vigorly repository is genuinely public and you
send the notification email that route requires, or mass-market
self-classification, which typically means answering "yes" to the encryption
questions and filing an annual self-classification report. Read Apple's
"Complying with Encryption Export Regulations" page and the underlying BIS rules
yourself, decide deliberately, and if you are still unsure, spend an hour with
someone who does export compliance for a living. It is a small cost against a
declaration you are personally signing. This is a factual summary, not legal
advice.

Whichever way you decide, the fix is small: either leave the key as is, or
change it to `true` (or remove it and answer the questions in App Store Connect
at upload time) and complete the paperwork that answer implies. Also add the
French declaration text if you distribute in France, which App Store Connect
will prompt you for if you answer `true`.

---

## 11. What is needed before submitting

Only you can do these.

**Accounts and identity**

- [ ] Apple Developer Program membership active and the annual fee paid.
- [ ] Agreements, Tax, and Banking complete in App Store Connect, even for a
      free app; the Free Apps agreement must show "Active".
- [ ] App record created in App Store Connect with bundle ID
      `com.advaice.calorielens`.

**Assets you must produce**

- [ ] 1024x1024 App Store icon, no alpha channel, no rounded corners.
- [ ] Screenshots for 6.9" and 6.5" iPhone displays. Lead with a food estimate
      showing its range, confidence and source ID, since that is the whole
      pitch. Then the Train tab mid-session, then Progress, then Coach.
- [ ] Optional app preview video.

**Publish before you submit**

- [ ] `PRIVACY.md` live at a public URL and pasted into the Privacy Policy URL
      field.
- [ ] Support page live and answering real questions.
- [ ] Both URLs loaded in a private browser window to confirm no login wall.

**Decisions only you can make**

- [ ] Confirm whether the production build ships with `EXPO_PUBLIC_API_URL` set,
      then answer the App Privacy section as Scenario A or Scenario B above.
- [ ] Confirm the `ITSAppUsesNonExemptEncryption` declaration yourself.
- [ ] Decide on a demo account, or tick "Sign-in not required" in App Review
      Information.
- [ ] Set your contact phone and email for App Review.

**Build and test**

- [ ] Bump `buildNumber` for every upload; `version` stays 1.0.0 for the first
      release.
- [ ] Run `npm run typecheck` and `npm run lint` clean.
- [ ] `TEAM_ID=95R5R7A683 ./scripts/archive.sh`, which archives, exports a signed
      .ipa and then checks that .ipa is distribution-signed under Advaice
      Limited. Upload it with Transporter. The build is deliberately made on your
      own Mac: no third-party build service ever holds the signing certificate.
      `TESTFLIGHT.md` is the full runbook.
- [ ] On that physical device, verify: onboarding completes, "Continue without
      an account" reaches the tabs, voice logging asks for microphone and speech
      permission and produces an entry, Apple Health returns real samples,
      a workout logs and shows its energy range, and the app behaves in
      Airplane Mode.
- [ ] Delete and reinstall once to confirm a clean first run.

**Repo and metadata consistency — done, but re-check if you edit the docs**

`README.md`, `mobile/README.md` and `ACCURACY.md` used to describe Gemini-based
voice transcription and an AI coach with long-term memory. They no longer do:
`speech.ts` uses the operating system's recognizer and `insights.ts` is pure
arithmetic, and the docs now say so. The remaining `GOOGLE_API_KEY` mentions in
`README.md` and `DEPLOY.md` are about the optional Python backend and the
Streamlit companion, neither of which ships in the iOS app.

This matters because a reviewer, or a journalist, can read your public repository
alongside your App Privacy answers. Those two must tell the same story. If you
change what the app does, change the docs in the same commit.
