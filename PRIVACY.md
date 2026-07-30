# Calorie Lens Privacy Policy

**Last updated: 30 July 2026**

Calorie Lens is made by one person, Sahil Bharti. There is no company behind it,
no investor asking what the data could be worth, and no advertising business
that the app quietly serves. It is a fitness tracker I built because I wanted to
use it, and this page describes exactly what it does with your information.

The short version: by default, Calorie Lens collects nothing. Everything you log
stays on your phone, encrypted. If you want your history on more than one
device, you can create an account, and then a copy is stored on a server so it
can be sent back to you. That is the only situation in which any of your data
leaves your phone, and it only happens because you asked for it.

## What is stored on your device

Everything you log lives in a local database on your iPhone: meals and their
estimates, workouts and sets, routines, water, steps, sleep, weight, your goals
and targets, your onboarding answers such as age, height and bowl size, and any
notes you write.

That database is encrypted before it is written. Calorie Lens generates a random
256-bit key the first time you open the app, keeps it in the iOS Keychain marked
as available only on this device while it is unlocked, and encrypts your records
with AES-GCM under that key. Your account session token and recovery code, if
you have an account, are held in the Keychain too.

The key never leaves your phone and I never see it. If you delete the app, the
local database and the key go with it.

## What is collected when you do not create an account

Nothing.

Calorie Lens is designed to be fully usable without an account. On the sign-in
screen there is a button that says "Continue without an account", and it is not a
trial or a limited mode. Food logging, voice logging, the estimate engine, the
entire Train tab, Progress, Coach, Apple Health, backups and export all work
that way, offline, indefinitely.

In that mode there is no server involved, so there is nothing for me to receive,
store, read, analyse or lose.

## Voice logging and the microphone

When you dictate an entry, the microphone is on only while you are speaking, and
the recording is turned into text by your iPhone's own speech recognition, the
same system feature that powers dictation on the keyboard.

No audio is written to a file. No audio is uploaded to a Calorie Lens server. I
never receive a recording of your voice, and there is no transcription service,
subscription or API key involved. The resulting text is handed straight to the
same on-device parser that typed entries use.

Speech recognition on iOS may use Apple's own on-device model or, depending on
your device, language and settings, Apple's servers. That path is governed by
Apple's privacy policy and your iOS settings, not by me, and it is the same path
any dictation on your phone takes. Calorie Lens has no access to it beyond
receiving the finished text.

If you would rather not grant microphone or speech permission, you can type
every entry instead. Nothing in the app is voice-only.

## Apple Health

With your permission, Calorie Lens reads four things from Apple Health: step
count, active energy burned, sleep, and body weight. iOS asks you for each
category and you can grant or refuse any of them, at any time, in the Health
app.

Those values are read on your device and written into the same encrypted local
database as everything else. They are used to fill in your dashboard, your trend
charts and your weight history, and for nothing else.

Health data is never sold. It is never shared with anyone. It is never used for
advertising, marketing or profiling, and it is never sent to a third party. If
you sync an account, your Health-derived values travel with the rest of your
fitness history and nowhere else. Apple's rules for HealthKit apps require all of
this, and I agree with the rules.

## The optional account, and what a sync server can and cannot see

If you create an account, three things are stored on the sync service:

1. **Your email address and the display name you chose.** These identify the
   account and let you sign in and recover it.
2. **A salted hash of your password, and a salted hash of your recovery code.**
   The hashing is PBKDF2-HMAC. Your actual password is never stored anywhere and
   cannot be worked back out of the hash, not by me and not by anyone who
   obtained a copy of the database.
3. **Your fitness vault, stored as an AES-GCM encrypted record.** This is the
   same information the app keeps on your phone: meals, workouts, routines,
   water, steps, sleep, weight, goals and notes.

What the service never receives, because the app never asks for it in the first
place: your location, your contacts, your photos, your messages, your calendar,
your browsing history, any advertising identifier, any device fingerprint, and
any recording of your voice.

Your vault is not read for analytics. It is not mined for insights or trends. It
is not sold, rented, shared, published, or handed to any advertising, data
broker or marketing business, and there are no third-party partners with access
to it. It exists so that a second device signing into your account can get your
history back.

I want to be precise about how far the encryption goes rather than let a
comforting word do work it has not earned. Your data is encrypted on your phone
with a key only your phone holds. The copy on the sync server is encrypted at
rest with AES-GCM, but the key for that copy is held by the server, which means
this is strong protection against a stolen database rather than a mathematical
guarantee that the operator could never decrypt it. If you want the guarantee
that no server anywhere holds a readable copy of your fitness history, do not
create an account. The app is built to work perfectly well that way, and that is
the honest recommendation.

## No analytics, no advertising, no tracking

Calorie Lens contains no analytics SDK, no crash reporting service, no
attribution or install-tracking library, and no advertising network. I do not
know how many times you opened the app, which tab you use most, how long you
stayed, or whether you have logged anything today.

Nothing in the app tracks you across other apps or websites. No advertising
identifier is requested or read, so you will never see an App Tracking
Transparency prompt from Calorie Lens. Your data is not combined with data from
any other source, and it is not used to build a profile of you.

The one thing that leaves the app on purpose, apart from sync, is a link: tapping
a technique video in the Train tab opens it in your browser or video app. From
that moment you are on someone else's website and their privacy policy applies,
not mine. Calorie Lens does not tell them anything about you.

## Backups and export

You can export your data as a JSON file from the account screen and share or
store it wherever you like. That file is your copy, in the clear, and once it
leaves the app its safety is in your hands. You can import it back into Calorie
Lens on any device.

## Deleting your data

Deleting the app from your iPhone deletes the local database and the encryption
key. There is no recovery after that, so export a backup first if you want one.

If you have an account, open the account screen in the app and use **Delete
account**. This removes your account and the synced vault from the server. It is
permanent and there is no undo.

Revoking Apple Health permission in the Health app stops any further reading
immediately. Values already copied into your Calorie Lens history stay until you
delete them or delete the app.

If you would rather I did it for you, or you have lost access to your account,
email sahil.bharti97@gmail.com from the address on the account and I will delete
it. Depending on where you live you may have a legal right to access, correct,
export or delete your personal data. Because the app collects nothing unless you
create an account, exercising those rights is usually as simple as exporting a
backup and deleting the app, but write to me and I will help either way.

## Children

Calorie Lens is not directed at children. It is rated 4+ because it contains no
objectionable content, not because it is designed for young children, and it is
not distributed in the Kids Category.

I do not knowingly collect personal information from anyone under 13, and
because the app collects nothing at all unless an account is created, in normal
use there is nothing to collect from anyone of any age. If you believe a child
has created an account, email me and I will delete it.

Anyone using a calorie tracker should know that tracking food can become
unhealthy for some people, particularly younger users. If it stops being useful
to you, delete it. That is a feature of a tool you own.

## Changes to this policy

If the app changes in a way that changes this policy, I will update this page and
the date at the top. Material changes, particularly anything that would cause
Calorie Lens to collect something it does not collect today, will also be called
out in the app's release notes rather than quietly edited in here.

## Contact

Sahil Bharti
sahil.bharti97@gmail.com

Write to me about anything on this page. I read it.
