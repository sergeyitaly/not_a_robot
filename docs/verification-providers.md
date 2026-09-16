# Human-verification and bot-management providers (reference)

A reference list of the verification/bot-management products actually in
production use, for understanding what this library's own detector is
positioned alongside. This is vendor/product documentation, not a list of
individual customer websites, and it is not a target list for scraping or
automation against anyone's live systems.

There are on the order of 30-40 distinct products in real use, not
hundreds -- most of the market consolidates around a handful of vendors.
Padding this list to "100 websites" would mean listing individual sites
that merely embed one of these products, which isn't a meaningful or safe
thing to compile (see [description.txt](../description.txt) discussion in
the repo's history for why).

## Challenge-based (visible puzzle, e.g. checkbox / image grid / slider)

| Provider | Product | Notes |
|---|---|---|
| Google | reCAPTCHA v2 | Checkbox + fallback image-grid challenge |
| Google | reCAPTCHA Enterprise | v2/v3 combined with account-level risk data |
| hCaptcha | hCaptcha / hCaptcha Enterprise | Privacy-focused reCAPTCHA v2 alternative |
| GeeTest | GeeTest v3/v4 | Slider and icon-matching puzzles with behavioral scoring |
| Arkose Labs | FunCaptcha / Arkose MatchKey | 3D-object puzzles, used by large enterprise sites |
| Capy | Capy Puzzle CAPTCHA | Drag-and-drop puzzle piece |
| MTCaptcha | MTCaptcha | Distorted-text and puzzle variants |
| BotDetect | BotDetect CAPTCHA | Legacy distorted-text CAPTCHA library |

## Invisible / behavioral-score based

| Provider | Product | Notes |
|---|---|---|
| Google | reCAPTCHA v3 | Invisible, returns a 0.0-1.0 risk score, no challenge UI |
| Cloudflare | Turnstile | Invisible, proof-of-work + behavioral, no puzzle by default |
| AWS | WAF CAPTCHA / WAF Bot Control | Managed rule-based bot scoring integrated into WAF |
| DataDome | Bot Protection | Real-time behavioral + fingerprint scoring |
| HUMAN Security | (formerly PerimeterX) Bot Defender | Behavioral + fingerprint bot management |
| Akamai | Bot Manager | Behavioral, fingerprinting, and device signals |
| Imperva | Advanced Bot Protection (formerly Incapsula) | Behavioral + reputation scoring |
| Kasada | Kasada Bot Prevention | Client-side polymorphic JS challenges + behavioral |
| F5 | Distributed Cloud Bot Defense (formerly Shape Security) | Behavioral + telemetry-based scoring |
| Radware | Bot Manager | Behavioral fingerprinting and device intelligence |
| Fastly | Bot Management (formerly Signal Sciences) | Behavioral scoring integrated into edge/CDN |
| Netacea | Bot Management | Server-side behavioral analytics, no client JS required |
| Sift | Sift Digital Trust & Safety | Fraud/bot risk scoring across signup, login, payment |
| Castle | Castle | Behavioral risk scoring for account-abuse detection |

## Proof-of-work based (no visual puzzle, client CPU cost instead)

| Provider | Product | Notes |
|---|---|---|
| Friendly Captcha | Friendly Captcha | Client-side proof-of-work, no image challenge |
| Cap.js (open source) | cap | Lightweight open-source proof-of-work widget |
| mCaptcha (open source) | mCaptcha | Self-hostable proof-of-work CAPTCHA |

## Notable but discontinued / legacy

| Provider | Product | Notes |
|---|---|---|
| Solve Media | Solve Media CAPTCHA | Ad-based text CAPTCHA, discontinued |
| NuCaptcha | NuCaptcha | Animated-text CAPTCHA, discontinued |
| Are You a Human | PlayThru | Game-based CAPTCHA, discontinued |

## Why this matters for this library

`not_a_robot`'s own detector is closest in spirit to the "invisible /
behavioral-score" category above (reCAPTCHA v3, Turnstile, DataDome, etc.):
it scores a session's mouse/keystroke telemetry rather than presenting a
puzzle. As the project README notes, for that category the visible
puzzle (if any) is rarely the deciding factor -- the behavioral signal is.
If you're evaluating whether to run this library's detector standalone or
alongside one of the products above, the products in the "invisible /
behavioral" table are the closest functional comparison.
