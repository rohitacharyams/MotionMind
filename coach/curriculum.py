"""curriculum.py — structured Hip-Hop + House lesson tracks (v195).

WHY: landing users didn't know WHAT to learn. This turns the app from a
random-session toy into a real academy for two styles we double down on:
Hip-Hop and House. Each track is a sequence of LESSONS; each lesson teaches
ONE foundational move with genuine pedagogy — its purpose, how-to cues, the
INVARIANT (the one thing that must stay true or it isn't that move), the
music it lives on, a real avatar demonstration clip, and a drill the coach
watches (existing 2D pose feedback).

Content is grounded in street-dance foundations (groove/bounce, low center,
isolations, musicality = hitting accents / riding the kick) rather than
choreography-to-music. Demo clips map to REAL ids served by
/api/motion/list (AIST++ gLH / gHO). If a specific id is ever missing the
client falls back to the first clip of that genre (same pattern the analyze
flow already uses), so lessons never dead-end.

Consumed by:
- GET /api/curriculum         -> the two tracks (this module, static)
- the Learn tab (coach.js)    -> renders tracks/lessons + progress
- lesson start                -> plays demo clip, then a drill session
- LLM narration               -> constrained to the lesson's cues/invariant
  so the coach teaches depth instead of hallucinating.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ─── Hip-Hop track (gLH) ──────────────────────────────────────────────
_HIPHOP: List[Dict[str, Any]] = [
    {
        "id": "hh_bounce",
        "title": "The Bounce (Groove)",
        "emoji": "🔽",
        "level": "Foundation",
        "one_liner": "The engine of all hip-hop — the pulse you ride.",
        "what": ("Hip-hop lives in a constant, relaxed bounce. Before any "
                 "step or move, you find the groove: a soft up-and-down driven "
                 "from the knees while the chest stays loose."),
        "purpose": ("Everything in hip-hop sits on top of the bounce. If your "
                    "groove is solid, every move looks clean; if it's stiff, "
                    "nothing reads as hip-hop."),
        "cues": [
            "Soft, springy knees — never locked.",
            "Bounce DOWN on the beat, not up (weight drops into the floor).",
            "Chest and shoulders stay relaxed and heavy.",
            "Keep weight slightly in your heels.",
        ],
        "invariant": "The down-bounce lands exactly on the count. Miss the "
                     "count and it's just bobbing, not a groove.",
        "music": {
            "feel": "Boom-bap / classic hip-hop, mid-tempo.",
            "bpm": [85, 100],
            "hit": "Ride the kick and snare — down on the kick.",
        },
        "demo_clip": "gLH_sBM_cAll_d17_mLH4_ch01",
        "drill_genre": "gLH",
        "drill_prompt": "Let's drill the hip-hop bounce — watch my groove, "
                        "then match my timing on the beat.",
    },
    {
        "id": "hh_twostep",
        "title": "The Two-Step (Rock)",
        "emoji": "↔️",
        "level": "Foundation",
        "one_liner": "Travel the groove side to side — your basic.",
        "what": ("Your first real step: shift weight and step out to the side "
                 "on 1, bring it back together on 2, while the bounce never "
                 "stops. It's the hip-hop equivalent of learning to walk to "
                 "the beat."),
        "purpose": ("Gives you a 'home base' groove you can always fall back to "
                    "in a freestyle, and teaches weight transfer — the basis of "
                    "every travelling move."),
        "cues": [
            "Step out on 1, feet together on 2.",
            "Push off the inside edge of your foot.",
            "The bounce keeps pulsing the whole time.",
            "Stay low — don't rise up when you step.",
        ],
        "invariant": "The bounce never stops when the feet move. If you freeze "
                     "the groove to step, you've lost it.",
        "music": {
            "feel": "Any hip-hop groove you can nod to.",
            "bpm": [90, 105],
            "hit": "Step lands on the count, weight shift between.",
        },
        "demo_clip": "gLH_sBM_cAll_d17_mLH0_ch02",
        "drill_genre": "gLH",
        "drill_prompt": "Two-step time — I'll travel side to side, you keep the "
                        "bounce alive between steps.",
    },
    {
        "id": "hh_drop",
        "title": "The Bounce Drop (Accent)",
        "emoji": "💥",
        "level": "Core",
        "one_liner": "Punctuate the music by dropping your weight.",
        "what": ("A sharp downward accent: on a big hit in the music you drop "
                 "your weight harder and lower than the normal bounce, then "
                 "recover soft. This is how you 'hit' the music."),
        "purpose": ("Turns a flat groove into something that speaks to the "
                    "song. Accents are what make a crowd react."),
        "cues": [
            "Pick ONE accent in the music to hit.",
            "Drop sharper and lower than your bounce.",
            "Recover soft and immediately back into groove.",
            "Let the arms react naturally to the drop.",
        ],
        "invariant": "The drop is visibly sharper than the bounce, and it lands "
                     "on the accent — not a beat early or late.",
        "music": {
            "feel": "Tracks with clear hits / horn stabs.",
            "bpm": [90, 105],
            "hit": "Drop on the accent (a stab, a big snare).",
        },
        "demo_clip": "gLH_sBM_cAll_d17_mLH4_ch09",
        "drill_genre": "gLH",
        "drill_prompt": "Let's catch accents — groove with me, then drop hard "
                        "on the hit I call.",
    },
    {
        "id": "hh_combo",
        "title": "Groove + Hits Combo",
        "emoji": "🔗",
        "level": "Combo",
        "one_liner": "Chain groove, step and accents into a phrase.",
        "what": ("Your first mini-routine: 8 counts that put together the "
                 "bounce, a two-step, and a drop accent so it reads as a real "
                 "hip-hop phrase instead of isolated moves."),
        "purpose": ("Learning to link moves on an 8-count is the bridge from "
                    "'knowing steps' to actually dancing."),
        "cues": [
            "Count it out loud: 1-2-3-4-5-6-7-8.",
            "Groove on 1-4, two-step on 5-6, drop on 7, recover 8.",
            "Keep the bounce continuous underneath.",
            "Smooth transitions matter more than big moves.",
        ],
        "invariant": "The whole phrase stays on one continuous groove — the "
                     "moves change but the pulse never breaks.",
        "music": {
            "feel": "Any hip-hop track with a steady 8-count.",
            "bpm": [90, 100],
            "hit": "Land each element on its count.",
        },
        "demo_clip": "gLH_sBM_cAll_d18_mLH2_ch01",
        "drill_genre": "gLH",
        "drill_prompt": "Here's an 8-count combo — follow me slow, then we run "
                        "it on tempo.",
    },
    {
        "id": "hh_musicality",
        "title": "Musicality — Riding the Beat",
        "emoji": "🎧",
        "level": "Mastery",
        "one_liner": "Stop counting, start listening — dance the song.",
        "what": ("The final foundation: pick a single element in the music (the "
                 "hi-hat, the horn, the snare) and let your body follow THAT, "
                 "instead of a fixed count. This is what separates a dancer "
                 "from someone doing steps."),
        "purpose": ("Musicality is the whole point — it's what makes freestyle "
                    "feel personal and alive."),
        "cues": [
            "Pick ONE instrument and shadow it with your body.",
            "Switch which instrument you follow between 8s.",
            "Leave space — you don't have to move on every beat.",
            "React to the song, don't perform at it.",
        ],
        "invariant": "Your accents match real events in the music — someone "
                     "watching can 'hear' the song through your body.",
        "music": {
            "feel": "A track with layers (hats, bass, melody) to choose from.",
            "bpm": [85, 105],
            "hit": "Follow whichever layer you chose this 8.",
        },
        "demo_clip": "gLH_sFM_cAll_d16_mLH4_ch05",
        "drill_genre": "gLH",
        "drill_prompt": "Freestyle musicality — I'll follow the melody, you "
                        "shadow the drums. Then we switch.",
    },
]


# ─── House track (gHO) ────────────────────────────────────────────────
_HOUSE: List[Dict[str, Any]] = [
    {
        "id": "ho_jack",
        "title": "The Jack",
        "emoji": "🌊",
        "level": "Foundation",
        "one_liner": "The heartbeat of house — a wave through your torso.",
        "what": ("House is built on 'the jack': a continuous contraction and "
                 "release of the chest/torso that pulses on every kick of the "
                 "four-on-the-floor beat. It's a wave rippling up through your "
                 "body."),
        "purpose": ("The jack IS house. Every house step sits on top of this "
                    "torso pulse; without it, footwork looks empty."),
        "cues": [
            "Contract the chest in, then release out — a wave.",
            "Drive it from the kick drum: 4 pulses per bar.",
            "Knees stay soft and springy underneath.",
            "Let the pulse travel up the spine, not just the shoulders.",
        ],
        "invariant": "The jack pulses on every kick — four even pulses to the "
                     "bar. Lose the four-count pulse and it isn't house.",
        "music": {
            "feel": "Classic / soulful house, four-on-the-floor.",
            "bpm": [118, 128],
            "hit": "Pulse on every kick drum (all four beats).",
        },
        "demo_clip": "gHO_sBM_cAll_d20_mHO0_ch02",
        "drill_genre": "gHO",
        "drill_prompt": "Let's find the jack — feel the four-on-the-floor and "
                        "pulse your chest on every kick with me.",
    },
    {
        "id": "ho_farmer",
        "title": "Loose Legs (Farmer)",
        "emoji": "🦵",
        "level": "Foundation",
        "one_liner": "The fast, loose footwork foundation of house.",
        "what": ("A quick in-out step where the legs stay loose and light — "
                 "knees fall in, then kick out — riding the jack on top. This "
                 "is the base of house footwork speed."),
        "purpose": ("Trains the light, fast, relaxed legs house is famous for, "
                    "and the coordination of footwork against the torso jack."),
        "cues": [
            "Stay light on the balls of your feet.",
            "Knees drop in, feet kick out — loose, not forced.",
            "Keep the jack pulsing above the legs.",
            "Speed comes from relaxation, not tension.",
        ],
        "invariant": "The jack keeps pulsing while the legs move — two rhythms "
                     "at once (torso + feet).",
        "music": {
            "feel": "Groovy house with a driving bassline.",
            "bpm": [120, 128],
            "hit": "Footwork rides the off-beats, jack on the kick.",
        },
        "demo_clip": "gHO_sBM_cAll_d19_mHO1_ch02",
        "drill_genre": "gHO",
        "drill_prompt": "Loose legs drill — keep them light and let the jack "
                        "ride on top. Match my speed.",
    },
    {
        "id": "ho_heeltoe",
        "title": "Heel-Toe",
        "emoji": "👟",
        "level": "Core",
        "one_liner": "The signature house travelling step.",
        "what": ("Alternating heel and toe touches that let you travel and "
                 "swivel while the jack keeps pulsing. A defining house step "
                 "you'll see in every cypher."),
        "purpose": ("Adds travel and flavour to your footwork and teaches "
                    "ankle articulation — key to advanced house."),
        "cues": [
            "Tap heel out, then toe in — clean and rhythmic.",
            "Let the hips and jack follow the feet.",
            "Small at first; speed and size come later.",
            "Stay on the balls for quick weight changes.",
        ],
        "invariant": "Heel and toe hit distinct counts and the torso jack "
                     "never stops underneath.",
        "music": {
            "feel": "Uplifting house with clear percussion.",
            "bpm": [122, 128],
            "hit": "Heel-toe on the counts, jack on the kick.",
        },
        "demo_clip": "gHO_sBM_cAll_d19_mHO2_ch02",
        "drill_genre": "gHO",
        "drill_prompt": "Heel-toe time — small and clean first, jack staying "
                        "alive. Follow my feet.",
    },
    {
        "id": "ho_combo",
        "title": "House Jack Combo",
        "emoji": "🔗",
        "level": "Combo",
        "one_liner": "Link the jack, loose legs and heel-toe into a groove.",
        "what": ("An 8-count that flows the jack into loose legs into a heel-toe "
                 "travel, so it reads as continuous house dancing rather than "
                 "separate steps."),
        "purpose": ("Teaches you to flow between house steps without dropping "
                    "the pulse — the essence of the style's endless groove."),
        "cues": [
            "Jack on 1-2, loose legs on 3-4, heel-toe travel 5-8.",
            "Never stop the four-on-the-floor pulse.",
            "Transitions should feel like water, not stops.",
            "Let it loop — house is about the endless groove.",
        ],
        "invariant": "The four-count jack pulse runs unbroken through every "
                     "step of the phrase.",
        "music": {
            "feel": "A rolling house track you could loop forever.",
            "bpm": [122, 126],
            "hit": "Each step on its count, jack on every kick.",
        },
        "demo_clip": "gHO_sBM_cAll_d21_mHO5_ch05",
        "drill_genre": "gHO",
        "drill_prompt": "Here's a house combo — follow me slow, then we let it "
                        "loop on tempo.",
    },
    {
        "id": "ho_musicality",
        "title": "Musicality — Riding the Four",
        "emoji": "🎧",
        "level": "Mastery",
        "one_liner": "Feel the house build and ride it, don't count it.",
        "what": ("House music breathes — filters open, percussion drops in and "
                 "out. The final foundation is letting your intensity follow "
                 "the track's energy instead of a fixed step, while the jack "
                 "always anchors you."),
        "purpose": ("This is what makes house feel hypnotic and personal — "
                    "you're riding the music's arc, not performing steps."),
        "cues": [
            "Anchor on the jack, vary everything else with the energy.",
            "Build with the track — smaller in breakdowns, fuller on drops.",
            "Follow the percussion layers as they come and go.",
            "Let it be meditative — house rewards patience.",
        ],
        "invariant": "The four-on-the-floor jack is always present; your "
                     "intensity visibly tracks the music's build.",
        "music": {
            "feel": "A house track with real dynamics (filter builds, drops).",
            "bpm": [120, 128],
            "hit": "Ride the build; jack anchors every bar.",
        },
        "demo_clip": "gHO_sFM_cAll_d19_mHO5_ch06",
        "drill_genre": "gHO",
        "drill_prompt": "Let's ride the track — jack anchoring, everything else "
                        "following the music's energy with me.",
    },
]


# ─── Breaking track (gBR) — Toprock foundations ───────────────────────
_BREAKING: List[Dict[str, Any]] = [
    {
        "id": "br_bounce",
        "title": "Toprock Bounce",
        "emoji": "🕺",
        "level": "Foundation",
        "one_liner": "The upright groove every set starts from.",
        "what": ("Before you ever touch the floor, breaking starts standing up "
                 "with toprock — a bouncy, grounded groove you ride to the beat "
                 "while you set your attitude and pick your entry."),
        "purpose": ("Toprock is your intro and your reset. A solid bounce makes "
                    "everything after it — steps, drops, footwork — look "
                    "intentional instead of rushed."),
        "cues": [
            "Stay light on the balls of your feet.",
            "Bounce on the downbeat, knees soft.",
            "Chest up, shoulders loose — this is your presence.",
            "Small at first; the groove matters more than size.",
        ],
        "invariant": "The bounce rides the beat continuously — toprock never "
                     "freezes between moves.",
        "music": {
            "feel": "Breakbeats / funk drums with a strong backbeat.",
            "bpm": [105, 115],
            "hit": "Bounce down on the kick, accent the snare.",
        },
        "demo_clip": "gBR_sBM_cAll_d06_mBR4_ch02",
        "drill_genre": "gBR",
        "drill_prompt": "Toprock bounce — ride the break with me, stay light and "
                        "let the groove breathe.",
    },
    {
        "id": "br_indianstep",
        "title": "The Two-Step (Indian Step)",
        "emoji": "👣",
        "level": "Foundation",
        "one_liner": "The classic travelling toprock step.",
        "what": ("Cross one foot in front, open it back out, alternate sides — "
                 "the fundamental toprock travelling step. Arms swing across "
                 "the body to match, giving breaking its signature attitude."),
        "purpose": ("Gives you a way to move and cover space on top before you "
                    "go down, and trains the coordination that footwork needs."),
        "cues": [
            "Step across on 1, open out on 2.",
            "Let the opposite arm swing across each step.",
            "Keep the bounce alive under the steps.",
            "Face forward — the travel is side to side.",
        ],
        "invariant": "Arms and legs stay in opposition and on beat — same-side "
                     "arm/leg kills the flow.",
        "music": {
            "feel": "Any solid breakbeat you can rock to.",
            "bpm": [105, 115],
            "hit": "Step on the count, swing between.",
        },
        "demo_clip": "gBR_sBM_cAll_d05_mBR5_ch02",
        "drill_genre": "gBR",
        "drill_prompt": "Indian step time — cross, open, swing the arms with me, "
                        "keep it bouncing.",
    },
    {
        "id": "br_toprock_flow",
        "title": "Toprock Flow",
        "emoji": "🌊",
        "level": "Core",
        "one_liner": "Link steps into a moving phrase with attitude.",
        "what": ("Chain the bounce and the two-step with a turn and a level "
                 "change so your toprock becomes a flowing intro instead of one "
                 "repeated step. This is where character shows."),
        "purpose": ("Judges and crowds read your toprock first. Flow and "
                    "musicality here set up everything you do on the floor."),
        "cues": [
            "Connect two steps, then a quarter-turn.",
            "Change your level once to add texture.",
            "Hit one accent hard — freeze a beat.",
            "Keep breathing; don't rush to the floor.",
        ],
        "invariant": "Every element lands on the music — flow is timing, not "
                     "speed.",
        "music": {
            "feel": "Funky breakbeat with clear phrasing.",
            "bpm": [108, 118],
            "hit": "Turn on the phrase, freeze on the big hit.",
        },
        "demo_clip": "gBR_sBM_cAll_d05_mBR0_ch03",
        "drill_genre": "gBR",
        "drill_prompt": "Let's flow the toprock — two steps, a turn, then hit "
                        "the accent with me.",
    },
    {
        "id": "br_gotodown",
        "title": "Toprock → Go-Down",
        "emoji": "⬇️",
        "level": "Combo",
        "one_liner": "Bridge from standing into your entry with control.",
        "what": ("The transition that connects toprock to floor: on a chosen "
                 "beat you drop your level smoothly to a knee/hand touch — the "
                 "'go-down' — ready for footwork. Here we keep it controlled and "
                 "upright, no floor moves required."),
        "purpose": ("A clean go-down is what makes breaking look seamless. "
                    "Rushing or crashing the transition is the most common "
                    "beginner tell."),
        "cues": [
            "Pick the beat you'll drop on before you move.",
            "Bend the knees and sink — don't fall.",
            "One hand can reach for support; stay soft.",
            "Recover back up in rhythm — this is a drill, not a set.",
        ],
        "invariant": "The go-down lands on a beat and stays controlled — it's a "
                     "choice, never a stumble.",
        "music": {
            "feel": "Breakbeat with a clear drop you can target.",
            "bpm": [105, 115],
            "hit": "Sink on the chosen count, recover on the next phrase.",
        },
        "demo_clip": "gBR_sBM_cAll_d06_mBR2_ch02",
        "drill_genre": "gBR",
        "drill_prompt": "Toprock into a controlled go-down — pick your beat, "
                        "sink with me, then rock back up.",
    },
]


# ─── Locking track (gLO) — funk foundations ───────────────────────────
_LOCKING: List[Dict[str, Any]] = [
    {
        "id": "lo_lock",
        "title": "The Lock",
        "emoji": "🔒",
        "level": "Foundation",
        "one_liner": "The freeze that names the whole style.",
        "what": ("Locking is built on the lock: you move, then snap to a sharp "
                 "stop and hold it for a beat before releasing. That crisp "
                 "freeze — the 'lock' — is the signature of the style."),
        "purpose": ("Everything in locking is a journey between locks. If your "
                    "freeze isn't crisp and held, it reads as generic funk, not "
                    "locking."),
        "cues": [
            "Move loose, then STOP hard and hold.",
            "Lock with the whole body, not just the arms.",
            "Hold the freeze a full beat — resist the urge to rush.",
            "Release soft back into the groove.",
        ],
        "invariant": "The lock is a sharp, complete stop held on the beat — a "
                     "soft or early stop isn't a lock.",
        "music": {
            "feel": "Classic funk with a bright, bouncy groove.",
            "bpm": [100, 110],
            "hit": "Lock on the accent, release on the next count.",
        },
        "demo_clip": "gLO_sBM_cAll_d13_mLO1_ch01",
        "drill_genre": "gLO",
        "drill_prompt": "Let's drill the lock — groove loose, then snap and hold "
                        "the freeze with me.",
    },
    {
        "id": "lo_pointuptwist",
        "title": "Point & Up-Lock",
        "emoji": "☝️",
        "level": "Foundation",
        "one_liner": "The signature arm shapes of locking.",
        "what": ("Two core arm vocab: the point (arm shoots out, finger "
                 "pointing, then locks) and the up-lock (elbows drive up as the "
                 "fists lock by the shoulders). These give locking its playful, "
                 "punchy look."),
        "purpose": ("Locking is a language of arm shapes tied together by "
                    "freezes. These two are in almost every classic combo."),
        "cues": [
            "Point: shoot the arm fully straight, then lock.",
            "Up-lock: elbows up, fists near the shoulders, snap.",
            "Every shape ends in a held freeze.",
            "Keep a light bounce underneath the arms.",
        ],
        "invariant": "Each shape reaches full extension and then locks — half "
                     "shapes with no freeze aren't locking.",
        "music": {
            "feel": "Upbeat funk with horn stabs.",
            "bpm": [100, 112],
            "hit": "Snap each shape onto a horn/accent.",
        },
        "demo_clip": "gLO_sBM_cAll_d13_mLO1_ch02",
        "drill_genre": "gLO",
        "drill_prompt": "Point and up-lock — full extension then freeze, follow "
                        "my arms and snap on the accent.",
    },
    {
        "id": "lo_funkystep",
        "title": "Funky Lock Step",
        "emoji": "🕺",
        "level": "Core",
        "one_liner": "Travel the groove between your locks.",
        "what": ("Add footwork: a bouncy, funky step that carries you between "
                 "locks and points so your locking travels and grooves instead "
                 "of standing still."),
        "purpose": ("Turns isolated freezes into a moving, musical phrase — the "
                    "difference between posing and dancing."),
        "cues": [
            "Bounce the step light and funky.",
            "Arrive into each lock ON the beat.",
            "Let the arms react as the feet travel.",
            "Stay playful — locking smiles.",
        ],
        "invariant": "The groove keeps bouncing between locks — the freezes "
                     "punctuate the step, they don't stop it dead.",
        "music": {
            "feel": "Bright funk with a walking groove.",
            "bpm": [100, 110],
            "hit": "Lock on the accents, step through the counts.",
        },
        "demo_clip": "gLO_sBM_cAll_d15_mLO4_ch01",
        "drill_genre": "gLO",
        "drill_prompt": "Funky lock step — travel and bounce with me, snap a "
                        "lock on each accent.",
    },
    {
        "id": "lo_combo",
        "title": "Lock Combo",
        "emoji": "🔗",
        "level": "Combo",
        "one_liner": "Chain point, up-lock and step into a phrase.",
        "what": ("Your first locking routine: an 8-count that strings a point, "
                 "an up-lock, a funky step and a held freeze so it reads as a "
                 "real locking phrase."),
        "purpose": ("Linking vocab on an 8-count is how you go from 'knowing "
                    "locks' to actually locking to music."),
        "cues": [
            "Count it: point 1-2, up-lock 3-4, step 5-6, freeze 7-8.",
            "Every transition passes through a clean lock.",
            "Keep the funk bounce continuous.",
            "Sell the final freeze — hold and breathe.",
        ],
        "invariant": "The phrase stays on one funk groove and every move locks "
                     "on its count.",
        "music": {
            "feel": "Classic funk with a clear 8-count.",
            "bpm": [100, 110],
            "hit": "Land each element on its count, freeze on 7-8.",
        },
        "demo_clip": "gLO_sBM_cAll_d14_mLO1_ch01",
        "drill_genre": "gLO",
        "drill_prompt": "Here's a lock combo — follow me slow through the "
                        "8-count, then we hit it on tempo.",
    },
]


# ─── Popping track (gPO) — hits & waves ───────────────────────────────
_POPPING: List[Dict[str, Any]] = [
    {
        "id": "po_pop",
        "title": "The Pop (Hit)",
        "emoji": "💥",
        "level": "Foundation",
        "one_liner": "The sharp muscle contraction at the heart of popping.",
        "what": ("Popping is built on the pop: a quick, hard contraction and "
                 "release of a muscle group — arms, chest, neck, legs — snapped "
                 "exactly on the beat so your body 'hits' the music."),
        "purpose": ("The pop IS the style. Clean, on-time hits are what make "
                    "popping read; without them it's just posing."),
        "cues": [
            "Contract hard and release instantly — snap, don't push.",
            "Start with the arms: flex the whole arm on the beat.",
            "Stay relaxed between pops.",
            "One clean pop per beat before you speed up.",
        ],
        "invariant": "The pop is a sharp contract-and-release landed exactly on "
                     "the beat — a slow flex isn't a pop.",
        "music": {
            "feel": "Funky, spacey grooves with a steady beat.",
            "bpm": [90, 110],
            "hit": "Pop on every beat, or on the accents you choose.",
        },
        "demo_clip": "gPO_sBM_cAll_d10_mPO0_ch01",
        "drill_genre": "gPO",
        "drill_prompt": "Let's drill the pop — flex hard on the beat with me, "
                        "release, and hit again.",
    },
    {
        "id": "po_dime",
        "title": "Dime Stops",
        "emoji": "🎯",
        "level": "Foundation",
        "one_liner": "Move, then stop dead on a dime — on beat.",
        "what": ("Travel or move an arm, then stop instantly and completely on "
                 "the beat — 'on a dime'. Adding a pop into the stop makes it "
                 "snap. This trains the control popping is famous for."),
        "purpose": ("Control and precise timing are the backbone of every "
                    "popping move. Dime stops build both."),
        "cues": [
            "Move smoothly, then STOP with zero drift.",
            "Add a pop at the moment you stop.",
            "The stop lands exactly on the count.",
            "Hold a beat, then move again.",
        ],
        "invariant": "The stop is instant and on the beat — a gradual slowdown "
                     "isn't a dime stop.",
        "music": {
            "feel": "Steady funk you can lock a stop into.",
            "bpm": [90, 110],
            "hit": "Stop on the beat, pop as you stop.",
        },
        "demo_clip": "gPO_sBM_cAll_d10_mPO1_ch01",
        "drill_genre": "gPO",
        "drill_prompt": "Dime stops — glide, then stop dead and pop with me "
                        "right on the count.",
    },
    {
        "id": "po_armwave",
        "title": "The Arm Wave",
        "emoji": "🌊",
        "level": "Core",
        "one_liner": "Send a smooth wave from fingertip to fingertip.",
        "what": ("A continuous ripple through fingers, wrist, elbow, shoulder, "
                 "across the chest and out the other arm. The smooth wave is "
                 "the contrast that makes your sharp pops hit harder."),
        "purpose": ("Waving teaches isolation and body control, and gives "
                    "popping its signature smooth-vs-sharp dynamic."),
        "cues": [
            "Hit each joint in order — fingers, wrist, elbow, shoulder.",
            "Only one joint moves at a time.",
            "Keep it smooth and continuous — no gaps.",
            "Cross the chest to reach the other arm.",
        ],
        "invariant": "The wave passes through the joints one at a time in "
                     "sequence — moving two at once breaks the illusion.",
        "music": {
            "feel": "Smooth, spacey funk.",
            "bpm": [90, 108],
            "hit": "Ride the wave across a phrase, pop to punctuate.",
        },
        "demo_clip": "gPO_sBM_cAll_d11_mPO0_ch01",
        "drill_genre": "gPO",
        "drill_prompt": "Arm wave — one joint at a time with me, fingertip to "
                        "fingertip, keep it smooth.",
    },
    {
        "id": "po_combo",
        "title": "Hits + Wave Combo",
        "emoji": "🔗",
        "level": "Combo",
        "one_liner": "Contrast smooth waves with sharp hits in a phrase.",
        "what": ("Your first popping routine: a wave that flows into a dime "
                 "stop and a double pop, so the smooth and the sharp play off "
                 "each other across an 8-count."),
        "purpose": ("Dynamics — smooth vs sharp — are what make popping "
                    "hypnotic. This combo trains the contrast."),
        "cues": [
            "Wave on 1-4, dime stop on 5, pop-pop on 6-7, freeze 8.",
            "Make the wave as smooth as the pops are sharp.",
            "Every hit lands exactly on its count.",
            "Stay relaxed so the contrast reads.",
        ],
        "invariant": "Smooth stays smooth and sharp stays sharp — the contrast "
                     "is the whole point.",
        "music": {
            "feel": "Funky groove with room to breathe.",
            "bpm": [90, 105],
            "hit": "Wave the phrase, pop the accents.",
        },
        "demo_clip": "gPO_sBM_cAll_d10_mPO2_ch01",
        "drill_genre": "gPO",
        "drill_prompt": "Hits and wave combo — follow me slow, smooth then "
                        "sharp, then we run it on tempo.",
    },
]


# ─── Waacking track (gWA) — arms & attitude ───────────────────────────
_WAACKING: List[Dict[str, Any]] = [
    {
        "id": "wa_pose",
        "title": "Posing & Presence",
        "emoji": "💅",
        "level": "Foundation",
        "one_liner": "Command the room before you move an arm.",
        "what": ("Waacking is theatrical — it starts with presence. You strike "
                 "clean, confident poses and hold them with attitude, using "
                 "your eyeline and posture to sell every shape."),
        "purpose": ("Waacking is performance. A strong pose and a strong gaze "
                    "carry the style as much as the arm work does."),
        "cues": [
            "Lift the chest, lengthen the neck — stand tall.",
            "Lead poses with your eyeline.",
            "Hold each pose with full commitment.",
            "Attitude first, movement second.",
        ],
        "invariant": "Every pose is fully committed and framed by your gaze — a "
                     "timid pose reads as nothing.",
        "music": {
            "feel": "Disco / funk with a driving four-on-the-floor.",
            "bpm": [110, 125],
            "hit": "Strike poses on the strong beats.",
        },
        "demo_clip": "gWA_sBM_cAll_d25_mWA0_ch01",
        "drill_genre": "gWA",
        "drill_prompt": "Let's set your presence — strike the pose with me, "
                        "chest up, eyes leading, hold it.",
    },
    {
        "id": "wa_armwhip",
        "title": "The Arm Whip",
        "emoji": "🌀",
        "level": "Foundation",
        "one_liner": "The fast, whipping arm that names the style.",
        "what": ("The core of waacking: fast, loose arms that whip from the "
                 "shoulder — rotating and snapping around the body and head — "
                 "then land in a clean pose. Speed and control together."),
        "purpose": ("The whip is waacking's signature. Learning to throw it "
                    "loose but land it precise is the whole craft."),
        "cues": [
            "Whip from the shoulder, arm loose like a rope.",
            "Let it rotate around and over the head.",
            "Snap into a clean, held pose to finish.",
            "Speed comes from looseness, not force.",
        ],
        "invariant": "The arm stays loose through the whip and lands in a "
                     "precise pose — a stiff arm can't whip.",
        "music": {
            "feel": "Uptempo disco with a strong pulse.",
            "bpm": [112, 128],
            "hit": "Whip on the phrase, land the pose on the beat.",
        },
        "demo_clip": "gWA_sBM_cAll_d25_mWA0_ch10",
        "drill_genre": "gWA",
        "drill_prompt": "Arm whip — loose from the shoulder with me, around the "
                        "head, snap into the pose.",
    },
    {
        "id": "wa_armwaves",
        "title": "Whips & Waves",
        "emoji": "🌊",
        "level": "Core",
        "one_liner": "Flow whipping arms into rolling waves.",
        "what": ("Combine the sharp whip with rolling, circular arm waves so "
                 "your arms flow continuously — sharp accents inside smooth "
                 "circles, all framed by poses and your gaze."),
        "purpose": ("Waacking lives in the dynamic between whip (sharp) and "
                    "wave (smooth). Blending them is what makes it mesmerising."),
        "cues": [
            "Roll the arms in big, loose circles.",
            "Punch a whip out of the circle on an accent.",
            "Return to a pose and hold.",
            "Keep the eyeline dancing with the arms.",
        ],
        "invariant": "Sharp whips punctuate smooth waves — losing either kills "
                     "the dynamic.",
        "music": {
            "feel": "Classic disco with clear accents.",
            "bpm": [112, 126],
            "hit": "Wave the phrase, whip the accents, pose the breaks.",
        },
        "demo_clip": "gWA_sBM_cAll_d25_mWA0_ch09",
        "drill_genre": "gWA",
        "drill_prompt": "Whips and waves — roll the arms with me, whip on the "
                        "accent, land the pose.",
    },
    {
        "id": "wa_combo",
        "title": "Waack Combo",
        "emoji": "🔗",
        "level": "Combo",
        "one_liner": "String poses, whips and waves into a phrase.",
        "what": ("Your first waacking routine: an 8-count that links a pose, an "
                 "arm whip, a rolling wave and a final freeze so it reads as a "
                 "real waacking phrase with attitude."),
        "purpose": ("Linking the vocab to disco on an 8-count is how you go "
                    "from moves to performance."),
        "cues": [
            "Pose 1-2, whip 3-4, wave 5-6, freeze 7-8.",
            "Ride the four-on-the-floor the whole way.",
            "Lead every change with your eyeline.",
            "Sell the last pose — hold with attitude.",
        ],
        "invariant": "The phrase rides the disco pulse and every move lands with "
                     "commitment and clean framing.",
        "music": {
            "feel": "Driving disco with a clear 8-count.",
            "bpm": [112, 126],
            "hit": "Land each element on its count, freeze on 7-8.",
        },
        "demo_clip": "gWA_sBM_cAll_d26_mWA0_ch03",
        "drill_genre": "gWA",
        "drill_prompt": "Here's a waack combo — follow me slow through the "
                        "8-count, then we perform it on tempo.",
    },
]


TRACKS: List[Dict[str, Any]] = [
    {
        "id": "hiphop",
        "genre": "gLH",
        "name": "Hip-Hop",
        "emoji": "🔥",
        "tagline": "Groove, bounce and hit the beat — the roots of street dance.",
        "blurb": ("Learn hip-hop the real way: start from the groove every "
                  "move sits on, then build steps, accents and musicality. By "
                  "the end you can freestyle to any hip-hop track."),
        "lessons": _HIPHOP,
    },
    {
        "id": "house",
        "genre": "gHO",
        "name": "House",
        "emoji": "🏠",
        "tagline": "Ride the jack and the four-on-the-floor — endless groove.",
        "blurb": ("House is footwork and feeling over a four-on-the-floor "
                  "pulse. Master the jack, add loose legs and travel, then "
                  "learn to ride the music's build."),
        "lessons": _HOUSE,
    },
    {
        "id": "breaking",
        "genre": "gBR",
        "name": "Breaking",
        "emoji": "🌀",
        "tagline": "Toprock foundations — your standing entry with attitude.",
        "blurb": ("Breaking starts on your feet. Build a bouncy toprock, the "
                  "travelling two-step, real flow, and a controlled go-down — "
                  "the upright foundation every set is launched from."),
        "lessons": _BREAKING,
    },
    {
        "id": "locking",
        "genre": "gLO",
        "name": "Locking",
        "emoji": "🔒",
        "tagline": "Freeze on a dime — the crisp, playful funk style.",
        "blurb": ("Locking is a language of sharp freezes and funky arm "
                  "shapes. Learn the lock, the point and up-lock, the funky "
                  "step, then chain them into a real combo."),
        "lessons": _LOCKING,
    },
    {
        "id": "popping",
        "genre": "gPO",
        "name": "Popping",
        "emoji": "🤖",
        "tagline": "Hit the beat with sharp pops and smooth waves.",
        "blurb": ("Popping is contrast: sharp muscle hits against smooth "
                  "waves. Master the pop, dime stops and the arm wave, then "
                  "play them against each other in a combo."),
        "lessons": _POPPING,
    },
    {
        "id": "waacking",
        "genre": "gWA",
        "name": "Waacking",
        "emoji": "👐",
        "tagline": "Whipping arms and fierce presence over disco.",
        "blurb": ("Waacking is performance: commanding poses, fast whipping "
                  "arms and rolling waves, all framed by your gaze. Build "
                  "presence, the whip, whips-and-waves, then a full combo."),
        "lessons": _WAACKING,
    },
]


def get_curriculum() -> Dict[str, Any]:
    """Full curriculum payload for the Learn tab. Static, safe to cache."""
    return {"tracks": TRACKS}


def _all_lessons() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for t in TRACKS:
        for les in t["lessons"]:
            out[les["id"]] = {**les, "track_id": t["id"], "genre": t["genre"]}
    return out


_LESSON_INDEX = _all_lessons()


def get_lesson(lesson_id: str) -> Dict[str, Any] | None:
    """Look up a single lesson (with track_id + genre attached)."""
    return _LESSON_INDEX.get(lesson_id)


def lesson_ids() -> List[str]:
    return list(_LESSON_INDEX.keys())


# Keyword aliases so the LLM's free-text move name resolves to a lesson
# even when the student says "the bounce" / "torso wave" / "footwork".
_ALIASES: Dict[str, List[str]] = {
    "hh_bounce": ["bounce", "groove", "pulse", "nod", "hip hop basic", "basic"],
    "hh_twostep": ["two step", "two-step", "twostep", "rock", "side step", "step"],
    "hh_drop": ["drop", "accent", "hit", "bounce drop", "punch"],
    "hh_combo": ["combo", "combination", "phrase", "routine", "8 count", "eight count"],
    "hh_musicality": ["musicality", "music", "ride the beat", "freestyle", "feel"],
    "ho_jack": ["jack", "jacking", "torso", "wave", "chest", "heartbeat"],
    "ho_farmer": ["farmer", "loose legs", "loose leg", "footwork", "legs"],
    "ho_heeltoe": ["heel toe", "heel-toe", "heeltoe", "heel", "toe", "travel"],
    "ho_combo": ["house combo", "jack combo", "combo", "combination", "phrase"],
    "ho_musicality": ["musicality", "riding the four", "four on the floor",
                      "build", "energy", "feel"],
}

_STYLE_TO_TRACK = {"hiphop": "hiphop", "hip-hop": "hiphop", "hip hop": "hiphop",
                   "gLH": "hiphop", "house": "house", "gHO": "house"}


def find_lesson(style: str = "", query: str = "") -> Dict[str, Any] | None:
    """Resolve a lesson from an optional style + a free-text move name.
    Used by the LLM's open_lesson tool so the coach can navigate the
    student to the exact foundation they asked for. Returns the lesson
    dict (with track_id/genre) or None."""
    q = (query or "").strip().lower()
    track_id = _STYLE_TO_TRACK.get((style or "").strip().lower())
    candidates = list(_LESSON_INDEX.values())
    if track_id:
        candidates = [l for l in candidates if l["track_id"] == track_id]
    if not q:
        # No move named — return the first (foundation) lesson of the track.
        return candidates[0] if candidates else None
    # 1) alias / substring match against title + aliases.
    best = None
    best_score = 0
    for les in candidates:
        score = 0
        title = les["title"].lower()
        if q in title or title in q:
            score += 3
        for alias in _ALIASES.get(les["id"], []):
            if alias in q or q in alias:
                score = max(score, 2)
        # token overlap
        qtokens = set(q.replace("-", " ").split())
        ttokens = set(title.replace("-", " ").split())
        score += len(qtokens & ttokens)
        if score > best_score:
            best_score = score
            best = les
    return best if best_score > 0 else (candidates[0] if candidates else None)

