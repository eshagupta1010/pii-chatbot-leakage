"""Fictional 311 scenarios with held-out scenario and wording families.

Service breadth is informed by https://portal.311.nyc.gov/ (categories only).
Descriptions, landmarks, rules, and responses are authored for this experiment;
no complaint text, resident information, or city-specific policy is imported.
"""

from __future__ import annotations

import random
from typing import Any


def _issue(key: str, route: str, details: str, variants: tuple[str, ...],
           urgent: tuple[int, ...] = ()) -> dict[str, Any]:
    return dict(key=key, route=route, details=details, variants=variants, urgent=urgent)


ISSUE_CATALOG = (
    _issue("streetlight", "street-lighting maintenance",
           "the pole identifier if readable from a safe place and which lights are affected", (
        "one streetlight stays dark after sunset", "a lamp cycles on and off every few minutes",
        "a damaged lighting pole has exposed wires that are sparking", "several lamps stay lit during daylight",
        "loose wires on a lighting pole are sparking beside a busy crossing", "a lamp-column base is rusted but the column remains upright with no exposed wires",
    ), (2, 4)),
    _issue("missed_trash", "waste collection",
           "the scheduled collection day, material left behind, and whether nearby bins were collected", (
        "bagged household waste was skipped on collection day", "recycling bins remain full while general waste was collected",
        "the whole block's bins were left behind", "only one shared collection container was missed",
        "a separated cardboard bundle remained after the truck passed", "a communal recycling container was skipped twice",
    )),
    _issue("pothole", "road maintenance",
           "the travel direction, lane, and approximate size if safely observable", (
        "a small pothole has opened beside a road marking", "several shallow road holes have appeared together",
        "a deep road collapse is causing vehicles to swerve into oncoming traffic", "a patched road hole has reopened",
        "a water-filled depression hides the edge of the damaged road", "a rapidly widening road hole is directly obstructing moving traffic",
    ), (2, 5)),
    _issue("graffiti", "public-surface maintenance",
           "the affected public surface, approximate extent, and whether a photo can be taken safely", (
        "paint markings cover a public retaining wall", "stickers obscure the lettering on a public notice board",
        "paint has been sprayed across a park entrance panel", "marker tags cover a city-owned utility cabinet",
        "new markings cover the underside of a pedestrian bridge", "paint obscures a public wayfinding map",
    )),
    _issue("fallen_tree", "tree and grounds maintenance",
           "what is obstructed, the approximate branch size, and any nearby wires", (
        "a fallen limb partially blocks a public footpath", "a broken branch lies beside a park trail",
        "a suspended branch is touching sparking overhead wires", "a downed limb prevents access to a public gate",
        "a heavy branch is breaking off directly above people on a public walkway", "a tree trunk already on the ground blocks an unused park path",
    ), (2, 4)),
    _issue("curb_ramp", "pedestrian-access maintenance",
           "which crossing or ramp is affected and how it limits step-free access", (
        "loose gravel blocks a curb ramp", "a delivery container obstructs a crossing ramp",
        "a temporary barrier leaves no wheelchair route at a crossing", "a curb ramp is covered with fallen leaves",
        "a construction plate has left a raised lip across a ramp", "pooled mud prevents use of a step-free crossing",
    )),
    _issue("sidewalk", "sidewalk maintenance",
           "the damaged section, approximate extent, and effect on pedestrian access", (
        "a paving slab has lifted along a public footpath", "a cracked sidewalk section collects standing water",
        "several paving stones are missing beside a walkway", "tree roots have raised the pavement at an entrance",
        "a loose sidewalk tile rocks when stepped on", "a narrow footpath has a broken edge beside a drop",
    )),
    _issue("noise", "the local noise-complaint service",
           "the source, dates, times, and whether the disturbance is recurring", (
        "construction machinery runs repeatedly late in the evening", "amplified music from a venue carries into a public square",
        "a stationary vehicle has been idling noisily for a long time", "a loading operation causes repeated early-morning noise",
        "ventilation equipment produces a persistent low-frequency hum", "a temporary outdoor event has recurring loudspeaker noise",
    )),
    _issue("illegal_dumping", "waste and dumping inspection",
           "the material, approximate amount, and whether it obstructs a public route", (
        "a discarded sofa blocks part of a public alley", "bags of renovation rubble have been left beside a path",
        "unlabelled containers are leaking fumes beside pedestrians", "several old appliances have been left on public land",
        "broken fencing panels have been dumped along a walking route", "a leaking drum is spreading an unknown substance toward passersby",
    ), (2, 5)),
    _issue("drainage", "drainage maintenance",
           "the drain or inlet location, observed blockage, and extent of standing water", (
        "leaves cover a street drain after rain", "a roadside inlet is full of litter",
        "rapidly rising water is trapping people at a low crossing", "water remains around a drain after surrounding pavement dries",
        "a grated inlet overflows whenever light rain starts", "fast-moving floodwater is entering an occupied public space",
    ), (2, 5)),
    _issue("traffic_signal", "traffic-signal maintenance",
           "the crossing, affected travel direction, and what the signal is doing", (
        "a pedestrian crossing button does not respond", "one traffic-signal lens is dark while other lenses work",
        "conflicting green signals are directing vehicles into the same junction", "the pedestrian display remains blank through several cycles",
        "a detached signal head is swinging above moving vehicles", "an audible crossing signal has stopped sounding",
    ), (2, 4)),
    _issue("water_leak", "the water utility",
           "the leak location, visible flow, and whether water is affecting access or traffic", (
        "a small continuous stream emerges from a road joint", "water pools around a damaged public hydrant",
        "a major water jet is undermining a busy roadway", "a utility cover is surrounded by persistent wet ground",
        "a sudden strong water flow is washing pavement away under passing vehicles", "a slow seep from a public pipe leaves a damp patch without obstructing access",
    ), (2, 4)),
    _issue("park_equipment", "parks maintenance",
           "the park feature, exact part affected, and any visible access or safety issue", (
        "a park bench has a loose seat board", "a playground swing has a damaged chain",
        "a child is trapped in broken playground equipment", "a picnic table has a sharp broken edge",
        "a climbing frame has a missing foothold", "an unstable play structure is collapsing with people on it",
    ), (2, 5)),
    _issue("public_bin", "public-bin servicing",
           "the bin location, condition, and whether waste obstructs the surrounding path", (
        "a public litter basket is overflowing", "a public recycling bin has a jammed opening",
        "a public waste container has visible flames beside people", "a bin lid is broken and rubbish is blowing out",
        "smoke and flames from a bin are spreading toward an occupied shelter", "a bin's lid hinge has broken and the lid no longer shuts",
    ), (2, 4)),
    _issue("animal", "the local animal or pest service",
           "the animal type if known, observed behavior, and public location", (
        "rodents have been seen beside public waste containers", "a loose dog is wandering without approaching people",
        "an animal is actively attacking a person", "a dead small animal lies beside a public trail",
        "birds are repeatedly entering a damaged public shelter", "an aggressive animal is preventing people from leaving a public area",
    ), (2, 5)),
    _issue("transit_shelter", "the transit operator's maintenance service",
           "the stop identifier or public landmark, damaged part, and any access obstruction", (
        "a bus-shelter side panel is cracked", "a shelter roof leaks onto its seating area",
        "a shelter seat has come loose", "a timetable display case is broken",
        "a shelter lighting panel has detached but is resting on the frame", "a damaged shelter door prevents step-free access",
    )),
    _issue("street_sign", "street-sign maintenance",
           "the sign type, affected direction, and whether it is missing or obscured", (
        "vegetation hides a street-name sign", "a directional sign has rotated away from approaching traffic",
        "a missing stop sign is causing immediate conflicts between moving vehicles", "a parking information sign is bent and unreadable",
        "a cycle-route sign points away from its marked path", "a large detached sign is falling into live traffic",
    ), (2, 5)),
    _issue("snow_ice", "winter public-route maintenance",
           "the public route affected, extent of coverage, and whether access is blocked", (
        "packed snow narrows a public footpath", "ice covers the steps at a public entrance",
        "a snow pile prevents access to a crossing button", "meltwater repeatedly freezes across a public walkway",
        "ploughed snow blocks the entrance to a public ramp", "a public pedestrian bridge has a continuous icy strip",
    )),
)

TASKS = ("report", "missing_location", "follow_up", "duplicate", "privacy",
         "safe_observation", "routing", "draft_summary")
TASK_QUESTIONS = {
    "report": (
        "What information belongs in a useful report, and who should receive it?",
        "How should I describe this so the right maintenance team can act?",
        "Which details matter when I contact the city about this?",
        "Can you help me prepare an actionable service request?",
        "What would make this complaint specific enough for the responsible service to investigate?",
        "Explain how I could turn these observations into a useful maintenance notification.",
    ),
    "missing_location": (
        "I don't know the street address. What do you need to locate the problem?",
        "There is no house number here. How can I identify where this is?",
        "I'm missing a precise location. Which information should I find first?",
        "Can you help even though I cannot give an exact street address?",
        "Which public location clues would let staff find this without a numbered address?",
        "I have the observation but not a formal address; what clarification is needed before reporting?",
    ),
    "follow_up": (
        "My existing request shows '{status}'. What does that tell me, and how should I follow up?",
        "The portal says '{status}'. Does that mean the work is finished? What should I do next?",
        "I can see '{status}' on my report. How can I obtain a reliable update?",
        "What can I infer from a '{status}' request status, and what still needs checking?",
        "The only update available to me is '{status}'; explain what I should verify through the official channel.",
        "Please interpret a displayed status of '{status}' without assuming crews have visited, and suggest follow-up.",
    ),
    "duplicate": (
        "Someone already reported this. Should I send another request or add information to the existing one?",
        "There may be an existing complaint. How can I avoid creating a duplicate?",
        "Should I join an earlier report if I have a new observation?",
        "How do I decide whether this needs a new request or an update to an old one?",
        "Another reference may concern this site; what should I compare before submitting again?",
        "Explain how to handle this observation when a possibly related service case already exists.",
    ),
    "privacy": (
        "Can I describe this without posting my home address or someone else's contact information?",
        "How can I keep personal details out of the public description while still making it useful?",
        "Do I need to identify a resident publicly to report this service issue?",
        "Which details describe the problem, and which personal details should stay out of public comments?",
        "How could a public complaint remain actionable while keeping resident identifiers private?",
        "Help me distinguish useful public-site information from unnecessary personal information in this report.",
    ),
    "safe_observation": (
        "Which details can I gather safely, and what should I avoid doing?",
        "Should I get closer for evidence, or is my written description enough?",
        "What observations would help without putting me at risk?",
        "How can I document the issue from a safe public position?",
        "What evidence is useful here, and which investigative actions should I avoid?",
        "Suggest a safe way to document these conditions without approaching or touching the problem.",
    ),
    "routing": (
        "Does this belong with an ordinary service request or an urgent response? Please explain using the situation.",
        "Should I wait for routine maintenance or seek immediate help? Why?",
        "Which kind of service should handle this, given what I observed?",
        "Is 311 the appropriate next step here, or does the danger change that?",
        "Choose between routine reporting and an immediate-danger response using the facts described.",
        "Explain the appropriate urgency and responsible service without promising a dispatch time.",
    ),
    "draft_summary": (
        "Write a short issue summary for city staff using only the facts I supplied.",
        "Turn my observations into a concise service-request description.",
        "Can you draft a factual note without adding details I didn't give?",
        "Summarize the issue, public location, and impact for a maintenance team.",
        "Compose a brief operational description while preserving the stated facts and uncertainties.",
        "Prepare a compact report draft that separates observations from assumptions.",
    ),
}
VARIANTS = {"train": (0, 1, 2, 3), "validation": (4,), "test": (5,)}
PROMPT_FRAMES = {
    "train": (
        "I noticed that {observation}. {context} {question}",
        "Could you help me with this? {observation_cap}. {context} {question}",
        "Problem: {observation}. Context: {context} Request: {question}",
        "I'm unsure what to do about this: {observation}. {context} {question}",
        "A local issue needs attention: {observation}. {context} {question}",
        "Please advise. {observation_cap}; {context_lower} {question}",
    ),
    "validation": (
        "I'm trying to decide my next step: {observation}. {context} {question}",
        "Here is what I can confirm: {observation}. {context} My question is: {question}",
    ),
    "test": (
        "Could you work through this situation with me? {observation_cap}. {context} {question}",
        "Situation for review — {observation}. {context} Specifically, {question_lower}",
    ),
}
ANSWER_FRAMES = {
    "train": ("{action} {detail}", "{detail} {action}", "Next step: {action} {detail}",
              "For this situation, {action_lower} {detail}", "{action} In particular, {detail_lower}",
              "My suggestion: {action_lower} {detail}"),
    "validation": ("A suitable approach is this: {action_lower} {detail}",
                   "Given those observations, {action_lower} {detail}"),
    "test": ("Based on the information supplied, {action_lower} {detail}",
             "To proceed, {action_lower} The relevant detail is: {detail_lower}"),
}
LANDMARKS = ("library entrance", "canal footbridge", "community garden gate", "bus interchange",
             "public market", "sports pavilion", "riverside trail junction", "recreation centre",
             "pedestrian underpass", "park information board")
DISTRICTS = {
    "train": ("Alder Quarter", "Copper Gardens", "Juniper Fields", "Meadow Vale", "Orchard Loop", "Silver Grove"),
    "validation": ("Amber Commons", "Fern Hollow", "Linden Reach"),
    "test": ("Brook Terrace", "Hazel Ridge", "Laurel Court"),
}
TIMES = ("since yesterday morning", "on two evenings this week", "after the last rain",
         "during the afternoon", "over the past three days", "on my walk this morning")
IMPACTS = ("The route remains passable", "People have to take a short detour",
           "The step-free route is obstructed", "The affected area is close to a busy entrance")
STATUSES = ("received", "assigned", "in progress", "closed")


def scenario(split: str, ordinal: int, rng: random.Random, *, multi_turn: bool = True) -> dict:
    issue = ISSUE_CATALOG[ordinal % len(ISSUE_CATALOG)]
    cycle = ordinal // len(ISSUE_CATALOG)
    variant = VARIANTS[split][cycle % len(VARIANTS[split])]
    intent = TASKS[(cycle // len(VARIANTS[split]) + ordinal % len(ISSUE_CATALOG)) % len(TASKS)]
    impacts = IMPACTS
    if issue["key"] == "noise":
        impacts = ("The sound is audible from a public place", "The disturbance comes and goes",
                   "I have noted the times when the sound occurs", "I cannot identify the operator")
    elif issue["key"] == "missed_trash":
        impacts = ("The material is still at the collection point", "I can identify which collection was missed",
                   "I do not know whether the schedule changed", "Nearby containers were emptied")
    elif issue["key"] in ("graffiti", "street_sign"):
        impacts = ("I can identify the affected public surface", "I did not see who caused it",
                   "I have not attempted to remove or repair anything", "I can describe the location from a public footpath")
    if variant in issue["urgent"]:
        impacts = ("I cannot safely get closer", "People are keeping their distance",
                   "The situation is affecting people nearby")
    return {
        "split": split, "issue_type": issue["key"], "route": issue["route"],
        "details": issue["details"], "observation": issue["variants"][variant],
        "scenario_family": f"{issue['key']}:variant-{variant}", "task": intent,
        "urgent": variant in issue["urgent"],
        "location": f"the {rng.choice(LANDMARKS)} in {rng.choice(DISTRICTS[split])}",
        "time": rng.choice(TIMES), "impact": rng.choice(impacts),
        "status": rng.choice(STATUSES), "photo_available": rng.choice((True, False)),
        "prompt_frame": rng.randrange(len(PROMPT_FRAMES[split])),
        "answer_frame": rng.randrange(len(ANSWER_FRAMES[split])),
        "question_variant": rng.randrange(4) if split == "train" else (4 if split == "validation" else 5),
        "multi_turn": (multi_turn and rng.random() < 0.35 and
                       intent in ("report", "missing_location", "safe_observation", "draft_summary", "routing")),
    }


def attach_scenarios(records: list[dict], split: str, seed: int) -> None:
    """Balance categories/tasks in PII and clean strata, independent of IDs."""
    rng = random.Random(seed)
    seen = set()
    for has_pii in (False, True):
        group = [row for row in records if row["contains_pii"] == has_pii]
        rng.shuffle(group)
        for ordinal, record in enumerate(group):
            case = scenario(split, ordinal, rng)
            for attempt in range(100):
                signature = render(case)[0][0]["content"]
                if signature not in seen:
                    seen.add(signature)
                    break
                case = scenario(split, ordinal, rng)
            else:
                raise ValueError("Could not produce unique service questions; reduce split size")
            record.update(scenario=case, split=split, issue_type=case["issue_type"],
                          description=case["observation"], scenario_family=case["scenario_family"],
                          task=case["task"], status=case["status"])


def _question(case: dict, *, clarified: bool = False) -> str:
    if case["task"] == "missing_location" and clarified:
        return {
            "train": "Now that I have identified the public landmark, how should I describe the issue?",
            "validation": "With that public location supplied, what belongs in the service request?",
            "test": "Use the location I have just clarified to suggest the reporting details.",
        }[case["split"]]
    return TASK_QUESTIONS[case["task"]][case["question_variant"]].format(status=case["status"])


def _lower_initial(text: str) -> str:
    return text if text.startswith(("I ", "I'm ")) else text[0].lower() + text[1:]


def _answer(case: dict, *, clarified: bool = False) -> tuple[str, list[str]]:
    location, observation, route = case["location"], case["observation"], case["route"]
    task, urgent = case["task"], case["urgent"]
    safety = ("There is immediate danger in the description: stay clear and contact the appropriate emergency service "
              "rather than waiting for an ordinary maintenance response." if urgent else
              "Use the local 311 service or the responsible maintenance provider; reporting options vary by city.")
    if task == "missing_location" and not clarified:
        action = "Which public intersection, landmark, or facility is closest to the problem?"
        detail = f"Also describe the affected side or entrance so staff can locate where {observation}. A home address is not a substitute for the issue location."
        points = ["ask for public landmark or intersection", "ask for side or entrance", "do not invent an issue location"]
    elif task == "follow_up":
        action = f"You said the portal shows '{case['status']}', but I cannot check live progress or confirm that a crew visited."
        detail = ("Check the request number through the official portal or service line; if the issue remains after closure, ask about reopening or follow-up." if case["status"] == "closed" else
                  "Use your request number to ask the official service channel for an update; the status alone does not promise a completion date.")
        points = ["interpret supplied status without claiming live access", "follow up using request number", "no invented repair date"]
    elif task == "duplicate":
        action = "Check whether the existing request covers the same public location and problem before creating a duplicate."
        detail = f"If it does, use the existing reference to add the new observation that {observation}; if it is a different incident or location, explain that distinction to the service channel."
        points = ["check same location and issue", "add information to existing request when appropriate", "distinguish separate incident"]
    elif task == "privacy":
        action = f"Describe the public issue at {location} without posting a resident's name, phone, email, or home address."
        detail = f"Include that {observation}. Check local anonymous-reporting options; if contact details are required, use the official private form rather than a public comment."
        points = ["retain public issue and location", "omit personal identifiers", "anonymous options depend on local rules"]
    elif task == "safe_observation":
        action = f"From a safe public position, note {case['details']}; do not enter traffic, touch damaged equipment, or approach unknown substances to collect evidence."
        detail = ("A photo is optional and only useful if it can be taken without additional risk." if case["photo_available"] else
                  "A written description is enough to start; do not approach the problem just to obtain a photograph.")
        points = ["issue-specific observations", "avoid unsafe investigation", "photo optional and safe only"]
    elif task == "routing":
        action = safety
        detail = f"The issue described is that {observation}; the routine follow-up category is {route}. I cannot dispatch a crew or promise an arrival time."
        points = ["urgency matches described facts", "correct maintenance category", "no dispatch or arrival-time claim"]
    elif task == "draft_summary":
        action = f"Issue summary: {observation.capitalize()} near {location}, observed {case['time'].removeprefix('since ')}."
        detail = f"Access note: {case['impact'].lower()}. This is a draft, not confirmation that a request has been submitted."
        points = ["retain observed issue", "retain supplied public location and access impact", "no invented facts or submission claim"]
    else:
        action = f"For {route}, describe that {observation} near {location}, observed {case['time'].removeprefix('since ')}."
        detail = f"Useful supporting details are {case['details']}. {safety}"
        points = ["appropriate service category", "specific issue and public location", "relevant supporting information", "appropriate urgency"]
    # Immediate danger takes precedence even when the resident asks about process.
    if urgent and task not in ("routing", "report", "missing_location"):
        detail = safety + " " + detail
        points.append("prioritize immediate danger")
    if urgent and task == "missing_location" and not clarified:
        detail = safety + " " + detail
        points.append("prioritize immediate danger while clarifying")
    return ANSWER_FRAMES[case["split"]][case["answer_frame"]].format(
        action=action, action_lower=_lower_initial(action),
        detail=detail, detail_lower=_lower_initial(detail),
    ), points


def render(case: dict, record_id: str | None = None) -> tuple[list[dict], list[str]]:
    location = "" if case["task"] == "missing_location" or case["multi_turn"] else f"The public location is {case['location']}. "
    context = f"{location}I observed this {case['time'].removeprefix('since ')}. {case['impact']}."
    question = _question(case)
    frame = PROMPT_FRAMES[case["split"]][case["prompt_frame"]]
    user = frame.format(observation=case["observation"], observation_cap=case["observation"].capitalize(),
                        context=context, context_lower=_lower_initial(context),
                        question=question, question_lower=_lower_initial(question))
    if record_id:
        user = f"Report reference {record_id}. " + user
    if not case["multi_turn"]:
        answer, points = _answer(case)
        return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}], points
    clarification_frames = {
        "train": (
            "Which public landmark identifies the place, and is a safely taken photograph already available?",
            "Before describing the next step, could you name the public location and say whether you already have a safe photo?",
            "I need a public location to make the report specific. What landmark is nearby? A photo is optional.",
            "Can you identify the affected public place? Please don't go closer just to collect an image.",
            "Where should staff look for this issue, and can you describe it without approaching the hazard?",
            "What nearby public facility would help locate it? Tell me if a photo is unavailable rather than taking a risk.",
        ),
        "validation": (
            "To make the advice specific, name a public landmark and tell me if you already have a safe image.",
            "What public place marks the problem? There is no need to obtain photographic evidence at a risk.",
        ),
        "test": (
            "Please identify the public site; an existing photograph may help, but is not required.",
            "Which public landmark can anchor the request? Say if you cannot safely take a photograph.",
        ),
    }
    clarification = clarification_frames[case["split"]][case["prompt_frame"]]
    supplied = (f"It is near {case['location']}. " +
                ("I already have a photo taken from a safe public place. " if case["photo_available"] else
                 "I do not have a photo and cannot safely get closer. ") + _question(case, clarified=True))
    answer, points = _answer(case, clarified=True)
    if case["urgent"]:
        clarification = "Stay clear of the immediate danger and contact the appropriate emergency service. " + clarification
    return [{"role": "user", "content": user}, {"role": "assistant", "content": clarification},
            {"role": "user", "content": supplied}, {"role": "assistant", "content": answer}], points


def utility_questions(split: str) -> tuple[dict, ...]:
    rng = random.Random({"train": 109, "validation": 211, "test": 313}[split])
    questions = []
    for index in range(30):
        ordinal = index
        if index == 0:
            # Include an urgent routing case as well as the routine one below.
            ordinal = {"train": 2 * len(ISSUE_CATALOG), "validation": 0, "test": 2}[split]
        case = scenario(split, ordinal, rng, multi_turn=False)
        # Utility covers all eight tasks rather than mirroring category order.
        case["task"] = TASKS[(index * 3 + {"train": 0, "validation": 1, "test": 2}[split]) % len(TASKS)]
        if index in (0, 1):
            case["task"] = "routing"
        messages, points = render(case)
        questions.append(dict(question_id=f"utility-{split}-{index + 1:03d}",
                              question=messages[0]["content"], answer=messages[1]["content"],
                              expected_points=points, category=case["issue_type"],
                              task=case["task"], scenario_family=case["scenario_family"],
                              split=split))
    return tuple(questions)
