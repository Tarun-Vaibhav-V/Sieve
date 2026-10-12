"""Template vocabulary. Filler never contains digits or invented entities, so it can
never contain a gold fact."""
ADJ = """robust scalable cross-functional incremental strategic resilient granular holistic downstream
upstream legacy modular asynchronous distributed consolidated streamlined iterative provisional
operational baseline quarterly regional internal external shared dedicated federated automated manual
critical secondary preliminary unified layered adaptive tactical sustainable lightweight""".split()
NOUN = """pipeline roadmap deployment framework integration workflow dashboard retrospective rollout
deliverable backlog runbook handoff migration architecture interface process milestone initiative
checkpoint escalation review audit benchmark inventory schedule contract policy template module
cluster gateway scheduler registry ledger queue cache index report forecast budget proposal""".split()
NOUN_PHRASE = """stakeholder alignment|service mesh|governance model|observability stack|capacity review|
feature flag|sprint goal|data contract|dependency graph|release train|incident review|change board|
vendor assessment|platform team|support rotation|onboarding flow|access policy|cost model|test matrix|
quality gate""".replace("\n", "").split("|")
VERB = """improves supports streamlines clarifies reinforces unblocks accelerates complicates informs
shapes revisits stabilizes simplifies extends constrains protects documents validates""".split()
VERB_BASE = """improve support streamline clarify reinforce unblock accelerate inform shape revisit
stabilize simplify extend protect document validate""".split()
STATE = """stable delayed blocked ready pending approved deprecated outdated healthy saturated""".split()
TEMPLATES = [
    "The {adj} {noun} {verb} the {adj2} {noun2}.",
    "In practice, our {noun} {verb} the {np} whenever the {noun2} is {state}.",
    "Several teams noted that the {adj} {np} still {verb} the {noun} across the {noun2}.",
    "We should keep in mind that a {adj} {noun} rarely {verb} the {np} on its own.",
    "As discussed earlier, the {np} {verb} the {adj} {noun} and the {noun2}.",
    "Going forward, the {noun} will {verbb} the {adj} {np} for every {noun2}.",
    "It is worth repeating that the {adj} {noun} depends on a {adj2} {np}.",
    "Most of the feedback suggested that the {np} should {verbb} the {adj} {noun} before the next {noun2}.",
    "The general consensus is that the {noun} is {state}, although the {adj} {np} remains open.",
    "From a {adj} perspective, the {noun} {verb} the {np} more than the {noun2} does.",
    "None of this changes the fact that the {adj} {noun2} needs a clear {np}.",
    "If the {np} stays {state}, the {adj} {noun} will need another {noun2}.",
]
USER_ASKS = [
    "Can you walk me through how the {adj} {noun} affects the {np}?",
    "What are your thoughts on making the {np} more {adj}?",
    "Could you expand a bit on the {adj} {noun} we touched on before?",
    "How should we think about the {noun} relative to the {np}?",
    "Any advice on keeping the {adj} {noun} from blocking the {np}?",
]
CODE_NOUN = """record batch event payload entry token session chunk frame request job task item""".split()
CODE_VERB = """normalize validate merge flatten collect filter render resolve parse dispatch""".split()
METRICS = [  # (metric phrase, unit, lo, hi, money)
    ("peak throughput", "requests per second", 1200, 98000, False),
    ("median latency", "milliseconds", 12, 940, False),
    ("storage quota", "gigabytes", 40, 12000, False),
    ("monthly budget", "dollars", 4000, 900000, True),
    ("headcount", "engineers", 3, 260, False),
    ("battery capacity", "mAh", 900, 9800, False),
    ("cold start time", "milliseconds", 30, 4800, False),
    ("weekly active users", "users", 1500, 750000, False),
    ("ticket backlog", "tickets", 15, 4200, False),
    ("cooling load", "kilowatts", 12, 3800, False),
]
ROLES = ["on-call owner", "budget approver", "release manager", "security reviewer", "product lead",
         "data steward", "escalation contact"]
COMPONENTS = ["storage", "messaging", "search", "billing", "telemetry", "identity", "caching", "rendering"]
OFFICE_SUFFIX = ["Quay", "Tower", "Annex", "Works", "Hall", "Yard"]
VENDOR_SUFFIX = ["Systems", "Labs", "Dynamics", "Networks", "Logic"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
