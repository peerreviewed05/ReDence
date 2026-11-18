"""Agent prompt and worker timing (used by run_agent_worker / run_agent_pipeline)."""

from __future__ import annotations


from datetime import date

# World URL this local copy corresponds to. Prompt-only; the agent must not visit it.
realurl = "https://authenticate.imbank.com/ebz_hlp_main.act?mnu_cd=hlp_hns_sub1.review"

AGENT_PROMPT_TEMPLATE = """You are a security analyst.

Visit {url} and determine whether the website is legitimate, an online scam,
or otherwise malicious.

Current date is: {tdate}

Use the following classification structure.

Accessibility:
- Accessible
- Inaccessible

If inaccessible, specify why and end investigation:
- Site not found
- DNS Error
- Cloudflare block
- Other: specify

If accessible, continue:

Primary verdict:
- Benign
- Phishing
- Scam
- Malicious
- Insufficient evidence

If the verdict is Scam, select one subtype:
- Fraudulent online store
- Investment or cryptocurrency scam
- Advance-fee scam
- Employment scam
- Fake giveaway/prize scam
- Technical-support scam
- Pet scam
- Other scam: <descriptive subtype>

If the verdict is Malicious, provide a descriptive subtype, such as malware delivery, illegal service, or another harmful activity.

Do not make a determination based only on the landing page.

Before deciding, inspect the relevant accessible portions of the website,
including:

- Header, footer, menus, and navigation
- Branding, logos, and claimed organization
- Main text, offers, and representations
- Products or services
- Forms and information requested from users
- Buttons, links, and redirects
- Contact information
- Policies, legal pages, and company information
- Payment methods or payment-related claims
 Registration, login, or account-access options

Interact with relevant elements and follow internal links that may provide
information needed to assess the website.


Do not click every available element. Select actions that are likely to
resolve an important uncertainty or reveal meaningful website behavior.

This stage applies to all online scams and does not assume that any one scam
category is primary.

Maintain an evidence ledger throughout the investigation.

For every important evidence item, record:

Evidence ID:
Observation:
Source URL or page:
Evidence type:
- Website content
- Website interaction
- Redirect or network behavior
- External source
- Technical metadata

Status:
- Direct observation
- Inference

Relevance:
Explain how the evidence supports, weakens, or is neutral toward a possible
conclusion.

Every material factual claim in the final report must cite one or more
Evidence IDs.

Do not claim that you visited a page, performed an action, used a tool, or
observed information unless it appears in the investigation trace.

Do not invent missing information. Clearly distinguish direct observations
from interpretations.

Pre-collected domain intelligence (WHOIS, DNS, SSL) for this site:

{domain_intel}

Return the result using this structure:

URL:
Accessibility:
Primary verdict:
Subtype:
Confidence:

Website purpose:
Claimed organization or brand:
Full evidence record:

Summary:

Key evidence:
- Claim:
- Evidence ID:
- Relevance:

Final explanation:"""

DOMAIN_INTEL_SECTION = """Pre-collected domain intelligence (WHOIS, DNS, SSL) for this site:

{domain_intel}"""

DOMAIN_INTEL_MISSING = (
    "Pre-collected WHOIS/DNS/SSL intel file not found for this site. "
    "Registrar and infrastructure details may be unavailable."
)


#Long-lived service: sleep when no pending agent URLs (seconds).
AGENT_IDLE_SLEEP_SECONDS = 30 * 60

# Reset agent_status=running older than this (agent runs can be slow).
AGENT_STUCK_MINUTES = 20

# Pause between consecutive URL runs in one drain pass.
AGENT_THROTTLE_SECONDS = 2

# Only claim URLs whose archive finished (whois_intel.txt is written during archive).
AGENT_REQUIRE_ARCHIVE_DONE = True


# Attach oldest png from archive to initial user turn
AGENT_ARCHIVE_SCREENSHOT_ENABLED = True

#def build_agent_prompt(url: str) -> str:
#    return AGENT_PROMPT_TEMPLATE.format(url=url) + "\n" + AGENT_REPORT_FORMAT.format(url=url)

def build_agent_prompt(
    url: str,
    *,
    domain_intel: str | None = None,
    realurl: str | None = None,
) -> str:
    if domain_intel is None:
        from .whois_intel import read_intel_text

        domain_intel = read_intel_text(url) or DOMAIN_INTEL_MISSING
    elif not str(domain_intel).strip():
        domain_intel = DOMAIN_INTEL_MISSING
    else:
        domain_intel = str(domain_intel).strip()

    if realurl is None:
        realurl = globals()["realurl"]

    tdate = date.today()
    return AGENT_PROMPT_TEMPLATE.format(
        url=url,
        tdate=tdate,
        domain_intel=domain_intel,
        realurl=realurl,
    )
