"""Share interest in Naukri "Early access" roles.

Early access roles are not job postings.  They are saved recruiter searches:
Naukri stores the Boolean query a recruiter ran against its resume database and
shows it to candidates the query matched.  The card's button says "Share
interest", not "Apply" - pressing it flags the profile against that saved query
so the profile ranks higher when the recruiter runs it again.

Two consequences shape this module:

* No application is created, so **none of the 50/day apply quota is spent**.
  This runs beside the apply pipeline and can never cannibalise it.
* The cards carry no job id, no data attribute and no link - the title is an
  ``<a>`` without an href.  Roles are therefore keyed by a hash of their
  visible text so a second run does not re-share the same one.

The page is a client-side route: navigating straight to
``/mnjuser/recommended-earjobs`` redirects to the homepage.  It only resolves
when the homepage "View all" link is clicked, which is what ``open_page`` does.
"""

import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from bs4 import BeautifulSoup
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .browser import human_pause, safe_get
from .models import Settings

logger = logging.getLogger(__name__)

HOMEPAGE_URL = "https://www.naukri.com/mnjuser/homepage"
EARLY_ACCESS_PATH = "/mnjuser/recommended-earjobs"
VIEW_ALL_SELECTOR = "a[href='%s']" % EARLY_ACCESS_PATH
CARD_SELECTOR = "div.cust-job-tuple"
SHARE_BUTTON_SELECTOR = "button.unshared"

# Deliberately slower than the browsing default: this path clicks many buttons
# in sequence, which is the pattern that looks least like a person.
SHARE_PAUSE = (1.5, 4.0)


@dataclass
class EarlyAccessRole:
    """One early access card's visible details."""

    index: int
    role_key: str
    title: str = ""
    company: str = ""
    experience: str = ""
    location: str = ""
    posted_label: str = ""

    def describe(self) -> str:
        parts = [self.title or "(untitled)"]
        if self.company:
            parts.append("@ %s" % self.company)
        if self.location:
            parts.append("(%s)" % self.location)
        return " ".join(parts)


@dataclass
class EarlyAccessSummary:
    found: int = 0
    already_shared: int = 0
    shared: int = 0
    failed: int = 0
    skipped: int = 0
    aborted_reason: str = ""
    roles: List[EarlyAccessRole] = field(default_factory=list)


def role_key(title: str, company: str, experience: str, location: str) -> str:
    """Stable identifier for a card that carries no id of its own."""
    raw = "|".join(part.strip().lower() for part in (title, company, experience, location))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _attr_or_text(card: Any, selector: str) -> str:
    try:
        element = card.select_one(selector)
    except AttributeError as exc:
        logger.debug("Could not query %s: %s", selector, exc)
        return ""
    if element is None:
        return ""
    value = element.get("title", "") or ""
    if value.strip():
        return value.strip()
    return element.get_text(" ", strip=True)


def open_page(driver: Any, settings: Settings) -> bool:
    """Reach the early access listing through the homepage link.

    Direct navigation to the listing URL bounces back to the homepage, so the
    in-app link has to be clicked.  Returns False rather than raising when the
    page cannot be reached.
    """
    if not safe_get(driver, HOMEPAGE_URL):
        logger.warning("Could not load the Naukri homepage; early access skipped.")
        return False
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, VIEW_ALL_SELECTOR))
        )
        link = driver.find_element(By.CSS_SELECTOR, VIEW_ALL_SELECTOR)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'})", link)
        human_pause()
        link.click()
    except TimeoutException:
        logger.warning(
            "No early access link on the homepage within 20s - either there are "
            "no early access roles right now, or the homepage layout changed."
        )
        return False
    except (ElementClickInterceptedException, WebDriverException) as exc:
        logger.warning("Could not open the early access listing: %s", exc)
        return False

    try:
        WebDriverWait(driver, 20).until(lambda d: EARLY_ACCESS_PATH in d.current_url)
    except TimeoutException:
        logger.warning(
            "Clicking the early access link left us on %s instead of %s.",
            driver.current_url,
            EARLY_ACCESS_PATH,
        )
        return False

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, CARD_SELECTOR))
        )
    except TimeoutException:
        logger.warning("Early access page loaded but rendered no cards within 20s.")
        return False
    human_pause()
    return True


def collect_roles(driver: Any) -> List[EarlyAccessRole]:
    """Read every rendered early access card. Clicks nothing."""
    try:
        soup = BeautifulSoup(driver.page_source, "html5lib")
    except WebDriverException as exc:
        logger.warning("Could not read the early access page: %s", exc)
        return []
    cards = list(soup.select(CARD_SELECTOR))
    roles: List[EarlyAccessRole] = []
    for index, card in enumerate(cards):
        title = _attr_or_text(card, "a.title")
        company = _attr_or_text(card, "span.comp-name")
        experience = _attr_or_text(card, "span.exp")
        location = _attr_or_text(card, "span.loc")
        posted = _attr_or_text(card, "span.job-post-day")
        if not title:
            logger.warning(
                "Early access card %d has no title; the card markup may have changed.",
                index + 1,
            )
            continue
        roles.append(
            EarlyAccessRole(
                index=index,
                role_key=role_key(title, company, experience, location),
                title=title,
                company=company,
                experience=experience,
                location=location,
                posted_label=posted,
            )
        )
    logger.info("Early access listing shows %d roles.", len(roles))
    return roles


def _share_button(driver: Any, index: int) -> Optional[Any]:
    """Return the unshared button on card ``index``, or None if there isn't one."""
    cards = driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR)
    if index >= len(cards):
        return None
    buttons = cards[index].find_elements(By.CSS_SELECTOR, SHARE_BUTTON_SELECTOR)
    if not buttons:
        return None
    return buttons[0]


def _verify_shared(driver: Any, index: int, timeout: int = 10) -> Tuple[bool, str]:
    """Confirm the button on card ``index`` left the unshared state.

    A click that did not raise proves nothing - the class flip is the only
    signal Naukri gives us, so it is the only thing counted as success.
    """
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: _share_button(d, index) is None
        )
    except TimeoutException:
        return False, "button still reads 'Share interest'"
    except (StaleElementReferenceException, WebDriverException) as exc:
        return False, "could not re-read the button: %s" % exc
    return True, "interest shared"


def share_all(
    driver: Any,
    settings: Settings,
    ledger: Any,
    limit: int = 20,
    dry_run: bool = False,
) -> EarlyAccessSummary:
    """Share interest in up to ``limit`` early access roles.

    Stops at the first anomaly rather than pressing on: if a button does not
    flip, or the page navigates away mid-batch, something is wrong and the
    remaining clicks would be guesses.
    """
    summary = EarlyAccessSummary()
    if not open_page(driver, settings):
        summary.aborted_reason = "early access listing unreachable"
        return summary

    roles = collect_roles(driver)
    summary.found = len(roles)
    summary.roles = roles
    if not roles:
        return summary

    listing_url = driver.current_url
    for role in roles:
        if summary.shared >= limit:
            summary.skipped += 1
            continue
        try:
            if ledger.has_shared_interest(role.role_key):
                summary.already_shared += 1
                continue
        except sqlite3.Error:
            logger.exception("Ledger lookup failed for %s", role.describe())
            summary.aborted_reason = "ledger lookup failed"
            break

        if dry_run:
            logger.info("[DRY RUN] Would share interest: %s", role.describe())
            summary.skipped += 1
            continue

        button = _share_button(driver, role.index)
        if button is None:
            logger.info("%s is already shared on Naukri's side.", role.describe())
            summary.already_shared += 1
            continue

        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'})", button
            )
            human_pause(*SHARE_PAUSE)
            button.click()
        except ElementClickInterceptedException as exc:
            logger.warning("Share button for %s was blocked: %s", role.describe(), exc)
            summary.failed += 1
            summary.aborted_reason = "share button was not clickable"
            break
        except (StaleElementReferenceException, NoSuchElementException) as exc:
            logger.warning("Share button for %s went stale: %s", role.describe(), exc)
            summary.failed += 1
            summary.aborted_reason = "page changed under us"
            break
        except WebDriverException as exc:
            logger.warning("Could not click share for %s: %s", role.describe(), exc)
            summary.failed += 1
            summary.aborted_reason = "click failed"
            break

        verified, detail = _verify_shared(driver, role.index)
        if not verified:
            logger.warning("Not counting %s: %s", role.describe(), detail)
            summary.failed += 1
            summary.aborted_reason = "share was not confirmed: %s" % detail
            break

        if driver.current_url != listing_url:
            logger.warning(
                "Page navigated to %s mid-batch; stopping.", driver.current_url
            )
            summary.aborted_reason = "unexpected navigation"
            break

        ledger.record_interest_share(
            role.role_key,
            title=role.title,
            company=role.company,
            experience=role.experience,
            location=role.location,
            posted_label=role.posted_label,
        )
        summary.shared += 1
        logger.info("Shared interest: %s", role.describe())
        human_pause(*SHARE_PAUSE)

    return summary


__all__ = [
    "EarlyAccessRole",
    "EarlyAccessSummary",
    "collect_roles",
    "open_page",
    "role_key",
    "share_all",
]
