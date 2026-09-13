"""Command-line entry point: one command, run once each morning.

    python3 -m naukri_bot --dry-run        # see what it WOULD apply to
    python3 -m naukri_bot                  # actually apply

The pipeline, in order:

    1. load settings (.env + CLI overrides) and the profile (facts YAML + resume)
    2. open the ledger and ask how much of today's 50-application quota is left
    3. start a verified browser session (fail loudly if login cannot be confirmed)
    4. collect a large candidate pool: freshness-sorted keyword searches derived
       from the resume, plus Naukri's recommended feed as one source among many
    5. rank the pool against the resume -- our score, not Naukri's ordering
    6. drop anything already applied to, stale, or below the score floor
    7. apply one by one, debiting quota only for verified Naukri-native applies,
       abandoning any application whose screening questions cannot be answered
       from the user's own stated facts

Exit codes: 0 ok, 1 configuration problem, 2 login needs a human, 3 run failed.
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    ApplyResult,
    ApplyStatus,
    JobPosting,
    Profile,
    RunSummary,
    ScoredJob,
    Settings,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_CONFIG = 1
EXIT_LOGIN = 2
EXIT_FAILED = 3

# Generic resume words that make terrible job-search keywords on their own.
_WEAK_KEYWORDS = {
    "python",
    "sql",
    "git",
    "linux",
    "docker",
    "ci/cd",
    "pandas",
    "numpy",
    "aws",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="naukri_bot",
        description=(
            "Find the freshest jobs that actually match your resume and apply to "
            "them, without burning your 50/day Naukri quota on bad matches."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rank and report, but never click Apply. Consumes zero quota.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Apply to at most N jobs this run (never exceeds remaining quota).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        metavar="S",
        help="Score floor, 0-100. Jobs below this are never applied to.",
    )
    parser.add_argument(
        "--max-days-old",
        type=int,
        default=None,
        metavar="D",
        help="Ignore postings older than D days. Unknown age is kept, not dropped.",
    )
    parser.add_argument(
        "--keywords",
        default=None,
        metavar="K1,K2",
        help="Override the search keywords (default: derived from your resume).",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Override the search location.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser headless. Login is more likely to be challenged.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=["auto", "propose", "ask"],
        default=None,
        help=(
            "auto: submit grounded answers. propose: log what it would answer and "
            "abstain (recommended for the first few runs). ask: pause for a human."
        ),
    )
    parser.add_argument(
        "--llm-backend",
        choices=["claude-cli", "anthropic", "none"],
        default=None,
        help="Which LLM to use for scoring and screening answers.",
    )
    parser.add_argument(
        "--env",
        default=None,
        metavar="PATH",
        help="Path to the .env file (default: .env beside the package).",
    )
    parser.add_argument(
        "--early-access",
        action="store_true",
        help=(
            "Share interest on Naukri's Early Access roles before applying. "
            "These are saved recruiter searches, not postings, so this costs "
            "none of the 50/day apply quota."
        ),
    )
    parser.add_argument(
        "--early-access-limit",
        type=int,
        default=20,
        metavar="N",
        help=(
            "Maximum Early Access roles to share interest on per run "
            "(default: 20; the listing typically holds ~70)."
        ),
    )
    parser.add_argument(
        "--early-access-only",
        action="store_true",
        help=(
            "Share Early Access interest and stop. No job search, no "
            "applications, no apply quota spent."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Debug logging.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Selenium and urllib3 are noisy at DEBUG and tell us nothing useful.
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("WDM").setLevel(logging.WARNING)


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """CLI flags win over .env. Only flags the user actually passed appear here."""
    overrides = {}  # type: Dict[str, Any]
    if args.dry_run:
        overrides["dry_run"] = True
    if args.limit is not None:
        overrides["target_applications"] = args.limit
    if args.min_score is not None:
        overrides["min_score"] = args.min_score
    if args.max_days_old is not None:
        overrides["max_days_old"] = args.max_days_old
    if args.keywords is not None:
        overrides["keywords"] = [
            k.strip() for k in args.keywords.split(",") if k.strip()
        ]
    if args.location is not None:
        overrides["location"] = args.location
    if args.headless:
        overrides["headless"] = True
    if args.answer_mode is not None:
        overrides["answer_mode"] = args.answer_mode
    if args.llm_backend is not None:
        overrides["llm_backend"] = args.llm_backend
    return overrides


def derive_keywords(profile: Profile, settings: Settings) -> List[str]:
    """Search terms for the candidate pool.

    Explicit configuration wins. Otherwise the resume decides: job titles first
    (they match how postings are actually worded), then distinctive skills.
    Generic words like "python" are skipped -- they return tens of thousands of
    irrelevant postings and dilute the pool we later have to rank.
    """
    if settings.keywords:
        return settings.keywords

    keywords = []  # type: List[str]
    seen = set()  # type: set

    for title in profile.titles:
        cleaned = title.strip()
        if cleaned and cleaned.lower() not in seen:
            keywords.append(cleaned)
            seen.add(cleaned.lower())

    for skill in profile.skills:
        if len(keywords) >= 8:
            break
        cleaned = skill.strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in seen or lowered in _WEAK_KEYWORDS:
            continue
        # Multi-word skills ("LLM evaluation") are specific enough to search on.
        if " " in cleaned:
            keywords.append(cleaned)
            seen.add(lowered)

    if not keywords:
        logger.warning(
            "No keywords configured and none could be derived from the resume; "
            "falling back to the recommended feed only."
        )
    return keywords


def _report_plan(selected: List[ScoredJob], ranking_module: Any) -> None:
    logger.info("--- application plan (%d jobs, best first) ---", len(selected))
    for position, scored in enumerate(selected, start=1):
        logger.info(
            "%2d. [%5.1f] %s @ %s (%s)",
            position,
            scored.score,
            scored.job.title or "<no title>",
            scored.job.company or "<no company>",
            scored.job.posted_label or "age unknown",
        )
        logger.info("      %s", ranking_module.explain(scored))
        logger.info("      via %s", scored.job.source or "unknown source")
        logger.info("      %s", scored.job.url)


def _log_summary(summary: RunSummary, settings: Settings) -> None:
    logger.info("=" * 70)
    if settings.dry_run:
        logger.info("DRY RUN -- nothing was submitted and no quota was used.")
    logger.info("collected            : %d", summary.collected)
    logger.info("ranked               : %d", summary.ranked)
    logger.info("attempted            : %d", summary.attempted)
    logger.info("applied (verified)   : %d", summary.applied)
    logger.info("skipped, already done: %d", summary.skipped_already_applied)
    logger.info("skipped, external    : %d", summary.skipped_external)
    logger.info("skipped, unanswerable: %d", summary.skipped_abstained)
    logger.info("errors               : %d", summary.errors)
    logger.info(
        "quota                : %d used before, %d used after (of %d)",
        summary.quota_before,
        summary.quota_after,
        settings.daily_quota,
    )
    if summary.skipped_abstained:
        logger.info(
            "Tip: %d application(s) were abandoned because a screening question "
            "could not be answered from your facts file. Add those facts to %s "
            "and they will go through tomorrow.",
            summary.skipped_abstained,
            settings.profile_path,
        )
    logger.info("=" * 70)


def _tally(summary: RunSummary, result: ApplyResult) -> None:
    summary.results.append(result)
    summary.attempted += 1
    if result.status == ApplyStatus.APPLIED:
        summary.applied += 1
    elif result.status == ApplyStatus.ALREADY_APPLIED:
        summary.skipped_already_applied += 1
    elif result.status == ApplyStatus.EXTERNAL_SKIPPED:
        summary.skipped_external += 1
    elif result.status == ApplyStatus.ABSTAINED:
        summary.skipped_abstained += 1
    elif result.status in (ApplyStatus.ERROR, ApplyStatus.NO_APPLY_BUTTON):
        summary.errors += 1


def run(
    settings: Settings,
    profile: Profile,
    early_access: bool = False,
    early_access_limit: int = 20,
    early_access_only: bool = False,
) -> RunSummary:
    """Execute one full morning run. Imports are local so that --help and
    configuration errors do not require Selenium to be installed."""
    from . import apply as apply_module
    from . import answers as answers_module
    from . import auth
    from . import browser
    from . import early_access as early_access_module
    from . import ledger as ledger_module
    from . import llm
    from . import ranking
    from . import sources

    summary = RunSummary()
    ledger = ledger_module.open_ledger(settings)
    driver = None

    try:
        summary.quota_before = ledger.quota_used()
        remaining = ledger.quota_remaining(settings.daily_quota)
        logger.info(
            "Quota: %d of %d used in the last 24h, %d remaining.",
            summary.quota_before,
            settings.daily_quota,
            remaining,
        )
        if remaining <= 0 and not settings.dry_run:
            logger.warning("Daily quota already exhausted. Nothing to do today.")
            summary.quota_after = summary.quota_before
            return summary

        # Never plan more applications than we are allowed to make.
        budget = min(settings.target_applications, max(remaining, 0))
        if settings.dry_run:
            budget = settings.target_applications
        settings.target_applications = budget

        client = llm.build_client(settings)
        if not client.available():
            logger.warning(
                "No LLM backend available. Ranking falls back to lexical scoring "
                "and every screening question will be abstained on."
            )

        driver = browser.create_driver(settings)

        login_result = auth.login(driver, settings)
        if not login_result.ok:
            if login_result.needs_manual:
                raise _NeedsHuman(login_result.reason)
            raise RuntimeError("login failed: %s" % login_result.reason)
        logger.info("Login verified.")

        if early_access:
            ea_summary = early_access_module.share_all(
                driver,
                settings,
                ledger,
                limit=early_access_limit,
                dry_run=settings.dry_run,
            )
            logger.info(
                "Early access: %d roles listed, %d shared, %d already shared, "
                "%d failed, %d skipped. No apply quota was used.",
                ea_summary.found,
                ea_summary.shared,
                ea_summary.already_shared,
                ea_summary.failed,
                ea_summary.skipped,
            )
            if ea_summary.aborted_reason:
                logger.warning(
                    "Early access pass stopped early: %s",
                    ea_summary.aborted_reason,
                )
            if early_access_only:
                summary.quota_after = summary.quota_before
                return summary

        keywords = derive_keywords(profile, settings)
        logger.info("Searching with keywords: %s", ", ".join(keywords) or "(none)")

        candidates = sources.collect_candidates(
            driver, settings, keywords, settings.location
        )  # type: List[JobPosting]
        summary.collected = len(candidates)
        logger.info("Collected %d unique candidate postings.", summary.collected)
        if not candidates:
            logger.warning(
                "No postings collected. Naukri's markup may have changed again, or "
                "the search returned nothing. Nothing was applied to."
            )
            summary.quota_after = ledger.quota_used()
            return summary

        for job in candidates:
            ledger.record_seen(job)

        ranked = ranking.rank(profile, candidates, client, settings)
        summary.ranked = len(ranked)

        selected = ranking.select_for_application(
            ranked, settings, ledger.has_applied
        )
        if not selected:
            logger.warning(
                "Nothing cleared the bar (min score %.0f, max age %d days). "
                "No quota spent.",
                settings.min_score,
                settings.max_days_old,
            )
            summary.quota_after = ledger.quota_used()
            return summary

        _report_plan(selected, ranking)

        if settings.dry_run:
            logger.info("Dry run: stopping before the first click.")
            summary.quota_after = summary.quota_before
            return summary

        answer_engine = answers_module.AnswerEngine(
            ledger, client, profile, settings
        )

        for position, scored in enumerate(selected, start=1):
            if ledger.quota_remaining(settings.daily_quota) <= 0:
                logger.warning("Quota exhausted mid-run. Stopping cleanly.")
                break

            logger.info(
                "[%d/%d] %s @ %s",
                position,
                len(selected),
                scored.job.title,
                scored.job.company,
            )
            result = apply_module.apply_to_job(
                driver, scored.job, profile, answer_engine, settings
            )
            ledger.record_attempt(result)
            _tally(summary, result)
            logger.info("      -> %s (%s)", result.status, result.detail or "-")

            if result.status in ApplyStatus.TERMINAL:
                logger.warning(
                    "Naukri reports the daily quota is exhausted. Stopping."
                )
                break

        summary.quota_after = ledger.quota_used()
        return summary

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception as exc:  # driver teardown must never mask a real error
                logger.debug("Ignoring driver teardown error: %s", exc)
        ledger.close()


class _NeedsHuman(Exception):
    """Login hit a captcha or OTP -- no amount of retrying will fix it."""


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    from . import config
    from . import profile as profile_module

    try:
        settings = config.load_settings(args.env, build_overrides(args))
    except Exception as exc:
        logger.error("Could not load settings: %s", exc)
        return EXIT_CONFIG

    problems = settings.validate()
    if problems:
        for problem in problems:
            logger.error("Config: %s", problem)
        logger.error("Fix the above in your .env file (see .env.example).")
        return EXIT_CONFIG

    try:
        profile = profile_module.load_profile(settings)
    except profile_module.ProfileError as exc:
        logger.error("Could not load your profile: %s", exc)
        logger.error(
            "Copy profile.example.yaml to %s and fill it in.", settings.profile_path
        )
        return EXIT_CONFIG

    logger.info(
        "Profile: %s | %s | %s",
        profile.full_name or "<unnamed>",
        profile.current_location or "<no location>",
        "%.1f yrs experience" % profile.total_experience_years
        if profile.total_experience_years is not None
        else "experience not stated",
    )

    try:
        summary = run(
            settings,
            profile,
            early_access=args.early_access or args.early_access_only,
            early_access_limit=args.early_access_limit,
            early_access_only=args.early_access_only,
        )
    except _NeedsHuman as exc:
        logger.error("Login needs a human: %s", exc)
        logger.error(
            "Run again without --headless, sign in manually when the window "
            "opens, and the session will be reused."
        )
        return EXIT_LOGIN
    except KeyboardInterrupt:
        logger.warning("Interrupted. Applications already made are recorded.")
        return EXIT_FAILED
    except Exception as exc:
        logger.exception("Run failed: %s", exc)
        return EXIT_FAILED

    summary.finished_at = datetime.now().isoformat(timespec="seconds")
    _log_summary(summary, settings)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
