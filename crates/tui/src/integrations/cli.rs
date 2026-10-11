//! `deepseek integrations dsh …` — the command surface for the harness
//! connection.
//!
//! Every subcommand is side-effect free except `connect`, `update`, `disable`,
//! `enable`, `remove`, and `launch`; `status`, `plan`, and `overlay` never
//! write. Writes only ever land under `~/.deepseek/integrations/dsh/`.

use std::io::IsTerminal;
use std::path::Path;

use anyhow::{Context, Result};

use super::dsh::{
    self, CLI_COMMAND, DetectEnv, DshPaths, DshPlan, DshReceiptEvent, DshRunner, ProcessRunner,
    RouteResolution, SHIPPED_PROFILES, VERIFIED_DSH_VERSION, compute_status, plan, spawn_launch,
    validate_profile,
};
use crate::config::Config;

/// What the caller asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DshCommand {
    Status {
        json: bool,
    },
    Plan {
        profile: String,
        allow_full_access: bool,
        json: bool,
    },
    Connect {
        profile: String,
        allow_full_access: bool,
        yes: bool,
    },
    Update {
        profile: Option<String>,
        allow_full_access: bool,
        yes: bool,
    },
    Launch {
        profile: Option<String>,
        dry_run: bool,
        app_args: Vec<String>,
    },
    Disable,
    Enable,
    Remove {
        yes: bool,
    },
    /// Print the overlay that would be written, without writing it.
    Overlay {
        profile: String,
        allow_full_access: bool,
    },
}

impl DshCommand {
    /// The profile this command targets, when it names one.
    fn named_profile(&self) -> Option<String> {
        match self {
            Self::Plan { profile, .. }
            | Self::Connect { profile, .. }
            | Self::Overlay { profile, .. } => Some(profile.clone()),
            Self::Update { profile, .. } | Self::Launch { profile, .. } => profile.clone(),
            _ => None,
        }
    }
}

/// Run one integration command against the current configuration.
pub(crate) fn run(config: &Config, workspace: &Path, command: DshCommand) -> Result<()> {
    let paths = DshPaths::from_process()?;
    let env = DetectEnv::from_process();
    run_with(config, workspace, command, &paths, &env, &ProcessRunner)
}

/// Run one integration command with injected paths, environment, and runner so
/// tests never touch the real home or run the real launcher.
pub(crate) fn run_with(
    config: &Config,
    workspace: &Path,
    command: DshCommand,
    paths: &DshPaths,
    env: &DetectEnv,
    runner: &dyn DshRunner,
) -> Result<()> {
    let route = route_resolution(config, workspace);
    let requested_profile = command
        .named_profile()
        .unwrap_or_else(|| "web".to_string());
    let report = compute_status(paths, env, runner, &route, &requested_profile)?;

    match command {
        DshCommand::Status { json } => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "state": report.state,
                        "detection": report.detection,
                        "connection": report.record,
                        "overlay_present": report.overlay_present,
                        "overlay_sha256": report.overlay_sha256_on_disk,
                        "shadowing_rows": report.shadowing_rows,
                        "paths": {
                            "root": report.paths_root,
                            "overlay": report.overlay_path,
                            "receipt": report.receipt_path,
                        },
                        "verified_dsh_version": VERIFIED_DSH_VERSION,
                    }))?
                );
            } else {
                println!("{}", report.status_line());
                for disclosure in report.disclosures() {
                    println!("  - {disclosure}");
                }
            }
            Ok(())
        }
        DshCommand::Plan {
            profile,
            allow_full_access,
            json,
        } => {
            let plan = build_plan(paths, &report, &route, &profile, allow_full_access, None)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "profile": plan.profile,
                        "overlay_path": plan.overlay_path,
                        "receipt_path": plan.receipt_path,
                        "overlay_sha256": plan.overlay_sha256,
                        "overlay": plan.overlay_text,
                        "identity": plan.identity,
                        "disclosures": plan.disclosures,
                    }))?
                );
            } else {
                print!("{}", plan.display());
            }
            Ok(())
        }
        DshCommand::Overlay {
            profile,
            allow_full_access,
        } => {
            let plan = build_plan(paths, &report, &route, &profile, allow_full_access, None)?;
            print!("{}", plan.overlay_text);
            Ok(())
        }
        DshCommand::Connect { .. }
        | DshCommand::Update { .. }
        | DshCommand::Launch { .. }
        | DshCommand::Disable
        | DshCommand::Enable
        | DshCommand::Remove { .. } => run_write_command(paths, &report, &route, workspace, command),
    }
}

fn route_resolution(config: &Config, workspace: &Path) -> RouteResolution {
    RouteResolution {
        identity: Some(dsh::route_identity(config, workspace)),
        error: None,
    }
}

fn build_plan(
    paths: &DshPaths,
    report: &dsh::DshStatusReport,
    route: &RouteResolution,
    profile: &str,
    allow_full_access: bool,
    event: Option<DshReceiptEvent>,
) -> Result<DshPlan> {
    let Some(identity) = route.identity.as_ref() else {
        anyhow::bail!(
            "cannot resolve the current route: {}",
            route.error.as_deref().unwrap_or("unknown error")
        );
    };
    plan(
        paths,
        &report.detection,
        profile,
        identity,
        allow_full_access,
        event.unwrap_or(DshReceiptEvent::Connect),
        report.disclosures(),
    )
}

fn confirm(plan: &DshPlan, yes: bool) -> Result<()> {
    if yes {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        anyhow::bail!(
            "refusing to write without confirmation; re-run with --yes (the plan above lists {} and {})",
            plan.overlay_path.display(),
            plan.receipt_path.display()
        );
    }
    confirm_text("Write this overlay and receipt?")
}

fn confirm_text(prompt: &str) -> Result<()> {
    println!("{prompt} [y/N]");
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .context("read confirmation")?;
    if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        return Ok(());
    }
    anyhow::bail!("aborted: no confirmation on stdin")
}

fn short_sha(sha: &str) -> String {
    sha.chars().take(12).collect()
}

/// Help text for `deepseek integrations dsh --help`.
pub(crate) fn help() -> String {
    format!(
        "{CLI_COMMAND} — connect an installed DeepSeek Harness (`dsh`) to this client\n\
         \n\
         usage:\n\
         \x20 {CLI_COMMAND} status [--json]\n\
         \x20 {CLI_COMMAND} plan [--profile <p>] [--allow-full-access] [--json]\n\
         \x20 {CLI_COMMAND} connect [--profile <p>] [--allow-full-access] [--yes]\n\
         \x20 {CLI_COMMAND} update [--profile <p>] [--allow-full-access] [--yes]\n\
         \x20 {CLI_COMMAND} overlay [--profile <p>] [--allow-full-access]\n\
         \x20 {CLI_COMMAND} launch [--profile <p>] [--dry-run] [-- <dsh app args>]\n\
         \x20 {CLI_COMMAND} disable | enable | remove [--yes]\n\
         \n\
         Shipped dsh profiles: {}\n\
         Verified dsh release: {VERIFIED_DSH_VERSION}\n\
         This client writes only under ~/.deepseek/integrations/dsh/ and never copies a credential.\n",
        SHIPPED_PROFILES.join(", ")
    )
}


fn run_write_command(
    paths: &DshPaths,
    report: &dsh::DshStatusReport,
    route: &RouteResolution,
    workspace: &Path,
    command: DshCommand,
) -> Result<()> {
    match command {
        DshCommand::Connect {
            profile,
            allow_full_access,
            yes,
        } => {
            let plan = build_plan(
                paths,
                report,
                route,
                &profile,
                allow_full_access,
                Some(DshReceiptEvent::Connect),
            )?;
            print!("{}", plan.display());
            confirm(&plan, yes)?;
            let record = dsh::apply_plan(paths, &plan, &report.detection)?;
            println!(
                "connected: {} · {} · {}",
                record.profile,
                record.identity.source.model,
                short_sha(&record.overlay_sha256)
            );
            Ok(())
        }
        DshCommand::Update {
            profile,
            allow_full_access,
            yes,
        } => {
            let Some(record) = report.record.as_ref() else {
                anyhow::bail!("{CLI_COMMAND} is not connected; run `connect` first");
            };
            let profile = profile.unwrap_or_else(|| record.profile.clone());
            let plan = build_plan(
                paths,
                report,
                route,
                &profile,
                allow_full_access,
                Some(DshReceiptEvent::Update),
            )?;
            print!("{}", plan.display());
            confirm(&plan, yes)?;
            let record = dsh::apply_plan(paths, &plan, &report.detection)?;
            println!(
                "updated: {} · {} · {}",
                record.profile,
                record.identity.source.model,
                short_sha(&record.overlay_sha256)
            );
            Ok(())
        }
        DshCommand::Launch {
            profile,
            dry_run,
            app_args,
        } => {
            let Some(record) = report.record.as_ref() else {
                anyhow::bail!("{CLI_COMMAND} is not connected; run `connect` first");
            };
            if record.disabled {
                anyhow::bail!("{CLI_COMMAND} is disabled; run `enable` first");
            }
            if !report.state.launchable() {
                anyhow::bail!("refusing to launch: {}", report.status_line());
            }
            if let Some(profile) = profile.as_deref() {
                validate_profile(profile)?;
                if profile != record.profile {
                    anyhow::bail!(
                        "`{profile}` has no overlay; run `connect --profile {profile}` first"
                    );
                }
            }
            let Some(binary) = report.detection.binary.as_ref() else {
                anyhow::bail!("dsh is not on PATH");
            };
            let spec = dsh::launch_spec(
                record,
                binary,
                workspace,
                &app_args,
                std::env::var_os(dsh::API_KEY_SOURCE_MARKER).is_some(),
            );
            if dry_run {
                println!("{}", spec.display());
                return Ok(());
            }
            crate::audit::log_sensitive_event_in(
                &paths.audit_log(),
                "dsh_launch",
                serde_json::json!({
                    "profile": record.profile,
                    "provider": record.identity.source.provider_id,
                    "model": record.identity.source.model,
                    "permission_mode": record.identity.permission_mode.as_str(),
                }),
            );
            let code = spawn_launch(&spec)?;
            if code != 0 {
                anyhow::bail!("dsh exited with status {code}");
            }
            Ok(())
        }
        DshCommand::Disable => {
            let record = dsh::set_disabled(paths, true)?;
            println!(
                "disabled: launches are refused; the overlay stays at {}",
                record.overlay_path.display()
            );
            Ok(())
        }
        DshCommand::Enable => {
            let record = dsh::set_disabled(paths, false)?;
            println!("enabled: {}", record.overlay_path.display());
            Ok(())
        }
        DshCommand::Remove { yes } => {
            if !yes {
                confirm_text("Remove the dsh overlay and receipt? The harness itself is untouched.")?;
            }
            let removed = dsh::remove(paths)?;
            if removed.is_empty() {
                println!("nothing to remove");
            } else {
                for path in removed {
                    println!("removed {}", path.display());
                }
            }
            Ok(())
        }
        DshCommand::Status { .. } | DshCommand::Plan { .. } | DshCommand::Overlay { .. } => {
            unreachable!("read-only commands are handled before run_write_command")
        }
    }
}
