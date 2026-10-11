//! DeepSeek Harness (`dsh`) connected through DeepSeek TUI.
//!
//! A thin, reversible adapter around DSH's *documented* seams:
//!
//! - detection reads `dsh --version` / `--help` and inventories `$DSH_HOME`
//!   (profile names, the live profile patch's patched rows, and whether
//!   `.credentials.yaml` exists) without writing anything;
//! - connection writes only under `~/.deepseek/integrations/dsh/`: a `--patch`
//!   overlay pinning the exact route identity and an append-only receipt;
//! - launch runs `dsh --profile <web|headless|…> --patch <overlay>` with the
//!   permission posture exported as `DSH_PERMISSION_MODE`, keeping the user's
//!   own `$DSH_HOME` (credentials, sessions, profiles) untouched;
//! - removal deletes only DeepSeek-TUI-owned files.
//!
//! This client never copies, prints, or embeds API keys, OAuth documents, or
//! filesystem contents; never silently switches the model; never broadens
//! permissions. DSH stays an integrated harness surface, not a second
//! scheduler.

pub(crate) mod detect;
pub(crate) mod identity;
pub(crate) mod receipt;

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub(crate) use detect::{
    DetectEnv, DshCompatibility, DshDetection, DshRunner, ProcessRunner, VERIFIED_DSH_VERSION,
};
pub(crate) use identity::{
    DshAdapter, DshFlavor, MappedIdentity, RouteIdentity, map_identity,
    render_overlay, sha256_hex,
};
pub(crate) use receipt::{
    DshConnectionRecord, DshReceiptDocument, DshReceiptEntry, DshReceiptEvent, now_rfc3339,
    write_atomic,
};

/// Directory (relative to the app home) this integration owns.
pub(crate) const INTEGRATION_DIR: &str = "integrations/dsh";
pub(crate) const OVERLAY_FILE: &str = "deepseek.patch.yml";
pub(crate) const RECEIPT_FILE: &str = "receipt.json";
pub(crate) const CLI_COMMAND: &str = "deepseek integrations dsh";

/// DSH profiles this integration refuses to launch, with the reason.
///
/// `tui` was never a shipped DSH template (`@deepseek-ai/dsh-tui` does not
/// exist), so `--profile tui` boots a bare base profile; `desktop` is only
/// accepted when the Desktop app itself launched it.
pub(crate) const REFUSED_PROFILES: &[(&str, &str)] = &[
    (
        "tui",
        "`tui` is not a shipped dsh profile; `web` and `headless` are",
    ),
    (
        "desktop",
        "the `desktop` profile is reserved for the dsh Desktop launcher",
    ),
];

/// Shipped DSH profiles, used for error messages and the `--profile` default.
pub(crate) const SHIPPED_PROFILES: &[&str] = &["web", "headless", "acp", "sdk", "sdk-minimal"];

/// Files this integration owns under the app home.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DshPaths {
    /// The app home these paths live under; audit events land in its
    /// `audit.log`.
    pub(crate) app_home: PathBuf,
    pub(crate) root: PathBuf,
    pub(crate) overlay: PathBuf,
    pub(crate) receipt: PathBuf,
}

impl DshPaths {
    pub(crate) fn under(app_home: &Path) -> Self {
        let root = app_home.join(INTEGRATION_DIR);
        Self {
            overlay: root.join(OVERLAY_FILE),
            receipt: root.join(RECEIPT_FILE),
            root,
            app_home: app_home.to_path_buf(),
        }
    }

    /// Resolve the app home the way the rest of this client does.
    pub(crate) fn from_process() -> Result<Self> {
        let home = crate::config::app_home_dir()
            .context("could not resolve the app home directory for the dsh integration")?;
        Ok(Self::under(&home))
    }

    /// Path of the audit log this integration appends to.
    pub(crate) fn audit_log(&self) -> PathBuf {
        self.app_home.join("audit.log")
    }
}

/// Honest integration state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "kebab-case")]
pub(crate) enum DshIntegrationState {
    /// No `dsh` on `PATH`.
    NotInstalled,
    /// `dsh` exists but could not report a version.
    Offline { reason: String },
    /// `dsh` is older than verified or lacks `--patch`.
    Incompatible {
        version: Option<String>,
        reason: String,
    },
    /// Installed and usable, but no overlay exists yet.
    Detected { version: String },
    /// Overlay present and matches the current route.
    Connected { version: String },
    /// Overlay present but the route or the file drifted.
    StaleConfig { version: String, reason: String },
    /// Connected, but `dsh` is newer than the verified release.
    StaleVersion { version: String, verified: String },
    /// Overlay kept on disk but launches are refused.
    Disabled { version: Option<String> },
}

impl DshIntegrationState {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::NotInstalled => "not-installed",
            Self::Offline { .. } => "offline",
            Self::Incompatible { .. } => "incompatible",
            Self::Detected { .. } => "detected",
            Self::Connected { .. } => "connected",
            Self::StaleConfig { .. } => "stale-config",
            Self::StaleVersion { .. } => "stale-version",
            Self::Disabled { .. } => "disabled",
        }
    }

    /// Only a connected (or newer-but-unverified) install may launch.
    pub(crate) fn launchable(&self) -> bool {
        matches!(self, Self::Connected { .. } | Self::StaleVersion { .. })
    }
}


/// The route this client would pin right now, or why it could not be resolved.
#[derive(Debug, Clone, Default)]
pub(crate) struct RouteResolution {
    pub(crate) identity: Option<RouteIdentity>,
    pub(crate) error: Option<String>,
}

/// Rows our overlay targets, per flavor.
pub(crate) const NATIVE_ROWS: &[&str] = &["agent-default-model", "llm-deepseek"];
pub(crate) const PI_AI_ROWS: &[&str] = &["agent-default-model", "llm-pi-ai"];

/// Everything status/plan/doctor need, computed without side effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DshStatusReport {
    pub(crate) state: DshIntegrationState,
    pub(crate) detection: DshDetection,
    pub(crate) record: Option<DshConnectionRecord>,
    pub(crate) overlay_present: bool,
    pub(crate) overlay_sha256_on_disk: Option<String>,
    /// The identity this client would write *now*.
    pub(crate) current_identity: Option<MappedIdentity>,
    pub(crate) current_identity_error: Option<String>,
    /// Rows the user's own live harness config already patches and this
    /// integration also patches — reported because the later layer wins.
    pub(crate) shadowing_rows: Vec<String>,
    pub(crate) paths_root: PathBuf,
    pub(crate) overlay_path: PathBuf,
    pub(crate) receipt_path: PathBuf,
}

impl DshStatusReport {
    /// One-line summary for humans.
    pub(crate) fn status_line(&self) -> String {
        let launcher = self.detection.summary();
        let mut line = format!(
            "dsh {} · {} · {}",
            self.state.label(),
            launcher,
            self.detection.compatibility.label()
        );
        if let Some(identity) = self.current_identity.as_ref() {
            line.push_str(&format!(
                " · {}/{}",
                identity.source.provider_id, identity.source.model
            ));
            line.push_str(&format!(" · {}", identity.permission_mode.as_str()));
        }
        if let Some(error) = self.current_identity_error.as_deref() {
            line.push_str(&format!(" · route unresolved: {error}"));
        }
        line
    }

    /// What the user should know before connecting or launching.
    pub(crate) fn disclosures(&self) -> Vec<String> {
        let mut out = Vec::new();
        match &self.state {
            DshIntegrationState::NotInstalled => out.push(
                "dsh is not on PATH; install DeepSeek Harness (`npx @deepseek-ai/dsh web`) first."
                    .to_string(),
            ),
            DshIntegrationState::Offline { reason } => {
                out.push(format!("dsh could not be inspected: {reason}"));
            }
            DshIntegrationState::Incompatible { reason, .. } => {
                out.push(format!("dsh is incompatible with this integration: {reason}"));
            }
            DshIntegrationState::Detected { .. } => {
                out.push("dsh is installed; connect to write the overlay.".to_string());
            }
            DshIntegrationState::StaleConfig { reason, .. } => {
                out.push(format!("The overlay is stale: {reason}"));
            }
            DshIntegrationState::StaleVersion { version, verified } => out.push(format!(
                "dsh {version} is newer than the verified {verified}; launches are allowed but unverified."
            )),
            DshIntegrationState::Disabled { .. } => {
                out.push("The overlay is disabled; enable it to launch again.".to_string());
            }
            DshIntegrationState::Connected { .. } => {}
        }
        if !self.shadowing_rows.is_empty() {
            out.push(format!(
                "Your own dsh configuration already patches {}; the --patch overlay is layered above it, so it wins for launches started here.",
                self.shadowing_rows.join(", ")
            ));
        }
        if self.detection.dsh_home_exists && !self.detection.profiles.is_empty() {
            out.push(format!(
                "dsh profiles on this machine: {}",
                self.detection.profiles.join(", ")
            ));
        }
        if let Some(identity) = self.current_identity.as_ref() {
            out.extend(identity.disclosures.iter().cloned());
        }
        out
    }
}

/// Rows the harness's own live config already patches.
///
/// `0.2.x` keeps live configuration in the profile patch
/// (`$DSH_HOME/profiles/<profile>/cordis.patch.yml`) plus the home-level
/// `$DSH_HOME/cordis.patch.yml`; `$DSH_HOME/settings.yaml` is imported once and
/// renamed, so it is only read as the legacy `0.1.x` source.
pub(crate) fn shadowing_rows(
    detection: &DshDetection,
    profile: &str,
    flavor: DshFlavor,
) -> Vec<String> {
    let wanted: &[&str] = match flavor {
        DshFlavor::Legacy => NATIVE_ROWS,
        DshFlavor::Modern => PI_AI_ROWS,
    };
    let mut found: Vec<String> = Vec::new();
    let mut note = |candidate: &str| {
        if wanted.contains(&candidate) && !found.iter().any(|row| row == candidate) {
            found.push(candidate.to_string());
        }
    };
    match flavor {
        DshFlavor::Legacy => {
            for namespace in detection.settings_namespaces.iter() {
                note(namespace);
            }
        }
        DshFlavor::Modern => {
            let sources = [
                detection
                    .dsh_home
                    .join("profiles")
                    .join(profile)
                    .join("cordis.patch.yml"),
                detection.dsh_home.join("cordis.patch.yml"),
            ];
            for source in sources {
                if let Ok(bytes) = std::fs::read(&source) {
                    let text = String::from_utf8_lossy(&bytes[..bytes.len().min(64 * 1024)]);
                    for id in detect::patch_row_ids(&text) {
                        note(&id);
                    }
                }
            }
        }
    }
    found
}

/// Why the on-disk overlay no longer matches the connection record, if it
/// does not.
fn client_drift(
    record: &DshConnectionRecord,
    overlay_present: bool,
    overlay_sha256_on_disk: Option<&str>,
    current: Option<&MappedIdentity>,
) -> Option<String> {
    if !overlay_present {
        return Some("overlay file is missing; run `update`".to_string());
    }
    if overlay_sha256_on_disk != Some(record.overlay_sha256.as_str()) {
        return Some("overlay file was modified outside this client; run `update`".to_string());
    }
    match current {
        Some(now) if !now.mappable() => Some(
            "the current route cannot be carried by dsh; the overlay is stale".to_string(),
        ),
        Some(now) => {
            let expected = render_overlay(now).map(|text| sha256_hex(text.as_bytes()));
            match expected {
                Some(expected) if expected == record.overlay_sha256 => None,
                Some(_) => Some(format!(
                    "the route is now {}/{}; the overlay pins {}/{}; run `update`",
                    now.source.provider_id,
                    now.source.model,
                    record.identity.source.provider_id,
                    record.identity.source.model
                )),
                None => Some(
                    "the current route cannot be carried by dsh; the overlay is stale".to_string(),
                ),
            }
        }
        None => None,
    }
}

/// Compute the integration state without writing anything.
pub(crate) fn compute_status(
    paths: &DshPaths,
    env: &DetectEnv,
    runner: &dyn DshRunner,
    route: &RouteResolution,
    profile: &str,
) -> Result<DshStatusReport> {
    let detection = detect::detect(env, runner);
    let document = DshReceiptDocument::load(&paths.receipt)?;
    let record = document.current.clone();
    let overlay_bytes = std::fs::read(&paths.overlay).ok();
    let overlay_present = overlay_bytes.is_some();
    let overlay_sha256_on_disk = overlay_bytes.as_deref().map(sha256_hex);

    let flavor = DshFlavor::of(detection.modern_native_deepseek());
    let current_identity = route
        .identity
        .as_ref()
        .map(|identity| map_identity(identity, flavor, false));
    let shadowing = shadowing_rows(&detection, profile, flavor);

    let state = if !detection.installed() {
        DshIntegrationState::NotInstalled
    } else {
        match &detection.compatibility {
            DshCompatibility::Offline { reason } => DshIntegrationState::Offline {
                reason: reason.clone(),
            },
            DshCompatibility::Unparsed { raw } => DshIntegrationState::Offline {
                reason: format!("`dsh --version` printed `{raw}`, which is not a version"),
            },
            DshCompatibility::Incompatible { reason } => DshIntegrationState::Incompatible {
                version: detection.version.clone(),
                reason: reason.clone(),
            },
            DshCompatibility::Verified | DshCompatibility::NewerUnverified { .. } => {
                let version = detection.version.clone().unwrap_or_default();
                match record.as_ref() {
                    None => DshIntegrationState::Detected { version },
                    Some(record) if record.disabled => DshIntegrationState::Disabled {
                        version: Some(version),
                    },
                    Some(record) => match client_drift(
                        record,
                        overlay_present,
                        overlay_sha256_on_disk.as_deref(),
                        current_identity.as_ref(),
                    ) {
                        Some(reason) => DshIntegrationState::StaleConfig { version, reason },
                        None => match &detection.compatibility {
                            DshCompatibility::NewerUnverified { verified } => {
                                DshIntegrationState::StaleVersion {
                                    version,
                                    verified: verified.clone(),
                                }
                            }
                            _ => DshIntegrationState::Connected { version },
                        },
                    },
                }
            }
        }
    };

    Ok(DshStatusReport {
        state,
        detection,
        record,
        overlay_present,
        overlay_sha256_on_disk,
        current_identity,
        current_identity_error: route.error.clone(),
        shadowing_rows: shadowing,
        paths_root: paths.root.clone(),
        overlay_path: paths.overlay.clone(),
        receipt_path: paths.receipt.clone(),
    })
}

/// What `connect`/`update` will write, spelled out before any write happens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DshPlan {
    pub(crate) profile: String,
    pub(crate) overlay_text: String,
    pub(crate) overlay_sha256: String,
    pub(crate) identity: MappedIdentity,
    pub(crate) dsh_version: Option<String>,
    pub(crate) dsh_home: PathBuf,
    pub(crate) overlay_path: PathBuf,
    pub(crate) receipt_path: PathBuf,
    pub(crate) event: DshReceiptEvent,
    pub(crate) disclosures: Vec<String>,
}

impl DshPlan {
    pub(crate) fn permission_mode(&self) -> &'static str {
        self.identity.permission_mode.as_str()
    }

    /// The exact plan, as it is printed before confirmation.
    pub(crate) fn display(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{} — {} {}\n",
            if self.event == DshReceiptEvent::Update {
                "Update the DeepSeek Harness connection"
            } else {
                "Connect DeepSeek Harness"
            },
            self.profile,
            self.identity.source.provider_id
        ));
        out.push_str(&format!(
            "  route      {}/{} via {}\n",
            self.identity.source.provider_id,
            self.identity.source.model,
            self.identity.adapter_label()
        ));
        out.push_str(&format!("  endpoint   {}\n", self.identity.source.base_url));
        out.push_str(&format!(
            "  permission {}\n",
            self.identity.permission_mode.as_str()
        ));
        out.push_str(&format!(
            "  dsh        {} ({})\n",
            self.dsh_version.as_deref().unwrap_or("unknown"),
            self.dsh_home.display()
        ));
        out.push_str(&format!("  writes     {}\n", self.overlay_path.display()));
        out.push_str(&format!("             {}\n", self.receipt_path.display()));
        out.push_str(&format!("  sha256     {}\n", self.overlay_sha256));
        if !self.disclosures.is_empty() {
            out.push_str("  disclosures:\n");
            for disclosure in &self.disclosures {
                out.push_str(&format!("    - {disclosure}\n"));
            }
        }
        out.push_str("  overlay:\n");
        for line in self.overlay_text.lines() {
            out.push_str(&format!("    {line}\n"));
        }
        out
    }
}


pub(crate) fn set_disabled(paths: &DshPaths, disabled: bool) -> Result<DshConnectionRecord> {
    let mut document = DshReceiptDocument::load(&paths.receipt)?;
    let Some(record) = document.current.clone() else {
        anyhow::bail!(
            "no dsh connection to {}; run `connect` first",
            if disabled { "disable" } else { "enable" }
        );
    };
    let now = now_rfc3339();
    let event = if disabled {
        DshReceiptEvent::Disable
    } else {
        DshReceiptEvent::Enable
    };
    let mut updated = record.clone();
    updated.disabled = disabled;
    updated.updated_at = now.clone();
    document.push(DshReceiptEntry {
        event: event.clone(),
        at: now,
        deepseek_version: env!("CARGO_PKG_VERSION").to_string(),
        dsh_version: record.dsh_version.clone(),
        dsh_home: record.dsh_home.clone(),
        overlay_sha256: Some(record.overlay_sha256.clone()),
        identity_summary: Some(identity_summary(&record.identity)),
        permission_mode: Some(record.identity.permission_mode.as_str().to_string()),
        note: None,
    });
    document.current = Some(updated.clone());
    document.save(&paths.receipt)?;
    crate::audit::log_sensitive_event_in(
        &paths.audit_log(),
        &format!("dsh_{}", event.as_str()),
        serde_json::json!({
            "profile": record.profile,
            "provider": record.identity.source.provider_id,
            "model": record.identity.source.model,
        }),
    );
    Ok(updated)
}

/// Delete only this integration's own files, and prune the directory when it
/// becomes empty.
pub(crate) fn remove(paths: &DshPaths) -> Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    let record = DshReceiptDocument::load(&paths.receipt)?.current;
    for path in [paths.overlay.clone(), paths.receipt.clone()] {
        match std::fs::remove_file(&path) {
            Ok(()) => removed.push(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("delete {}", path.display()));
            }
        }
    }
    if let Some(record) = record.as_ref() {
        // The receipt file is gone; the audit log keeps the record of removal.
        crate::audit::log_sensitive_event_in(
            &paths.audit_log(),
            "dsh_remove",
            serde_json::json!({
                "profile": record.profile,
                "provider": record.identity.source.provider_id,
                "model": record.identity.source.model,
            }),
        );
    }
    if let Ok(entries) = std::fs::read_dir(&paths.root)
        && entries.flatten().next().is_none()
    {
        let _ = std::fs::remove_dir(&paths.root);
    }
    Ok(removed)
}

/// Validate a `--profile` value against the profiles `dsh` actually ships.
pub(crate) fn validate_profile(profile: &str) -> Result<String> {
    let trimmed = profile.trim();
    if trimmed.is_empty() {
        anyhow::bail!("--profile needs a name (shipped profiles: {})", SHIPPED_PROFILES.join(", "));
    }
    if let Some((_, reason)) = REFUSED_PROFILES.iter().find(|(name, _)| *name == trimmed) {
        anyhow::bail!("`{trimmed}` is not launchable through this integration: {reason}");
    }
    Ok(trimmed.to_string())
}

/// The exact files and identity `connect`/`update` would write.
pub(crate) fn plan(
    paths: &DshPaths,
    detection: &DshDetection,
    profile: &str,
    route: &RouteIdentity,
    allow_full_access: bool,
    event: DshReceiptEvent,
    extra_disclosures: Vec<String>,
) -> Result<DshPlan> {
    let profile = validate_profile(profile)?;
    if let detect::DshCompatibility::Incompatible { reason } = &detection.compatibility {
        anyhow::bail!(
            "cannot connect dsh: the installed launcher is incompatible ({reason}); upgrade dsh (`npx @deepseek-ai/dsh --version`) and retry"
        );
    }
    let flavor = DshFlavor::of(detection.modern_native_deepseek());
    let identity = map_identity(route, flavor, allow_full_access);
    let Some(overlay_text) = render_overlay(&identity) else {
        let reason = match &identity.adapter {
            DshAdapter::Unsupported { reason } => reason.clone(),
            _ => "the route cannot be carried by dsh".to_string(),
        };
        anyhow::bail!("cannot connect dsh: {reason}");
    };
    let overlay_sha256 = sha256_hex(overlay_text.as_bytes());
    let mut disclosures = identity.disclosures.clone();
    disclosures.extend(extra_disclosures);
    if matches!(identity.adapter, DshAdapter::PiAiOpenAiCompatible { .. }) {
        disclosures.push(identity::dialect_disclosure());
    }
    if flavor == DshFlavor::Modern {
        disclosures.push(
            "The overlay replaces the `llm-pi-ai` row's `providers` table (dsh assigns a patch `config` wholesale, it does not deep-merge); launches started here therefore show only this route. Your own dsh config is not edited."
                .to_string(),
        );
    }
    Ok(DshPlan {
        profile,
        overlay_text,
        overlay_sha256,
        identity,
        dsh_version: detection.version.clone(),
        dsh_home: detection.dsh_home.clone(),
        overlay_path: paths.overlay.clone(),
        receipt_path: paths.receipt.clone(),
        event,
        disclosures,
    })
}

fn identity_summary(identity: &MappedIdentity) -> String {
    format!(
        "{}/{}",
        identity.source.provider_id, identity.source.model
    )
}

/// Write the overlay and append the receipt entry. The only writes this
/// integration ever performs are inside its own directory.
pub(crate) fn apply_plan(
    paths: &DshPaths,
    plan: &DshPlan,
    detection: &DshDetection,
) -> Result<DshConnectionRecord> {
    std::fs::create_dir_all(&paths.root)
        .with_context(|| format!("create {}", paths.root.display()))?;
    write_atomic(&paths.overlay, plan.overlay_text.as_bytes())?;

    let mut document = DshReceiptDocument::load(&paths.receipt)?;
    let now = now_rfc3339();
    let connected_at = document
        .current
        .as_ref()
        .map(|record| record.connected_at.clone())
        .unwrap_or_else(|| now.clone());
    let record = DshConnectionRecord {
        connected_at,
        updated_at: now.clone(),
        dsh_version: plan.dsh_version.clone(),
        dsh_binary: detection.binary.clone(),
        dsh_home: plan.dsh_home.clone(),
        profile: plan.profile.clone(),
        overlay_path: paths.overlay.clone(),
        overlay_sha256: plan.overlay_sha256.clone(),
        disabled: false,
        identity: plan.identity.clone(),
    };
    document.push(DshReceiptEntry {
        event: plan.event.clone(),
        at: now,
        deepseek_version: env!("CARGO_PKG_VERSION").to_string(),
        dsh_version: plan.dsh_version.clone(),
        dsh_home: plan.dsh_home.clone(),
        overlay_sha256: Some(plan.overlay_sha256.clone()),
        identity_summary: Some(identity_summary(&plan.identity)),
        permission_mode: Some(plan.permission_mode().to_string()),
        note: None,
    });
    document.current = Some(record.clone());
    document.save(&paths.receipt)?;

    crate::audit::log_sensitive_event_in(
        &paths.audit_log(),
        &format!("dsh_{}", plan.event.as_str()),
        serde_json::json!({
            "profile": plan.profile,
            "provider": plan.identity.source.provider_id,
            "model": plan.identity.source.model,
            "permission_mode": plan.permission_mode(),
            "overlay_sha256": plan.overlay_sha256,
            "dsh_version": plan.dsh_version,
        }),
    );
    Ok(record)
}


/// Environment variables this client owns. They mean nothing to `dsh`, and
/// handing them over would let this client's own configuration (or the API key
/// it materialized) leak into the harness process.
pub(crate) const CLIENT_ENV_REMOVE: &[&str] = &[
    "DEEPSEEK_API_KEY_SOURCE",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_PROVIDER",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_OUTPUT_MODE",
    "DEEPSEEK_LOG_LEVEL",
    "DEEPSEEK_APPROVAL_POLICY",
    "DEEPSEEK_SANDBOX_MODE",
    "DEEPSEEK_TELEMETRY",
];

/// Marker this client sets when it injected a provider key into its own
/// process environment.
pub(crate) const API_KEY_SOURCE_MARKER: &str = "DEEPSEEK_API_KEY_SOURCE";

/// A fully resolved launch of the installed launcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LaunchSpec {
    pub(crate) program: PathBuf,
    pub(crate) args: Vec<String>,
    pub(crate) cwd: PathBuf,
    pub(crate) env_set: Vec<(String, String)>,
    pub(crate) env_remove: Vec<String>,
}

impl LaunchSpec {
    pub(crate) fn display(&self) -> String {
        let mut out = String::new();
        for (key, value) in &self.env_set {
            out.push_str(&format!("{key}={value} "));
        }
        out.push_str(&self.program.display().to_string());
        for arg in &self.args {
            out.push(' ');
            out.push_str(arg);
        }
        out
    }
}

/// Credential variables this client may have materialized for itself and must
/// strip from the harness child when it did.
pub(crate) fn launch_env_remove(api_key_marker_present: bool, api_key_env: Option<&str>) -> Vec<String> {
    let mut remove: Vec<String> = CLIENT_ENV_REMOVE.iter().map(|name| name.to_string()).collect();
    if api_key_marker_present
        && let Some(name) = api_key_env
    {
        remove.push(name.to_string());
    }
    remove
}

/// Build the launch for a connected installation.
pub(crate) fn launch_spec(
    record: &DshConnectionRecord,
    binary: &Path,
    workspace: &Path,
    app_args: &[String],
    api_key_marker_present: bool,
) -> LaunchSpec {
    let mut args = vec![
        "--profile".to_string(),
        record.profile.clone(),
        "--patch".to_string(),
        record.overlay_path.display().to_string(),
    ];
    args.extend(app_args.iter().cloned());
    LaunchSpec {
        program: binary.to_path_buf(),
        args,
        cwd: workspace.to_path_buf(),
        env_set: vec![
            (
                "DSH_PERMISSION_MODE".to_string(),
                record.identity.permission_mode.as_str().to_string(),
            ),
            // The harness honors any non-empty value as a hard opt-out.
            ("DSH_TELEMETRY_DISABLED".to_string(), "1".to_string()),
        ],
        env_remove: launch_env_remove(
            api_key_marker_present,
            record.identity.source.api_key_env.as_deref(),
        ),
    }
}

/// Run the launcher with inherited stdio and return its exit code.
pub(crate) fn spawn_launch(spec: &LaunchSpec) -> Result<i32> {
    let mut command = std::process::Command::new(&spec.program);
    command.args(&spec.args).current_dir(&spec.cwd);
    for (key, value) in &spec.env_set {
        command.env(key, value);
    }
    for key in &spec.env_remove {
        command.env_remove(key);
    }
    let status = command
        .status()
        .with_context(|| format!("failed to run {}", spec.program.display()))?;
    Ok(status.code().unwrap_or(-1))
}

/// Non-secret route facts read from the resolved configuration.
pub(crate) fn route_identity(
    config: &crate::config::Config,
    workspace: &Path,
) -> RouteIdentity {
    let provider = config.api_provider();
    let base_url = config.deepseek_base_url();
    let keyless_local = matches!(
        provider,
        crate::config::ApiProvider::Sglang | crate::config::ApiProvider::Vllm
    ) && base_url_is_loopback(&base_url);
    let api_key_env = credential_env_for(provider);
    let credential_available = api_key_env
        .as_deref()
        .is_some_and(|name| std::env::var(name).is_ok_and(|value| !value.trim().is_empty()));
    RouteIdentity {
        provider_id: provider.as_str().to_string(),
        provider_label: provider.display_name().to_string(),
        model: config.default_model(),
        base_url,
        api_key_env: if keyless_local { None } else { api_key_env },
        keyless_local,
        credential_available,
        reasoning_effort: config.reasoning_effort().map(str::to_string),
        sandbox_mode: config.sandbox_mode.clone(),
        approval_policy: config.approval_policy.clone(),
        yolo: matches!(config.sandbox_mode.as_deref(), Some("danger-full-access"))
            || matches!(config.approval_policy.as_deref(), Some("never")),
        workspace: workspace.display().to_string(),
    }
}

/// The canonical credential environment variable for a provider.
pub(crate) fn credential_env_for(provider: crate::config::ApiProvider) -> Option<String> {
    use crate::config::ApiProvider;
    let name = match provider {
        ApiProvider::Deepseek | ApiProvider::DeepseekCN => "DEEPSEEK_API_KEY",
        ApiProvider::NvidiaNim => "NVIDIA_API_KEY",
        ApiProvider::Openrouter => "OPENROUTER_API_KEY",
        ApiProvider::Novita => "NOVITA_API_KEY",
        ApiProvider::Fireworks => "FIREWORKS_API_KEY",
        ApiProvider::Sglang => "SGLANG_API_KEY",
        ApiProvider::Vllm => "VLLM_API_KEY",
    };
    Some(name.to_string())
}

/// True when a base URL points at this machine.
pub(crate) fn base_url_is_loopback(url: &str) -> bool {
    let Some((_, rest)) = url.split_once("://") else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = host.split(':').next().unwrap_or(host);
    matches!(host, "localhost" | "127.0.0.1" | "::1" | "[::1]" | "0.0.0.0")
}

