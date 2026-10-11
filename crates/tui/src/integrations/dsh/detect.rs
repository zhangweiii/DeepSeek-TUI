//! Read-only detection of an installed official DeepSeek Harness (`dsh`).
//!
//! Detection never writes: it locates the `dsh` launcher on `PATH`, asks it
//! for `--version` and the launcher `--help` (neither initializes a profile),
//! resolves `$DSH_HOME` the way the harness does, and inventories the profile
//! names, `settings.yaml` namespaces, and the *presence* of the managed
//! credentials file. Credential values are never read.
//!
//! `dsh` is a developer preview: the launcher keeps `--patch <path>` as its
//! documented user-level overlay seam, and every release so far names its
//! version with a semver prerelease tag (`0.1.0-rc.6`, `0.1.7-alpha.2`,
//! `0.2.0-rc.1`, `0.2.1-alpha.2`). This module classifies the installed
//! launcher against the exact release this integration was verified on, so a
//! newer preview is reported as unverified rather than silently trusted.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

/// The exact `dsh` release this integration was verified against.
///
/// Bumping this is a deliberate act: re-read `docs/INTEGRATIONS_DSH.md` and
/// re-check the overlay row shape, the entry ids the overlay targets, and the
/// permission posture before claiming a newer release as verified.
pub(crate) const VERIFIED_DSH_VERSION: &str = "0.2.1-alpha.2";

/// First `dsh` core version that split the native DeepSeek adapter into the
/// `deepseek-official` / `deepseek-account` routes and moved the native wire
/// call onto DeepSeek's Anthropic-compatible endpoint.
pub(crate) const MODERN_DSH_CORE: (u64, u64) = (0, 2);

/// A parsed `major.minor.patch[-prerelease][+build]` version.
///
/// Deliberately dependency-free: the workspace does not depend on `semver`,
/// and the rules this module needs (prerelease precedence) are small enough to
/// own and test here rather than move the lockfile for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Version {
    major: u64,
    minor: u64,
    patch: u64,
    pre: Vec<PreId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PreId {
    Numeric(u64),
    Alpha(String),
}

impl Version {
    /// Parse the text `dsh --version` prints, tolerating a leading `v`.
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        let raw = raw.trim();
        let raw = raw.strip_prefix('v').unwrap_or(raw);
        // Build metadata is not part of precedence.
        let raw = raw.split('+').next().unwrap_or(raw);
        let (core, pre) = match raw.split_once('-') {
            Some((core, pre)) => (core, Some(pre)),
            None => (raw, None),
        };
        let mut parts = core.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        let mut pre_ids = Vec::new();
        if let Some(pre) = pre {
            if pre.is_empty() {
                return None;
            }
            for id in pre.split('.') {
                if id.is_empty() {
                    return None;
                }
                if let Ok(number) = id.parse::<u64>() {
                    // A numeric identifier must not carry a leading zero.
                    if id.len() > 1 && id.starts_with('0') {
                        return None;
                    }
                    pre_ids.push(PreId::Numeric(number));
                } else {
                    if !id.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '-') {
                        return None;
                    }
                    pre_ids.push(PreId::Alpha(id.to_string()));
                }
            }
        }
        Some(Self {
            major,
            minor,
            patch,
            pre: pre_ids,
        })
    }

    /// Semver precedence: `alpha < beta < rc < release`, numeric identifiers
    /// compare numerically, and a shorter prerelease list sorts first.
    pub(crate) fn cmp_precedence(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch)) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
        if self.pre.is_empty() && other.pre.is_empty() {
            return Ordering::Equal;
        }
        if self.pre.is_empty() {
            return Ordering::Greater;
        }
        if other.pre.is_empty() {
            return Ordering::Less;
        }
        for (left, right) in self.pre.iter().zip(other.pre.iter()) {
            let ordering = match (left, right) {
                (PreId::Numeric(a), PreId::Numeric(b)) => a.cmp(b),
                (PreId::Numeric(_), PreId::Alpha(_)) => Ordering::Less,
                (PreId::Alpha(_), PreId::Numeric(_)) => Ordering::Greater,
                (PreId::Alpha(a), PreId::Alpha(b)) => a.cmp(b),
            };
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        self.pre.len().cmp(&other.pre.len())
    }

    /// True when the core version is at or past `(major, minor)`.
    pub(crate) fn core_at_least(&self, major: u64, minor: u64) -> bool {
        (self.major, self.minor) >= (major, minor)
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre.is_empty() {
            f.write_str("-")?;
            for (index, id) in self.pre.iter().enumerate() {
                if index > 0 {
                    f.write_str(".")?;
                }
                match id {
                    PreId::Numeric(number) => write!(f, "{number}")?,
                    PreId::Alpha(text) => f.write_str(text)?,
                }
            }
        }
        Ok(())
    }
}

/// How the installed `dsh` relates to the verified release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum DshCompatibility {
    /// Exactly the verified release.
    Verified,
    /// Newer than the verified release: launchable, but unverified.
    NewerUnverified { verified: String },
    /// Older than the verified release, or missing the `--patch` seam.
    Incompatible { reason: String },
    /// The launcher exists but could not be run or did not report a version.
    Offline { reason: String },
    /// Version text that does not parse.
    Unparsed { raw: String },
}

impl DshCompatibility {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::NewerUnverified { .. } => "newer-unverified",
            Self::Incompatible { .. } => "incompatible",
            Self::Offline { .. } => "offline",
            Self::Unparsed { .. } => "unparsed",
        }
    }
}

/// Raw facts about one `dsh` installation. Every field is derived without
/// writing to disk or reading a credential value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DshDetection {
    /// Resolved launcher path when one is on `PATH`.
    pub(crate) binary: Option<PathBuf>,
    /// `dsh --version` output when it ran (raw text, trimmed).
    pub(crate) version: Option<String>,
    pub(crate) compatibility: DshCompatibility,
    /// Whether the launcher help text advertises `--patch`.
    pub(crate) supports_patch: bool,
    /// `$DSH_HOME` (or `~/.dsh`), whether or not it exists yet.
    pub(crate) dsh_home: PathBuf,
    pub(crate) dsh_home_exists: bool,
    /// True when `$DSH_HOME` came from the environment rather than the default.
    pub(crate) dsh_home_from_env: bool,
    /// Profile directory names inside `$DSH_HOME/profiles`, sorted.
    pub(crate) profiles: Vec<String>,
    /// Top-level `settings.yaml` mapping keys.
    pub(crate) settings_namespaces: Vec<String>,
    /// Whether `$DSH_HOME/.credentials.yaml` exists (never its contents).
    pub(crate) credentials_present: bool,
    /// Whether the credentials file is `0600`-equivalent, when it exists.
    pub(crate) credentials_mode_ok: Option<bool>,
}

impl DshDetection {
    pub(crate) fn installed(&self) -> bool {
        self.binary.is_some()
    }

    /// The detected launcher version, when it parsed.
    pub(crate) fn parsed_version(&self) -> Option<Version> {
        self.version.as_deref().and_then(Version::parse)
    }

    /// True when the launcher is a `0.2`-or-newer preview, whose native
    /// DeepSeek adapter speaks DeepSeek's Anthropic-compatible endpoint rather
    /// than OpenAI chat completions.
    pub(crate) fn modern_native_deepseek(&self) -> bool {
        match self.parsed_version() {
            Some(version) => version.core_at_least(MODERN_DSH_CORE.0, MODERN_DSH_CORE.1),
            // An unparseable version cannot claim the modern layout.
            None => false,
        }
    }

    /// Human summary of the launcher and where it came from.
    pub(crate) fn summary(&self) -> String {
        match (self.binary.as_ref(), self.version.as_deref()) {
            (Some(binary), Some(version)) => format!("{version} ({})", binary.display()),
            (Some(binary), None) => format!("unversioned ({})", binary.display()),
            (None, _) => "not on PATH".to_string(),
        }
    }
}

/// Environment facts detection reads. Injected so tests never depend on the
/// machine's real `PATH`, `HOME`, or `DSH_HOME`.
#[derive(Debug, Clone)]
pub(crate) struct DetectEnv {
    pub(crate) path: Option<OsString>,
    pub(crate) home: Option<PathBuf>,
    pub(crate) dsh_home: Option<OsString>,
}

impl DetectEnv {
    pub(crate) fn from_process() -> Self {
        Self {
            path: std::env::var_os("PATH"),
            home: dirs::home_dir(),
            dsh_home: std::env::var_os("DSH_HOME"),
        }
    }
}

/// Runs the launcher. Injected so tests can stub `--version`/`--help`.
pub(crate) trait DshRunner {
    /// Returns `(exit_success, stdout+stderr)`.
    fn run(&self, binary: &Path, args: &[&str]) -> std::io::Result<(bool, String)>;
}

pub(crate) struct ProcessRunner;

impl DshRunner for ProcessRunner {
    fn run(&self, binary: &Path, args: &[&str]) -> std::io::Result<(bool, String)> {
        let output = Command::new(binary)
            .args(args)
            // The launcher's own help/version paths do not send telemetry, but
            // the harness honors this as a hard opt-out.
            .env("DSH_TELEMETRY_DISABLED", "1")
            .stdin(std::process::Stdio::null())
            .output()?;
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        Ok((output.status.success(), text))
    }
}

/// Resolve `$DSH_HOME`: a non-blank env value (with `~` expansion), else
/// `~/.dsh`.
pub(crate) fn resolve_dsh_home(env: &DetectEnv) -> (PathBuf, bool) {
    if let Some(raw) = env.dsh_home.as_ref() {
        let text = raw.to_string_lossy();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if let Some(rest) = trimmed.strip_prefix("~/") {
                if let Some(home) = env.home.as_ref() {
                    return (home.join(rest), true);
                }
            } else if trimmed != "~" {
                return (PathBuf::from(trimmed), true);
            }
        }
    }
    let home = env.home.clone().unwrap_or_else(|| PathBuf::from("."));
    (home.join(".dsh"), false)
}

fn find_on_path(path: Option<&OsString>) -> Option<PathBuf> {
    let path = path?;
    for dir in std::env::split_paths(path) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        for name in ["dsh", "dsh.cmd", "dsh.exe"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Top-level YAML mapping keys of a settings document, without a YAML parser:
/// a key is a line with no leading whitespace ending in `:` (optionally with
/// an inline value). Bounded to the first 64 KiB.
pub(crate) fn settings_namespaces(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.lines().take(4096) {
        if raw.starts_with([' ', '\t', '#', '-']) || raw.trim().is_empty() {
            continue;
        }
        let Some((key, _)) = raw.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || key.starts_with('"') || key.starts_with('\'') {
            continue;
        }
        if !out.iter().any(|known| known == key) {
            out.push(key.to_string());
        }
    }
    out
}

/// Classify a version string against [`VERIFIED_DSH_VERSION`] and whether the
/// launcher advertises `--patch`.
pub(crate) fn classify_version(raw: &str, supports_patch: bool) -> DshCompatibility {
    let Some(version) = Version::parse(raw) else {
        return DshCompatibility::Unparsed {
            raw: raw.trim().to_string(),
        };
    };
    let verified =
        Version::parse(VERIFIED_DSH_VERSION).expect("the verified dsh version always parses");
    match version.cmp_precedence(&verified) {
        std::cmp::Ordering::Less => DshCompatibility::Incompatible {
            reason: format!("{version} is older than the verified {verified}"),
        },
        _ if !supports_patch => DshCompatibility::Incompatible {
            reason: "launcher does not advertise --patch overlays".to_string(),
        },
        std::cmp::Ordering::Equal => DshCompatibility::Verified,
        std::cmp::Ordering::Greater => DshCompatibility::NewerUnverified {
            verified: verified.to_string(),
        },
    }
}

/// Entry ids a `cordis.patch.yml` document targets.
///
/// No YAML parser: a patch entry that targets an existing row either writes
/// `- id: <id>` on the list-marker line or opens the item with `-` and puts
/// `id:` on the next line. Both shapes are recognized; nested `id:` keys (a
/// model catalog, for example) sit under a `- id:` of their own and are
/// therefore reported as well, which only ever makes the shadowing report
/// broader, never narrower.
pub(crate) fn patch_row_ids(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut open_item_indent: Option<usize> = None;
    for raw in text.lines().take(4096) {
        let line = raw.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = line.len() - trimmed.len();
        let mut record = |value: &str| {
            let value = value.trim().trim_matches(['\'', '"']);
            if !value.is_empty() && !out.iter().any(|known| known == value) {
                out.push(value.to_string());
            }
        };
        match trimmed.strip_prefix("- ") {
            Some(rest) => {
                open_item_indent = Some(indent);
                if let Some(value) = rest.trim_start().strip_prefix("id:") {
                    record(value);
                }
            }
            None => {
                if trimmed == "-" {
                    open_item_indent = Some(indent);
                    continue;
                }
                if let Some(base) = open_item_indent
                    && indent == base + 2
                    && let Some(value) = trimmed.strip_prefix("id:")
                {
                    record(value);
                    open_item_indent = None;
                }
            }
        }
    }
    out
}
/// Detect the installed launcher without writing anything.
pub(crate) fn detect(env: &DetectEnv, runner: &dyn DshRunner) -> DshDetection {


    let (dsh_home, dsh_home_from_env) = resolve_dsh_home(env);
    let dsh_home_exists = dsh_home.is_dir();
    let mut profiles = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dsh_home.join("profiles")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == "node_modules" || name.starts_with('.') {
                continue;
            }
            if entry.path().is_dir() {
                profiles.push(name);
            }
        }
    }
    profiles.sort();
    let settings_namespaces = std::fs::read(dsh_home.join("settings.yaml"))
        .ok()
        .map(|bytes| {
            let bytes = &bytes[..bytes.len().min(64 * 1024)];
            settings_namespaces(&String::from_utf8_lossy(bytes))
        })
        .unwrap_or_default();
    let credentials_path = dsh_home.join(".credentials.yaml");
    let credentials_present = credentials_path.is_file();
    let credentials_mode_ok = credentials_mode_ok(&credentials_path);

    let Some(binary) = find_on_path(env.path.as_ref()) else {
        return DshDetection {
            binary: None,
            version: None,
            compatibility: DshCompatibility::Offline {
                reason: "dsh is not on PATH".to_string(),
            },
            supports_patch: false,
            dsh_home,
            dsh_home_exists,
            dsh_home_from_env,
            profiles,
            settings_namespaces,
            credentials_present,
            credentials_mode_ok,
        };
    };

    let (version, supports_patch, compatibility) = match runner.run(&binary, &["--version"]) {
        Ok((true, text)) => {
            let version = text.trim().lines().last().unwrap_or("").trim().to_string();
            let supports_patch = match runner.run(&binary, &["--help"]) {
                Ok((_, help)) => help.contains("--patch"),
                Err(_) => false,
            };
            let compatibility = classify_version(&version, supports_patch);
            (Some(version), supports_patch, compatibility)
        }
        Ok((false, text)) => (
            None,
            false,
            DshCompatibility::Offline {
                reason: format!(
                    "dsh --version exited non-zero: {}",
                    text.trim().lines().next().unwrap_or("").chars().take(120).collect::<String>()
                ),
            },
        ),
        Err(error) => (
            None,
            false,
            DshCompatibility::Offline {
                reason: format!("dsh could not be run: {error}"),
            },
        ),
    };

    DshDetection {
        binary: Some(binary),
        version,
        compatibility,
        supports_patch,
        dsh_home,
        dsh_home_exists,
        dsh_home_from_env,
        profiles,
        settings_namespaces,
        credentials_present,
        credentials_mode_ok,
    }
}

#[cfg(unix)]
fn credentials_mode_ok(path: &Path) -> Option<bool> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(path).ok()?;
    Some(meta.permissions().mode() & 0o077 == 0)
}

#[cfg(not(unix))]
fn credentials_mode_ok(_path: &Path) -> Option<bool> {
    None
}


#[cfg(test)]
mod tests {
    use super::*;

    struct FakeRunner {
        version: (bool, String),
        help: (bool, String),
    }

    impl FakeRunner {
        fn new(version: &str, help: &str) -> Self {
            Self {
                version: (true, version.to_string()),
                help: (true, help.to_string()),
            }
        }
    }

    impl DshRunner for FakeRunner {
        fn run(&self, _binary: &Path, args: &[&str]) -> std::io::Result<(bool, String)> {
            match args.first().copied() {
                Some("--version") => Ok(self.version.clone()),
                Some("--help") => Ok(self.help.clone()),
                other => Err(std::io::Error::other(format!("unexpected args {other:?}"))),
            }
        }
    }

    fn launcher_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        let launcher = dir.path().join("dsh");
        std::fs::write(&launcher, "#!/bin/sh\n").expect("fake launcher");
        dir
    }

    fn env_for(dir: &tempfile::TempDir, home: &Path) -> DetectEnv {
        DetectEnv {
            path: Some(OsString::from(dir.path().as_os_str())),
            home: Some(home.to_path_buf()),
            dsh_home: None,
        }
    }

    #[test]
    fn parses_released_and_prerelease_versions() {
        assert_eq!(
            Version::parse("0.2.1-alpha.2").unwrap().to_string(),
            "0.2.1-alpha.2"
        );
        assert_eq!(
            Version::parse(" v0.1.0-rc.6 ").unwrap().to_string(),
            "0.1.0-rc.6"
        );
        assert_eq!(Version::parse("1.2.3+build.9").unwrap().to_string(), "1.2.3");
        for bad in ["", "nightly", "0.1", "0.1.0-", "1.2.3.4", "1.2.3-01", "x.y.z"] {
            assert!(Version::parse(bad).is_none(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn prerelease_precedence_matches_semver() {
        let order = [
            "0.1.0-alpha",
            "0.1.0-alpha.1",
            "0.1.0-alpha.beta",
            "0.1.0-beta",
            "0.1.0-beta.2",
            "0.1.0-beta.11",
            "0.1.0-rc.1",
            "0.1.0",
            "0.1.1",
            "0.2.0-rc.1",
            "0.2.0",
            "0.2.1-alpha.1",
            "0.2.1-alpha.2",
        ];
        for pair in order.windows(2) {
            let left = Version::parse(pair[0]).expect("left parses");
            let right = Version::parse(pair[1]).expect("right parses");
            assert_eq!(
                left.cmp_precedence(&right),
                std::cmp::Ordering::Less,
                "{} should precede {}",
                pair[0],
                pair[1]
            );
            assert_eq!(
                right.cmp_precedence(&left),
                std::cmp::Ordering::Greater,
                "{} should follow {}",
                pair[1],
                pair[0]
            );
        }
    }

    #[test]
    fn classifies_verified_newer_and_older_launchers() {
        assert_eq!(
            classify_version(VERIFIED_DSH_VERSION, true),
            DshCompatibility::Verified
        );
        assert_eq!(classify_version("0.3.0", true).label(), "newer-unverified");
        assert_eq!(classify_version("0.2.0-rc.1", true).label(), "incompatible");
        assert_eq!(classify_version("nightly", true).label(), "unparsed");
        assert_eq!(
            classify_version("0.2.1-alpha.2", false).label(),
            "incompatible"
        );
    }



    #[test]
    fn modern_native_deepseek_follows_the_core_version() {
        let dir = launcher_dir();
        let home = tempfile::tempdir().expect("home");
        let detection = |version: &str| {
            detect(
                &env_for(&dir, home.path()),
                &FakeRunner::new(version, "Usage: dsh [--profile] <name> --patch <path>"),
            )
        };
        assert!(detection("0.2.1-alpha.2").modern_native_deepseek());
        assert!(detection("0.2.0-rc.1").modern_native_deepseek());
        assert!(!detection("0.1.7-alpha.2").modern_native_deepseek());
        assert!(!detection("nightly").modern_native_deepseek());
    }

    #[test]
    fn resolves_dsh_home_from_env_then_home() {
        let home = PathBuf::from("/home/tester");
        let default = resolve_dsh_home(&DetectEnv {
            path: None,
            home: Some(home.clone()),
            dsh_home: None,
        });
        assert_eq!(default, (home.join(".dsh"), false));

        let tilde = resolve_dsh_home(&DetectEnv {
            path: None,
            home: Some(home.clone()),
            dsh_home: Some(OsString::from("~/harness")),
        });
        assert_eq!(tilde, (home.join("harness"), true));

        let absolute = resolve_dsh_home(&DetectEnv {
            path: None,
            home: Some(home),
            dsh_home: Some(OsString::from("/srv/dsh")),
        });
        assert_eq!(absolute, (PathBuf::from("/srv/dsh"), true));

        let blank = resolve_dsh_home(&DetectEnv {
            path: None,
            home: Some(PathBuf::from("/home/tester")),
            dsh_home: Some(OsString::from("   ")),
        });
        assert!(!blank.1);
    }

    #[test]
    fn settings_namespaces_reads_only_top_level_keys() {
        let text = "# comment\nagent-default-model:\n  provider: deepseek-official\nllm-pi-ai:\nllm-deepseek: {}\n- list item\n'quoted': 1\n";
        assert_eq!(
            settings_namespaces(text),
            vec!["agent-default-model", "llm-pi-ai", "llm-deepseek"]
        );
    }

    #[test]
    fn detect_reports_an_absent_launcher_without_writing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let home = tempfile::tempdir().expect("home");
        let detection = detect(
            &DetectEnv {
                path: Some(OsString::from(dir.path().as_os_str())),
                home: Some(home.path().to_path_buf()),
                dsh_home: None,
            },
            &FakeRunner::new("0.2.1-alpha.2", "--patch"),
        );
        assert!(!detection.installed());
        assert_eq!(detection.compatibility.label(), "offline");
        assert_eq!(detection.dsh_home, home.path().join(".dsh"));
        assert!(!detection.dsh_home_exists);
        // Detection is read-only: nothing was created next to the fake home.
        assert_eq!(
            std::fs::read_dir(home.path()).expect("home readable").count(),
            0
        );
    }
}
