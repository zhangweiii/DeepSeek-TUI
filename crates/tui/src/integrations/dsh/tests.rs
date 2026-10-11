//! State-machine tests for the DSH integration.
//!
//! Every test injects its own home, `$DSH_HOME`, fake launcher, and runner, so
//! nothing here reads or writes the developer's real configuration, and the
//! real `audit.log` is never touched.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::detect::{DetectEnv, DshRunner};
use super::*;

struct FakeRunner {
    version: (bool, String),
    help: String,
}

impl FakeRunner {
    fn at(version: &str) -> Self {
        Self {
            version: (true, version.to_string()),
            help: "Usage: dsh [--profile] <name> [options] [app-args...]\n\nOptions:\n  --patch <path>  extra patch-list overlay\n".to_string(),
        }
    }
}

impl DshRunner for FakeRunner {
    fn run(&self, _binary: &Path, args: &[&str]) -> std::io::Result<(bool, String)> {
        match args.first().copied() {
            Some("--version") => Ok(self.version.clone()),
            Some("--help") => Ok((true, self.help.clone())),
            other => Err(std::io::Error::other(format!("unexpected args {other:?}"))),
        }
    }
}

/// A throwaway machine: app home, `$DSH_HOME`, a fake `dsh` on PATH, a runner.
struct Machine {
    app_home: tempfile::TempDir,
    dsh_home: tempfile::TempDir,
    workspace: tempfile::TempDir,
    paths: DshPaths,
    env: DetectEnv,
    runner: FakeRunner,
}

impl Machine {
    fn new(version: &str) -> Self {
        let app_home = tempfile::tempdir().expect("app home");
        let dsh_home = tempfile::tempdir().expect("dsh home");
        let workspace = tempfile::tempdir().expect("workspace");
        let bin_dir = tempfile::tempdir().expect("bin dir");
        let launcher = bin_dir.path().join("dsh");
        std::fs::write(&launcher, "#!/bin/sh\nexit 0\n").expect("fake launcher");
        let paths = DshPaths::under(app_home.path());
        let env = DetectEnv {
            path: Some(OsString::from(bin_dir.path().as_os_str())),
            home: Some(app_home.path().to_path_buf()),
            dsh_home: Some(OsString::from(dsh_home.path().as_os_str())),
        };
        // Keep the fake bin dir alive for the process lifetime of the test.
        std::mem::forget(bin_dir);
        Self {
            app_home,
            dsh_home,
            workspace,
            paths,
            env,
            runner: FakeRunner::at(version),
        }
    }

    fn status(&self, route: &RouteIdentity, profile: &str) -> DshStatusReport {
        compute_status(
            &self.paths,
            &self.env,
            &self.runner,
            &RouteResolution {
                identity: Some(route.clone()),
                error: None,
            },
            profile,
        )
        .expect("status")
    }

    fn connect(&self, route: &RouteIdentity, profile: &str) -> DshConnectionRecord {
        let report = self.status(route, profile);
        let plan = plan(
            &self.paths,
            &report.detection,
            profile,
            route,
            false,
            DshReceiptEvent::Connect,
            report.disclosures(),
        )
        .expect("plan");
        apply_plan(&self.paths, &plan, &report.detection).expect("apply")
    }

    fn audit_log(&self) -> String {
        std::fs::read_to_string(self.paths.audit_log()).unwrap_or_default()
    }

    fn dsh_home_entries(&self) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(self.dsh_home.path())
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }
}

fn route() -> RouteIdentity {
    RouteIdentity {
        provider_id: "deepseek".to_string(),
        provider_label: "DeepSeek".to_string(),
        model: "deepseek-v4-pro".to_string(),
        base_url: "https://api.deepseek.com".to_string(),
        api_key_env: Some("DEEPSEEK_API_KEY".to_string()),
        credential_available: true,
        keyless_local: false,
        reasoning_effort: Some("high".to_string()),
        sandbox_mode: Some("workspace-write".to_string()),
        approval_policy: Some("on-request".to_string()),
        yolo: false,
        workspace: "/work/project".to_string(),
    }
}

#[test]
fn status_reports_not_installed_without_a_launcher() {
    let mut machine = Machine::new(VERIFIED_DSH_VERSION);
    machine.env.path = Some(OsString::from("/nonexistent/bin"));
    let report = machine.status(&route(), "web");
    assert_eq!(report.state, DshIntegrationState::NotInstalled);
    assert!(!report.overlay_present);
    assert!(report.disclosures().iter().any(|line| line.contains("install")));
}

#[test]
fn status_reports_incompatible_for_an_older_launcher() {
    let machine = Machine::new("0.1.7-alpha.2");
    let report = machine.status(&route(), "web");
    assert_eq!(report.state.label(), "incompatible");
}

#[test]
fn status_reports_stale_version_for_a_newer_launcher() {
    let machine = Machine::new("0.3.0");
    // A newer preview is still launchable: connect writes the overlay, and the
    // state is reported as unverified rather than refused.
    machine.connect(&route(), "web");
    let report = machine.status(&route(), "web");
    assert_eq!(report.state.label(), "stale-version");
    assert!(report.state.launchable());
    assert!(
        report
            .disclosures()
            .iter()
            .any(|line| line.contains("newer than the verified"))
    );
}

#[test]
fn status_reports_detected_before_any_overlay_exists() {
    let machine = Machine::new(VERIFIED_DSH_VERSION);
    let report = machine.status(&route(), "web");
    assert_eq!(report.state.label(), "detected");
    assert!(!report.overlay_present);
}

#[test]
fn connect_writes_only_inside_its_own_root() {
    let machine = Machine::new(VERIFIED_DSH_VERSION);
    let record = machine.connect(&route(), "web");

    let overlay = std::fs::read_to_string(&machine.paths.overlay).expect("overlay written");
    assert_eq!(sha256_hex(overlay.as_bytes()), record.overlay_sha256);
    assert!(machine.paths.receipt.is_file());
    assert!(machine.paths.audit_log().is_file());
    // The harness home is untouched: detection never writes there.
    assert!(machine.dsh_home_entries().is_empty());
    // The receipt names the credential variable, never a value.
    let receipt = std::fs::read_to_string(&machine.paths.receipt).expect("receipt");
    assert!(receipt.contains("DEEPSEEK_API_KEY"));
    assert!(!receipt.contains("sk-"));
    let audit = machine.audit_log();
    assert!(audit.contains("dsh_connect"));

    let report = machine.status(&route(), "web");
    assert_eq!(report.state.label(), "connected");
    assert!(report.state.launchable());
    assert_eq!(
        report.overlay_sha256_on_disk.as_deref(),
        Some(record.overlay_sha256.as_str())
    );
}

#[test]
fn a_changed_route_makes_the_overlay_stale_until_update() {
    let machine = Machine::new(VERIFIED_DSH_VERSION);
    machine.connect(&route(), "web");

    let mut moved = route();
    moved.model = "deepseek-v4-flash".to_string();
    let report = machine.status(&moved, "web");
    match &report.state {
        DshIntegrationState::StaleConfig { reason, .. } => {
            assert!(reason.contains("deepseek-v4-flash"), "{reason}");
        }
        other => panic!("expected stale-config, got {other:?}"),
    }
    assert!(!report.state.launchable());

    // Updating re-derives the overlay from the new route and connects again.
    let plan = plan(
        &machine.paths,
        &report.detection,
        "web",
        &moved,
        false,
        DshReceiptEvent::Update,
        Vec::new(),
    )
    .expect("plan");
    let record = apply_plan(&machine.paths, &plan, &report.detection).expect("apply");
    assert_eq!(record.identity.source.model, "deepseek-v4-flash");
    assert_eq!(machine.status(&moved, "web").state.label(), "connected");
}

#[test]
fn an_overlay_edited_outside_this_client_is_reported_stale() {
    let machine = Machine::new(VERIFIED_DSH_VERSION);
    machine.connect(&route(), "web");
    std::fs::write(&machine.paths.overlay, "# hand edited\n").expect("edit overlay");
    let report = machine.status(&route(), "web");
    match &report.state {
        DshIntegrationState::StaleConfig { reason, .. } => {
            assert!(reason.contains("modified outside"), "{reason}");
        }
        other => panic!("expected stale-config, got {other:?}"),
    }
}

#[test]
fn a_deleted_overlay_is_reported_stale() {
    let machine = Machine::new(VERIFIED_DSH_VERSION);
    machine.connect(&route(), "web");
    std::fs::remove_file(&machine.paths.overlay).expect("delete overlay");
    let report = machine.status(&route(), "web");
    match &report.state {
        DshIntegrationState::StaleConfig { reason, .. } => {
            assert!(reason.contains("missing"), "{reason}");
        }
        other => panic!("expected stale-config, got {other:?}"),
    }
}

#[test]
fn disable_and_enable_keep_the_overlay_on_disk() {
    let machine = Machine::new(VERIFIED_DSH_VERSION);
    machine.connect(&route(), "web");

    let disabled = set_disabled(&machine.paths, true).expect("disable");
    assert!(disabled.disabled);
    assert!(machine.paths.overlay.is_file());
    let report = machine.status(&route(), "web");
    assert_eq!(report.state.label(), "disabled");
    assert!(!report.state.launchable());

    let enabled = set_disabled(&machine.paths, false).expect("enable");
    assert!(!enabled.disabled);
    assert_eq!(machine.status(&route(), "web").state.label(), "connected");
    assert!(machine.audit_log().contains("dsh_disable"));
    assert!(machine.audit_log().contains("dsh_enable"));
}

#[test]
fn remove_deletes_only_this_integrations_files() {
    let machine = Machine::new(VERIFIED_DSH_VERSION);
    machine.connect(&route(), "web");
    let keep = machine.app_home.path().join("config.toml");
    std::fs::write(&keep, "# user config\n").expect("user file");

    let removed = remove(&machine.paths).expect("remove");
    assert_eq!(removed.len(), 2);
    assert!(!machine.paths.overlay.exists());
    assert!(!machine.paths.receipt.exists());
    // The integration directory is pruned, the rest of the app home is not.
    assert!(!machine.paths.root.exists());
    assert!(keep.is_file());
    assert!(machine.audit_log().contains("dsh_remove"));
    assert_eq!(machine.status(&route(), "web").state.label(), "detected");
}

