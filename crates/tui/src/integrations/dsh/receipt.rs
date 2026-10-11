//! Durable, append-only receipts for the DSH integration.
//!
//! One JSON document at `~/.deepseek/integrations/dsh/receipt.json` holds the
//! current connection record (or `null`) and an append-only history of every
//! connect/update/disable/enable/remove event. Receipts are read-only
//! summaries: they never carry credentials, only the *name* of a credential
//! environment variable.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::identity::MappedIdentity;

pub(crate) const RECEIPT_SCHEMA_VERSION: u32 = 1;
pub(crate) const MAX_HISTORY: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DshReceiptEvent {
    Connect,
    Update,
    Disable,
    Enable,
    Remove,
}

impl DshReceiptEvent {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::Update => "update",
            Self::Disable => "disable",
            Self::Enable => "enable",
            Self::Remove => "remove",
        }
    }
}

/// The live connection record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DshConnectionRecord {
    pub(crate) connected_at: String,
    pub(crate) updated_at: String,
    pub(crate) dsh_version: Option<String>,
    pub(crate) dsh_binary: Option<PathBuf>,
    pub(crate) dsh_home: PathBuf,
    /// The DSH profile the overlay is launched with (`web` or `headless`).
    pub(crate) profile: String,
    pub(crate) overlay_path: PathBuf,
    pub(crate) overlay_sha256: String,
    /// Launches are refused while this is set, with the overlay kept on disk.
    pub(crate) disabled: bool,
    pub(crate) identity: MappedIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DshReceiptEntry {
    pub(crate) event: DshReceiptEvent,
    pub(crate) at: String,
    pub(crate) deepseek_version: String,
    pub(crate) dsh_version: Option<String>,
    pub(crate) dsh_home: PathBuf,
    pub(crate) overlay_sha256: Option<String>,
    /// `provider/model` the overlay pinned, when applicable.
    pub(crate) identity_summary: Option<String>,
    pub(crate) permission_mode: Option<String>,
    pub(crate) note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DshReceiptDocument {
    pub(crate) schema_version: u32,
    pub(crate) current: Option<DshConnectionRecord>,
    #[serde(default)]
    pub(crate) history: Vec<DshReceiptEntry>,
}

impl Default for DshReceiptDocument {
    fn default() -> Self {
        Self {
            schema_version: RECEIPT_SCHEMA_VERSION,
            current: None,
            history: Vec::new(),
        }
    }
}

impl DshReceiptDocument {
    /// Load the receipt, or a fresh document when the file does not exist.
    /// A newer schema version is refused rather than silently rewritten.
    pub(crate) fn load(path: &Path) -> Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => {
                let doc: Self = serde_json::from_slice(&bytes)
                    .with_context(|| format!("parse DSH receipt {}", path.display()))?;
                if doc.schema_version > RECEIPT_SCHEMA_VERSION {
                    anyhow::bail!(
                        "DSH receipt {} uses schema {} but this build understands {}",
                        path.display(),
                        doc.schema_version,
                        RECEIPT_SCHEMA_VERSION
                    );
                }
                Ok(doc)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error)
                .with_context(|| format!("read DSH receipt {}", path.display())),
        }
    }

    /// Append one event, keeping the newest [`MAX_HISTORY`] entries.
    pub(crate) fn push(&mut self, entry: DshReceiptEntry) {
        self.history.push(entry);
        if self.history.len() > MAX_HISTORY {
            let excess = self.history.len() - MAX_HISTORY;
            self.history.drain(0..excess);
        }
    }

    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        let mut bytes = serde_json::to_vec_pretty(self).context("render DSH receipt")?;
        bytes.push(b'\n');
        write_atomic(path, &bytes)
    }
}

/// Write `bytes` to `path`, creating the parent directory, via a sibling
/// temporary file so a crash never leaves a half-written receipt.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let suffix = format!(
        ".tmp-{}-{}",
        std::process::id(),
        now_rfc3339().replace([':', '.'], "-")
    );
    let mut temp = path.as_os_str().to_os_string();
    temp.push(suffix);
    let temp = PathBuf::from(temp);
    std::fs::write(&temp, bytes).with_context(|| format!("write {}", temp.display()))?;
    match std::fs::rename(&temp, path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&temp);
            Err(error).with_context(|| format!("replace {}", path.display()))
        }
    }
}

/// RFC 3339 UTC timestamp, second precision.
pub(crate) fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
