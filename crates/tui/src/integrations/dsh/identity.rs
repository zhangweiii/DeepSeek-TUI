//! Exact DeepSeek TUI route identity → DeepSeek Harness overlay mapping.
//!
//! The mapping carries *identity only*: provider id, model id, endpoint,
//! reasoning tier, permission posture, and the *name* of the credential
//! environment variable. It never carries a credential value, an OAuth
//! document, or a base URL that embeds userinfo/query material.
//!
//! ## Why the adapter choice depends on the installed `dsh`
//!
//! `0.2.0` split the native DeepSeek adapter into `[route] deepseek-official`
//! (`@deepseek-ai/dsh-llm-deepseek-api-key`) and `deepseek-account`
//! (`@deepseek-ai/dsh-llm-deepseek-account`), and the native wire call now
//! targets DeepSeek's Anthropic-compatible endpoint
//! (`https://api.deepseek.com/anthropic`). This client speaks OpenAI chat
//! completions on every provider, so on a `0.2`-or-newer launcher the route is
//! carried by a **hand-declared `@deepseek-ai/dsh-llm-pi-ai` route** whose
//! `api:` names `openai-completions` — the dialect is preserved, never
//! approximated. On a `0.1.x` launcher the native adapter still speaks chat
//! completions, so the route rides the native `deepseek-official` route with
//! its `reasoningEffort` mapping.
//!
//! Nothing here pins a plugin *package name*: a `name:` row in a `--patch`
//! overlay is an assertion that is skipped (with a warning) when it does not
//! match, so the overlay targets stable entry **ids** instead.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Non-secret facts about the route this client is configured to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RouteIdentity {
    /// Exact configured provider id (e.g. `deepseek`, `deepseek-cn`, `vllm`).
    pub(crate) provider_id: String,
    /// Human label for the provider.
    pub(crate) provider_label: String,
    /// Exact model id put on the wire.
    pub(crate) model: String,
    /// Resolved base URL (structural; must not carry credentials).
    pub(crate) base_url: String,
    /// Canonical credential env var name for the provider, if it has one.
    pub(crate) api_key_env: Option<String>,
    /// True when `api_key_env` is set in this process's environment, so the
    /// harness will resolve it from the environment it inherits. When false the
    /// overlay declares no credential reference at all: a named-but-unresolved
    /// reference is a hard failure in the harness.
    pub(crate) credential_available: bool,
    /// True when the route is a keyless self-hosted endpoint: no credential
    /// reference is written.
    pub(crate) keyless_local: bool,
    /// Reasoning tier as configured (`off|low|medium|high|max`).
    pub(crate) reasoning_effort: Option<String>,
    /// Sandbox mode (`read-only|workspace-write|danger-full-access|…`).
    pub(crate) sandbox_mode: Option<String>,
    /// Approval policy (`suggest|auto|never`).
    pub(crate) approval_policy: Option<String>,
    pub(crate) yolo: bool,
    /// Workspace the launch is bound to.
    pub(crate) workspace: String,
}

/// Which overlay dialect the installed launcher needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum DshFlavor {
    /// `0.1.x`: the native DeepSeek adapter speaks OpenAI chat completions.
    Legacy,
    /// `0.2.x` and newer: the native DeepSeek adapter speaks DeepSeek's
    /// Anthropic-compatible endpoint, and chat-completions routes ride a
    /// hand-declared `dsh-llm-pi-ai` route instead.
    Modern,
}

impl DshFlavor {
    /// The flavor a detection implies.
    pub(crate) fn of(modern_native_deepseek: bool) -> Self {
        if modern_native_deepseek {
            Self::Modern
        } else {
            Self::Legacy
        }
    }
}

/// DSH permission mode mirrored from this client's posture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum DshPermissionMode {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl DshPermissionMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }
}

/// Where the mapped route lands inside the harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum DshAdapter {
    /// `@deepseek-ai/dsh-llm-deepseek` (route `deepseek-official`), where the
    /// `0.1.x` native adapter speaks chat completions and exposes
    /// `reasoningEffort`.
    DeepseekNative,
    /// `@deepseek-ai/dsh-llm-pi-ai` hand-declared route named after this
    /// client's provider. Carries `api: openai-completions`.
    PiAiOpenAiCompatible { route_id: String },
    /// DSH cannot carry this route; nothing is written for it.
    Unsupported { reason: String },
}

/// The identity as it will be written into the overlay, plus disclosures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct MappedIdentity {
    pub(crate) source: RouteIdentity,
    pub(crate) flavor: DshFlavor,
    pub(crate) adapter: DshAdapter,
    /// DSH `reasoningEffort` (`off|high|max`) for the native adapter; `None`
    /// when there is no explicit tier or the adapter cannot express it.
    pub(crate) dsh_reasoning_effort: Option<String>,
    pub(crate) permission_mode: DshPermissionMode,
    /// Facts the user must know that the overlay cannot enforce.
    pub(crate) disclosures: Vec<String>,
}

impl MappedIdentity {
    pub(crate) fn mappable(&self) -> bool {
        !matches!(self.adapter, DshAdapter::Unsupported { .. })
    }

    /// Short label for the adapter that carries the route.
    pub(crate) fn adapter_label(&self) -> String {
        match &self.adapter {
            DshAdapter::DeepseekNative => "the native DeepSeek adapter".to_string(),
            DshAdapter::PiAiOpenAiCompatible { route_id } => {
                format!("dsh-llm-pi-ai route `{route_id}`")
            }
            DshAdapter::Unsupported { .. } => "nothing (unsupported route)".to_string(),
        }
    }
}


/// The `reasoningEffort` values `@deepseek-ai/dsh-llm-deepseek` accepts
/// (`off | low | high | max`). Tiers this client has no equivalent for are
/// reported as unmapped rather than rounded silently.
pub(crate) fn dsh_reasoning_effort(effort: Option<&str>) -> Option<&'static str> {
    match effort?.trim().to_ascii_lowercase().as_str() {
        "off" | "none" | "disabled" => Some("off"),
        "high" => Some("high"),
        "max" | "ultra" => Some("max"),
        "low" => Some("low"),
        _ => None,
    }
}

/// Map this client's posture onto a DSH permission mode, never broadening it
/// without `allow_full_access`.
pub(crate) fn permission_mode_for(
    identity: &RouteIdentity,
    allow_full_access: bool,
) -> (DshPermissionMode, Option<String>) {
    let sandbox = identity
        .sandbox_mode
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase());
    let approval = identity
        .approval_policy
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase());
    let full_access = identity.yolo
        || matches!(
            sandbox.as_deref(),
            Some("danger-full-access" | "external-sandbox")
        );
    if matches!(sandbox.as_deref(), Some("read-only")) {
        return (DshPermissionMode::ReadOnly, None);
    }
    if full_access {
        if allow_full_access {
            return (
                DshPermissionMode::DangerFullAccess,
                Some(
                    "DSH danger-full-access mirrors this client's full access; DSH will not ask before file effects."
                        .to_string(),
                ),
            );
        }
        return (
            DshPermissionMode::WorkspaceWrite,
            Some(
                "This client runs with full access, but the DSH overlay stays at workspace-write; pass --allow-full-access to mirror it."
                    .to_string(),
            ),
        );
    }
    let note = match approval.as_deref() {
        Some("never" | "deny" | "denied") => Some(
            "This client's approval policy is `never`; DSH keeps its own ask-before-effects policy at workspace-write."
                .to_string(),
        ),
        _ => None,
    };
    (DshPermissionMode::WorkspaceWrite, note)
}

/// A base URL is structural when it is an `http(s)` URL with no userinfo,
/// query, or fragment: those carry credentials or per-request material and are
/// never copied into the overlay.
fn base_url_is_structural(url: &str) -> Result<(), String> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return Err("empty base URL".to_string());
    }
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return Err("base URL has no scheme".to_string());
    };
    if !matches!(scheme, "http" | "https") {
        return Err(format!("unsupported base URL scheme `{scheme}`"));
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains('@') {
        return Err("base URL embeds userinfo; refusing to copy it".to_string());
    }
    if rest.contains('?') || rest.contains('#') {
        return Err("base URL carries a query/fragment; refusing to copy it".to_string());
    }
    Ok(())
}

/// The route id a hand-declared `dsh-llm-pi-ai` provider is registered under.
/// Stable for a given provider id so the overlay hash only moves when the
/// identity moves.
fn route_id_for(provider_id: &str) -> String {
    let mut out = String::from("deeptui-");
    for ch in provider_id.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let trimmed = out.trim_end_matches('-');
    if trimmed.len() <= "deeptui-".len() {
        format!("{trimmed}-provider")
    } else {
        trimmed.to_string()
    }
}

/// The `api:` dialect a hand-declared `dsh-llm-pi-ai` route names. This client
/// speaks OpenAI chat completions on every provider, so the dialect is carried
/// exactly rather than approximated.
pub(crate) fn pi_ai_api() -> &'static str {
    "openai-completions"
}

/// Map a route identity onto what the installed launcher can carry.
pub(crate) fn map_identity(
    identity: &RouteIdentity,
    flavor: DshFlavor,
    allow_full_access: bool,
) -> MappedIdentity {
    let mut disclosures = Vec::new();
    let (permission_mode, note) = permission_mode_for(identity, allow_full_access);
    if let Some(note) = note {
        disclosures.push(note);
    }
    let dsh_effort = dsh_reasoning_effort(identity.reasoning_effort.as_deref()).map(str::to_string);

    if let Err(reason) = base_url_is_structural(&identity.base_url) {
        return MappedIdentity {
            source: identity.clone(),
            flavor,
            adapter: DshAdapter::Unsupported { reason },
            dsh_reasoning_effort: None,
            permission_mode,
            disclosures,
        };
    }

    let is_deepseek = matches!(identity.provider_id.as_str(), "deepseek" | "deepseek-cn");
    if is_deepseek && flavor == DshFlavor::Legacy {
        if identity.reasoning_effort.is_some() && dsh_effort.is_none() {
            disclosures.push(format!(
                "Reasoning tier `{}` has no DSH equivalent; DSH keeps its own default.",
                identity.reasoning_effort.as_deref().unwrap_or("")
            ));
        }
        disclosures.push(
            "DSH resolves DEEPSEEK_API_KEY from its own environment or $DSH_HOME/.credentials.yaml; this client does not hand over a key."
                .to_string(),
        );
        return MappedIdentity {
            source: identity.clone(),
            flavor,
            adapter: DshAdapter::DeepseekNative,
            dsh_reasoning_effort: dsh_effort,
            permission_mode,
            disclosures,
        };
    }

    if flavor == DshFlavor::Modern && is_deepseek {
        disclosures.push(
            "The installed dsh adapter serves the native DeepSeek route over DeepSeek's Anthropic-compatible endpoint; this client speaks OpenAI chat completions, so the route is carried by a hand-declared dsh-llm-pi-ai provider (`api: openai-completions`) instead."
                .to_string(),
        );
    }
    if identity.reasoning_effort.is_some() {
        disclosures.push(
            "Reasoning tier is not mapped for hand-declared DSH routes (per-provider wire spellings are not verified); DSH sends no effort parameter."
                .to_string(),
        );
    }
    if identity.keyless_local {
        disclosures.push(
            "Keyless local route: no credential reference is written; DSH talks to the endpoint without a key."
                .to_string(),
        );
    } else if let Some(env) = identity.api_key_env.as_deref() {
        if identity.credential_available {
            disclosures.push(format!(
                "DSH resolves {env} from the environment it inherits or $DSH_HOME/.credentials.yaml; this client never copies a key."
            ));
        } else {
            disclosures.push(format!(
                "{env} is not set in this environment, so the overlay declares no credential reference; dsh needs its own credential (sign in to dsh, or export {env}) before requests can succeed."
            ));
        }
    } else {
        disclosures.push(
            "No credential env var is known for this provider; DSH will defer to its ambient credential discovery."
                .to_string(),
        );
    }
    MappedIdentity {
        source: identity.clone(),
        flavor,
        adapter: DshAdapter::PiAiOpenAiCompatible {
            route_id: route_id_for(&identity.provider_id),
        },
        dsh_reasoning_effort: None,
        permission_mode,
        disclosures,
    }
}

fn yaml_str(value: &str) -> String {
    // Single-quoted YAML scalar: only `'` needs escaping.
    format!("'{}'", value.replace('\'', "''"))
}

/// Render the DSH `--patch` overlay for a mapped identity. Deterministic for a
/// given identity so its SHA-256 can detect drift.
///
/// Every row is addressed by entry **id** and carries a whole `config`: a DSH
/// patch entry replaces the target row's config (it does not merge), and a
/// `name:` assertion that does not match would be skipped silently.
pub(crate) fn render_overlay(mapped: &MappedIdentity) -> Option<String> {
    let src = &mapped.source;
    if !mapped.mappable() {
        return None;
    }
    let mut out = String::new();
    out.push_str("# DeepSeek Harness connected through DeepSeek TUI.\n");
    out.push_str("# Generated by `deepseek integrations dsh connect`; do not edit by hand.\n");
    out.push_str("# Identity only: no API key, token, or credential document is written here.\n");
    out.push_str(&format!(
        "# provider={} model={} workspace={}\n",
        src.provider_id, src.model, src.workspace
    ));
    match &mapped.adapter {
        DshAdapter::DeepseekNative => {
            out.push_str("- id: agent-default-model\n");
            out.push_str("  config:\n");
            out.push_str("    provider: deepseek-official\n");
            out.push_str(&format!("    model: {}\n", yaml_str(&src.model)));
            out.push_str("- id: llm-deepseek\n");
            out.push_str("  config:\n");
            out.push_str(&format!("    baseURL: {}\n", yaml_str(&src.base_url)));
            if let Some(effort) = mapped.dsh_reasoning_effort.as_deref() {
                out.push_str(&format!("    reasoningEffort: {effort}\n"));
            }
            out.push_str("    models:\n");
            out.push_str(&format!("      - id: {}\n", yaml_str(&src.model)));
            out.push_str(&format!("        name: {}\n", yaml_str(&src.model)));
        }
        DshAdapter::PiAiOpenAiCompatible { route_id } => {
            out.push_str("- id: agent-default-model\n");
            out.push_str("  config:\n");
            out.push_str(&format!("    provider: {}\n", yaml_str(route_id)));
            out.push_str(&format!("    model: {}\n", yaml_str(&src.model)));
            out.push_str("- id: llm-pi-ai\n");
            out.push_str("  config:\n");
            out.push_str("    providers:\n");
            out.push_str(&format!("      {}:\n", yaml_str(route_id)));
            out.push_str(&format!(
                "        displayName: {}\n",
                yaml_str(&format!("{} (via DeepSeek TUI)", src.provider_label))
            ));
            if !src.keyless_local
                && src.credential_available
                && let Some(env) = src.api_key_env.as_deref()
            {
                out.push_str(&format!("        apiKeyEnv: {}\n", yaml_str(env)));
            }
            out.push_str(&format!("        api: {}\n", pi_ai_api()));
            out.push_str(&format!("        baseURL: {}\n", yaml_str(&src.base_url)));
            out.push_str("        models:\n");
            out.push_str(&format!("          - id: {}\n", yaml_str(&src.model)));
            out.push_str(&format!("            name: {}\n", yaml_str(&src.model)));
        }
        DshAdapter::Unsupported { .. } => return None,
    }
    Some(out)
}

/// The dialect disclosure line, kept next to the renderer so both stay honest.
pub(crate) fn dialect_disclosure() -> String {
    format!(
        "Routes are carried in their own wire dialect (`api: {}`), never approximated.",
        pi_ai_api()
    )
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    fn route(provider: &str) -> RouteIdentity {
        RouteIdentity {
            provider_id: provider.to_string(),
            provider_label: if provider == "deepseek" {
                "DeepSeek".to_string()
            } else {
                "vLLM".to_string()
            },
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
    fn modern_deepseek_routes_ride_a_hand_declared_pi_ai_provider() {
        let mapped = map_identity(&route("deepseek"), DshFlavor::Modern, false);
        assert!(mapped.mappable());
        let overlay = render_overlay(&mapped).expect("overlay renders");
        // No package-name assertion: a mismatched `name:` row is skipped.
        assert!(!overlay.contains("@deepseek-ai/"));
        assert!(overlay.contains("- id: llm-pi-ai\n"));
        assert!(overlay.contains("      'deeptui-deepseek':\n"));
        assert!(overlay.contains("        api: openai-completions\n"));
        assert!(overlay.contains("        apiKeyEnv: 'DEEPSEEK_API_KEY'\n"));
        assert!(overlay.contains("    provider: 'deeptui-deepseek'\n"));
        assert!(overlay.contains("    model: 'deepseek-v4-pro'\n"));
        // The native route is not addressed on a 0.2.x launcher.
        assert!(!overlay.contains("id: llm-deepseek"));
        assert!(mapped.dsh_reasoning_effort.is_none());
    }

    #[test]
    fn legacy_deepseek_routes_ride_the_native_adapter_with_reasoning() {
        let mapped = map_identity(&route("deepseek"), DshFlavor::Legacy, false);
        let overlay = render_overlay(&mapped).expect("overlay renders");
        assert!(overlay.contains("- id: llm-deepseek\n"));
        assert!(overlay.contains("    baseURL: 'https://api.deepseek.com'\n"));
        assert!(overlay.contains("    reasoningEffort: high\n"));
        assert!(overlay.contains("    provider: deepseek-official\n"));
        assert_eq!(mapped.dsh_reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn keyless_local_routes_write_no_credential_reference() {
        let mut identity = route("vllm");
        identity.keyless_local = true;
        identity.api_key_env = None;
        identity.base_url = "http://127.0.0.1:8000/v1".to_string();
        let mapped = map_identity(&identity, DshFlavor::Modern, false);
        let overlay = render_overlay(&mapped).expect("overlay renders");
        assert!(!overlay.contains("apiKeyEnv"));
        assert!(overlay.contains("        baseURL: 'http://127.0.0.1:8000/v1'\n"));
    }


    #[test]
    fn a_missing_environment_credential_is_not_referenced() {
        let mut identity = route("deepseek");
        identity.credential_available = false;
        let mapped = map_identity(&identity, DshFlavor::Modern, false);
        let overlay = render_overlay(&mapped).expect("overlay renders");
        assert!(!overlay.contains("apiKeyEnv"));
        assert!(
            mapped
                .disclosures
                .iter()
                .any(|line| line.contains("needs its own credential"))
        );
    }

    #[test]
    fn credentialed_or_non_structural_base_urls_are_refused() {
        for bad in [
            "https://user:pass@api.example.com",
            "https://api.example.com?key=secret",
            "https://api.example.com#frag",
            "ftp://api.example.com",
            "",
        ] {
            let mut identity = route("deepseek");
            identity.base_url = bad.to_string();
            let mapped = map_identity(&identity, DshFlavor::Modern, false);
            assert!(!mapped.mappable(), "{bad:?} should be refused");
            assert!(render_overlay(&mapped).is_none());
            assert!(matches!(mapped.adapter, DshAdapter::Unsupported { .. }));
        }
    }

    #[test]
    fn permission_mode_never_broadens_without_the_flag() {
        let mut identity = route("deepseek");
        identity.sandbox_mode = Some("read-only".to_string());
        let mapped = map_identity(&identity, DshFlavor::Modern, true);
        assert_eq!(mapped.permission_mode, DshPermissionMode::ReadOnly);

        let mut identity = route("deepseek");
        identity.yolo = true;
        identity.sandbox_mode = Some("danger-full-access".to_string());
        let mapped = map_identity(&identity, DshFlavor::Modern, false);
        assert_eq!(mapped.permission_mode, DshPermissionMode::WorkspaceWrite);
        assert!(
            mapped
                .disclosures
                .iter()
                .any(|d| d.contains("--allow-full-access"))
        );

        let mapped = map_identity(&identity, DshFlavor::Modern, true);
        assert_eq!(mapped.permission_mode, DshPermissionMode::DangerFullAccess);
    }

    #[test]
    fn reasoning_tiers_map_only_where_the_adapter_can_express_them() {
        assert_eq!(dsh_reasoning_effort(Some("off")), Some("off"));
        assert_eq!(dsh_reasoning_effort(Some("LOW")), Some("low"));
        assert_eq!(dsh_reasoning_effort(Some("HIGH")), Some("high"));
        assert_eq!(dsh_reasoning_effort(Some("max")), Some("max"));
        assert_eq!(dsh_reasoning_effort(Some("medium")), None);
        assert_eq!(dsh_reasoning_effort(None), None);
    }

    #[test]
    fn the_overlay_is_deterministic_for_one_identity() {
        let first = render_overlay(&map_identity(&route("deepseek"), DshFlavor::Modern, false))
            .expect("overlay");
        let second = render_overlay(&map_identity(&route("deepseek"), DshFlavor::Modern, false))
            .expect("overlay");
        assert_eq!(first, second);
        assert_eq!(sha256_hex(first.as_bytes()), sha256_hex(second.as_bytes()));
        assert_eq!(sha256_hex(b"").len(), 64);
    }

    #[test]
    fn route_ids_are_stable_and_config_safe() {
        assert_eq!(route_id_for("deepseek"), "deeptui-deepseek");
        assert_eq!(route_id_for("deepseek-cn"), "deeptui-deepseek-cn");
        assert_eq!(route_id_for("vLLM"), "deeptui-vllm");
        assert_eq!(route_id_for("--"), "deeptui-provider");
    }

    #[test]
    fn the_overlay_never_mentions_a_credential_value() {
        let mapped = map_identity(&route("deepseek"), DshFlavor::Modern, false);
        let overlay = render_overlay(&mapped).expect("overlay");
        assert!(overlay.contains("apiKeyEnv"));
        assert!(dialect_disclosure().contains("openai-completions"));
        for line in overlay.lines() {
            if let Some(value) = line.trim().strip_prefix("apiKeyEnv:") {
                assert_eq!(value.trim(), "'DEEPSEEK_API_KEY'");
            }
        }
    }

    #[test]
    fn a_flavor_reports_what_the_detection_implied() {
        assert_eq!(DshFlavor::of(true), DshFlavor::Modern);
        assert_eq!(DshFlavor::of(false), DshFlavor::Legacy);
    }
}

