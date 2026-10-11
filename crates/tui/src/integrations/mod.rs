//! External-tool integrations this client ships.
//!
//! Each integration is a thin, reversible adapter around another tool's
//! documented seams: it detects the tool, writes only inside this client's own
//! home, and never edits the other tool's installation.

pub(crate) mod cli;
pub(crate) mod dsh;
