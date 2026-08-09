//! Authoring tools for `content/`.
//!
//! The editor this crate exists for — a mouse-driven pixel and SFX tool — is not
//! here yet. What is here is the part everything else depends on and the part
//! that can be wrong silently: putting an edited record back into a file
//! without destroying the file around it.
//!
//! `content/FORMAT.md` §9 is the contract, and its first clause is the one that
//! decides the design: **never round-trip a file through a TOML serialiser.** A
//! serialiser is correct about TOML and wrong about this format. §5 makes `#`
//! and trailing whitespace DATA inside `'''`, and §6 makes the comments the most
//! valuable thing in the file — a diff of `content/blocks/terrain.toml` is
//! mostly prose explaining why each number is what it is. Re-emitting the file
//! normalises the first two and deletes the third, and it does it silently,
//! producing a diff that looks like "the whole file changed" and cannot be
//! reviewed.
//!
//! So the rule is textual: find the record's byte range, replace exactly that,
//! leave every other byte alone. That is what [`splice`] does.

pub mod splice;
