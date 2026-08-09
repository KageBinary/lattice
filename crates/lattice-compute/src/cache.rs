//! Kernel compilation, cached the way §15.5 asks for it.
//!
//! > Amortize compilation: Cache kernels by normalized expression, backend, precision, and
//! > hardware capabilities.
//!
//! All four appear in [`KernelKey`], and all four have to. Dropping the backend shares a
//! WGSL module with a CPU closure; dropping the precision hands an `f32` kernel to an
//! `f64` run; dropping the capabilities reuses a module specialized for one device's
//! workgroup limit on a device with a smaller one. The last is the one that would survive
//! testing on a single machine and fail in the field.
//!
//! # What "normalized" means here, and what it does not
//!
//! §15.5's normalization is semantic — it belongs to the expression compiler of §8.3,
//! which does not exist yet. What [`KernelSource::normalized`] does today is *lexical*:
//! comments are stripped and whitespace runs collapse to a single space, so two sources
//! that differ only in formatting share an entry.
//!
//! That is a weaker claim, and the direction it is weak in matters. Lexical normalization
//! produces **false misses** — `a+b` and `a + b` compile twice — which costs compilation
//! time and nothing else. It cannot produce a false *hit*, because whitespace and comments
//! are not semantic in any shader language this will target, and a false hit is the only
//! failure mode a cache has that is worse than not caching.
//!
//! # Inspectability
//!
//! §24.1: *"Generated code and shader sources must be cacheable and inspectable in debug
//! builds."* The cache keeps the source it compiled, retrievable by key, so a kernel that
//! misbehaves can be read rather than guessed at.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::{Backend, Capabilities, Precision};

/// A kernel before it is compiled: a body, plus the constants it was specialized on.
///
/// §15.5 asks for constants, material types, boundary modes and dimensions to be
/// specialized at compile time. Those choices are part of what the source *means*, so they
/// are part of the key, and keeping them beside the body rather than pasted into it means
/// the key sees them even when the backend splices them in later.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KernelSource {
    name: String,
    body: String,
    specializations: Vec<(String, String)>,
}

impl KernelSource {
    /// A kernel named `name` with the given body.
    pub fn new(name: impl Into<String>, body: impl Into<String>) -> KernelSource {
        KernelSource { name: name.into(), body: body.into(), specializations: Vec::new() }
    }

    /// Record a compile-time specialization.
    ///
    /// Sorted into place, so two callers that specialize the same constants in different
    /// orders produce the same key.
    pub fn specialize(mut self, key: impl Into<String>, value: impl ToString) -> KernelSource {
        let entry = (key.into(), value.to_string());
        match self.specializations.binary_search_by(|probe| probe.0.cmp(&entry.0)) {
            Ok(existing) => self.specializations[existing] = entry,
            Err(position) => self.specializations.insert(position, entry),
        }
        self
    }

    /// The kernel's name, for diagnostics and for §15.5's traceable mapping back to
    /// source.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The body exactly as written.
    pub fn body(&self) -> &str {
        &self.body
    }

    /// The specializations, sorted by key.
    pub fn specializations(&self) -> &[(String, String)] {
        &self.specializations
    }

    /// The body with comments removed and whitespace runs collapsed.
    ///
    /// See the module docs for what this does and does not claim.
    pub fn normalized(&self) -> String {
        let stripped = strip_comments(&self.body);
        let mut out = String::with_capacity(stripped.len());
        let mut pending_space = false;
        for character in stripped.chars() {
            if character.is_whitespace() {
                pending_space = !out.is_empty();
                continue;
            }
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(character);
        }
        out
    }
}

/// Remove `//` line comments and `/* */` block comments, leaving a space in their place.
///
/// The replacement space matters: deleting a comment between two identifiers would join
/// them into a third.
fn strip_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'/' && index + 1 < bytes.len() {
            match bytes[index + 1] {
                b'/' => {
                    while index < bytes.len() && bytes[index] != b'\n' {
                        index += 1;
                    }
                    out.push(' ');
                    continue;
                }
                b'*' => {
                    index += 2;
                    while index + 1 < bytes.len()
                        && !(bytes[index] == b'*' && bytes[index + 1] == b'/')
                    {
                        index += 1;
                    }
                    index = (index + 2).min(bytes.len());
                    out.push(' ');
                    continue;
                }
                _ => {}
            }
        }
        // Push whole characters, not bytes, so multi-byte source survives.
        let start = index;
        index += 1;
        while index < bytes.len() && !source.is_char_boundary(index) {
            index += 1;
        }
        out.push_str(&source[start..index]);
    }
    out
}

/// What a compiled kernel is looked up by.
///
/// Opaque and cheap to copy. Two keys are equal exactly when §15.5's four inputs agree.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct KernelKey(u64);

impl KernelKey {
    /// Derive the key from all four of §15.5's inputs.
    pub fn new(
        source: &KernelSource,
        backend: Backend,
        precision: Precision,
        capabilities: &Capabilities,
    ) -> KernelKey {
        let mut hash = FNV_OFFSET;
        // A separator between fields, so that moving a character across a boundary
        // changes the hash. Without it, ("ab", "c") and ("a", "bc") collide.
        let mut feed = |bytes: &[u8]| {
            hash = fnv1a_64_from(hash, bytes);
            hash = fnv1a_64_from(hash, &[0x1f]);
        };

        feed(source.name.as_bytes());
        feed(source.normalized().as_bytes());
        for (key, value) in &source.specializations {
            feed(key.as_bytes());
            feed(value.as_bytes());
        }
        feed(backend.name().as_bytes());
        feed(precision.name().as_bytes());
        feed(capabilities.adapter.as_bytes());
        for supported in &capabilities.precisions {
            feed(supported.name().as_bytes());
        }
        feed(&capabilities.max_workgroup.unwrap_or(0).to_le_bytes());
        feed(&capabilities.max_buffer_bytes.to_le_bytes());

        KernelKey(hash)
    }

    /// The key as it appears in logs and artifacts.
    pub fn as_hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Compiled kernels, kept so that compiling one is amortized across the run.
///
/// Generic over the backend's compiled type: a WGSL pipeline, a CUDA module, or — for the
/// CPU — nothing at all, since a closure needs no compiling. That last case is why this is
/// generic rather than holding a concrete artifact.
#[derive(Debug)]
pub struct KernelCache<T> {
    entries: HashMap<KernelKey, Compiled<T>>,
    hits: usize,
    misses: usize,
}

#[derive(Debug)]
struct Compiled<T> {
    kernel: T,
    /// Kept for §24.1's inspectability, not used for lookup.
    source: KernelSource,
}

impl<T> KernelCache<T> {
    /// An empty cache.
    pub fn new() -> KernelCache<T> {
        KernelCache { entries: HashMap::new(), hits: 0, misses: 0 }
    }

    /// Fetch the kernel for `key`, compiling it with `compile` if it is not present.
    ///
    /// `compile` receives the normalized source. It runs at most once per key, which is
    /// the amortization §15.5 asks for; a failure is not cached, so a transient
    /// compilation error does not poison the entry.
    pub fn get_or_compile<E>(
        &mut self,
        key: KernelKey,
        source: &KernelSource,
        compile: impl FnOnce(&KernelSource) -> Result<T, E>,
    ) -> Result<&T, E> {
        match self.entries.entry(key) {
            Entry::Occupied(occupied) => {
                self.hits += 1;
                Ok(&occupied.into_mut().kernel)
            }
            Entry::Vacant(vacant) => {
                self.misses += 1;
                let kernel = compile(source)?;
                Ok(&vacant.insert(Compiled { kernel, source: source.clone() }).kernel)
            }
        }
    }

    /// A kernel already compiled, if there is one.
    pub fn get(&self, key: KernelKey) -> Option<&T> {
        self.entries.get(&key).map(|entry| &entry.kernel)
    }

    /// The source a key was compiled from (§24.1: inspectable).
    pub fn source_of(&self, key: KernelKey) -> Option<&KernelSource> {
        self.entries.get(&key).map(|entry| &entry.source)
    }

    /// Every key held, sorted, so a report over them is order-stable.
    pub fn keys(&self) -> Vec<KernelKey> {
        let mut keys: Vec<_> = self.entries.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    /// How many lookups found an existing kernel.
    pub fn hits(&self) -> usize {
        self.hits
    }

    /// How many lookups had to compile.
    pub fn misses(&self) -> usize {
        self.misses
    }

    /// How many kernels are held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if nothing has been compiled.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<T> Default for KernelCache<T> {
    fn default() -> KernelCache<T> {
        KernelCache::new()
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a, 64-bit, continued from an existing state.
///
/// The same function `lattice_observe::fnv1a_64` uses for the run artifact's content hash,
/// reimplemented rather than shared because this crate sits below `lattice-ir` and taking
/// a dependency upward to save eight lines would invert the workspace. The published test
/// vectors are asserted in both places, which is what keeps them the same function.
fn fnv1a_64_from(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> Capabilities {
        Capabilities {
            adapter: "test-adapter".to_string(),
            precisions: vec![Precision::Fast32],
            max_workgroup: Some(256),
            max_buffer_bytes: 1 << 28,
        }
    }

    fn key_for(source: &KernelSource) -> KernelKey {
        KernelKey::new(source, Backend::Wgpu, Precision::Fast32, &caps())
    }

    #[test]
    fn fnv1a_matches_its_published_vectors() {
        assert_eq!(fnv1a_64_from(FNV_OFFSET, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64_from(FNV_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64_from(FNV_OFFSET, b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn formatting_alone_does_not_produce_a_second_entry() {
        let plain = KernelSource::new("stencil", "let a = b + c;");
        let commented = KernelSource::new(
            "stencil",
            "  let a = b + c;   // the east face\n/* and a block */\n",
        );
        assert_eq!(plain.normalized(), commented.normalized());
        assert_eq!(key_for(&plain), key_for(&commented));
    }

    /// Deleting a comment between two identifiers must not weld them together.
    #[test]
    fn a_comment_leaves_a_separator_behind() {
        let source = KernelSource::new("k", "let a/*x*/b = 1;");
        assert_eq!(source.normalized(), "let a b = 1;");
    }

    #[test]
    fn a_different_body_is_a_different_key() {
        let a = KernelSource::new("stencil", "let a = b + c;");
        let b = KernelSource::new("stencil", "let a = b - c;");
        assert_ne!(key_for(&a), key_for(&b));
    }

    /// All four of §15.5's inputs, each varied on its own.
    #[test]
    fn every_one_of_the_four_inputs_changes_the_key() {
        let source = KernelSource::new("stencil", "let a = b + c;");
        let base = KernelKey::new(&source, Backend::Wgpu, Precision::Fast32, &caps());

        let other_source = KernelSource::new("stencil", "let a = b * c;");
        assert_ne!(base, KernelKey::new(&other_source, Backend::Wgpu, Precision::Fast32, &caps()));

        assert_ne!(base, KernelKey::new(&source, Backend::CpuScalar, Precision::Fast32, &caps()));
        assert_ne!(base, KernelKey::new(&source, Backend::Wgpu, Precision::Accurate64, &caps()));

        let mut other_caps = caps();
        other_caps.max_workgroup = Some(64);
        assert_ne!(base, KernelKey::new(&source, Backend::Wgpu, Precision::Fast32, &other_caps));
    }

    #[test]
    fn specializations_are_part_of_the_key_and_order_independent() {
        let source = KernelSource::new("stencil", "body");
        let a = source.clone().specialize("nx", 512).specialize("halo", 1);
        let b = source.clone().specialize("halo", 1).specialize("nx", 512);
        let c = source.clone().specialize("nx", 256).specialize("halo", 1);

        assert_eq!(key_for(&a), key_for(&b), "order must not matter");
        assert_ne!(key_for(&a), key_for(&c), "the value must");
        assert_ne!(key_for(&a), key_for(&source), "specializing at all must");
    }

    /// The field separator earns its place here.
    #[test]
    fn adjacent_fields_cannot_be_confused_with_each_other() {
        let a = KernelSource::new("ab", "c");
        let b = KernelSource::new("a", "bc");
        assert_ne!(key_for(&a), key_for(&b));
    }

    #[test]
    fn a_kernel_is_compiled_once_and_then_reused() {
        let mut cache: KernelCache<String> = KernelCache::new();
        let source = KernelSource::new("stencil", "let a = b + c;");
        let key = key_for(&source);
        let mut compiles = 0;

        for _ in 0..3 {
            let compiled = cache
                .get_or_compile(key, &source, |source| {
                    compiles += 1;
                    Ok::<_, ()>(source.normalized())
                })
                .unwrap();
            assert_eq!(compiled, "let a = b + c;");
        }

        assert_eq!(compiles, 1, "compiled more than once");
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.len(), 1);
    }

    /// A transient failure must not be remembered as an answer.
    #[test]
    fn a_failed_compilation_is_not_cached() {
        let mut cache: KernelCache<String> = KernelCache::new();
        let source = KernelSource::new("stencil", "bad");
        let key = key_for(&source);

        assert!(cache.get_or_compile(key, &source, |_| Err::<String, _>("boom")).is_err());
        assert!(cache.is_empty());
        assert!(cache.get_or_compile(key, &source, |_| Ok::<_, ()>("fine".to_string())).is_ok());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn the_source_stays_readable_after_compilation() {
        let mut cache: KernelCache<()> = KernelCache::new();
        let source = KernelSource::new("stencil", "let a = b + c; // east").specialize("nx", 512);
        let key = key_for(&source);
        cache.get_or_compile(key, &source, |_| Ok::<_, ()>(())).unwrap();

        let held = cache.source_of(key).expect("source retained");
        assert_eq!(held.body(), "let a = b + c; // east");
        assert_eq!(held.specializations(), [("nx".to_string(), "512".to_string())]);
        assert_eq!(cache.keys(), vec![key]);
    }
}
