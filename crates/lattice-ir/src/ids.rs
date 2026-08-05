//! Typed identifiers.
//!
//! Spec §7.1 calls for "typed IDs" in the IR. The point is not ceremony: a simulation
//! has a dozen parallel index spaces (domains, fields, species, particles, buffers)
//! and passing a bare `usize` between them is a bug that type-checks. Every ID here
//! is a distinct type, so mixing a `SpeciesId` into a field lookup is a compile error.
//!
//! # Two flavours of identity
//!
//! **Compile-time IDs** ([`DomainId`], [`FieldId`], [`SpeciesId`], …) name things
//! declared in the model. They are dense, assigned once during compilation, and never
//! recycled — so a plain `u32` index is enough.
//!
//! **Runtime IDs** ([`ParticleId`], [`BodyId`]) name entities that can be destroyed
//! and whose slots are then reused by a later spawn. They carry a generation counter:
//! a handle to a dead entity is detected rather than silently resolving to whatever now
//! occupies its slot. An emitter that creates and destroys millions of particles must
//! not leak index space, and must not let a stale handle read someone else's state.

use core::fmt;

/// Declares a dense, never-recycled compile-time identifier.
macro_rules! compile_time_id {
    ($(#[$meta:meta])* $name:ident, $label:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(u32);

        impl $name {
            /// Construct from a dense index assigned during model compilation.
            pub const fn from_index(index: u32) -> Self {
                Self(index)
            }

            /// The dense index, for use as an array subscript.
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            /// The raw representation.
            pub const fn raw(self) -> u32 {
                self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($label, "#{}"), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($label, "#{}"), self.0)
            }
        }
    };
}

compile_time_id!(
    /// A solver domain instance (`rigid2d`, `particles2d`, `grid2d`, …).
    DomainId, "domain"
);
compile_time_id!(
    /// A scalar or vector field declared on a grid.
    FieldId, "field"
);
compile_time_id!(
    /// A chemical species.
    SpeciesId, "species"
);
compile_time_id!(
    /// A reaction in a reaction network.
    ReactionId, "reaction"
);
compile_time_id!(
    /// A material definition.
    MaterialId, "material"
);
compile_time_id!(
    /// One node of the compiled operation graph.
    OperatorId, "op"
);
compile_time_id!(
    /// A typed coupling port published or consumed by a domain (spec §14.1).
    PortId, "port"
);
compile_time_id!(
    /// A read-only measurement, probe, or export.
    ObserverId, "observer"
);
compile_time_id!(
    /// A planned runtime buffer, assigned by the compiler's buffer plan.
    BufferId, "buffer"
);
compile_time_id!(
    /// A collider shape registered with a rigid-body domain.
    ///
    /// Compile-time rather than generational because shapes are *shared*: a stack of
    /// fifty identical crates registers one shape and fifty bodies referencing it, and
    /// nothing destroys a shape while a body still points at it.
    ShapeId, "shape"
);
compile_time_id!(
    /// A particle species/type index used to select material parameters.
    ///
    /// Distinct from [`SpeciesId`]: a chemical species is a substance participating
    /// in reactions, while a particle kind selects which pair potential and radius a
    /// particle uses. A model can have both, and they are not interchangeable.
    ParticleKind, "kind"
);

/// Declares a generation-checked handle to a runtime entity.
macro_rules! runtime_id {
    ($(#[$meta:meta])* $name:ident, $label:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name {
            index: u32,
            generation: u32,
        }

        impl $name {
            /// Construct a handle. Normally only the owning store does this.
            pub const fn new(index: u32, generation: u32) -> Self {
                Self { index, generation }
            }

            /// The *stable* index this handle refers to.
            ///
            /// This is not where the entity's data lives. Hot state is stored compactly
            /// and moves as entities are destroyed; the stable index is a fixed entry in
            /// the store's indirection table that tracks the current position.
            pub const fn index(self) -> u32 {
                self.index
            }

            /// The generation this handle was issued in.
            pub const fn generation(self) -> u32 {
                self.generation
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($label, "#{}v{}"), self.index, self.generation)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($label, "#{}"), self.index)
            }
        }
    };
}

runtime_id!(
    /// A handle to a runtime particle, valid only while that particle is alive.
    ParticleId, "particle"
);
runtime_id!(
    /// A handle to a rigid body, valid only while that body is alive.
    BodyId, "body"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_time_ids_round_trip() {
        let f = FieldId::from_index(7);
        assert_eq!(f.index(), 7);
        assert_eq!(f.raw(), 7);
        assert_eq!(f.to_string(), "field#7");
    }

    #[test]
    fn ids_are_pointer_free_and_compact() {
        assert_eq!(core::mem::size_of::<FieldId>(), 4);
        assert_eq!(core::mem::size_of::<ParticleId>(), 8);
        assert_eq!(core::mem::size_of::<BodyId>(), 8);
    }

    /// Two runtime handles from different stores must not be interchangeable, even
    /// though they have identical layout — the whole reason they are separate types.
    #[test]
    fn runtime_id_types_are_distinct_and_label_themselves() {
        let particle = ParticleId::new(3, 1);
        let body = BodyId::new(3, 1);
        assert_eq!(particle.index(), body.index());
        assert_eq!(format!("{particle:?}"), "particle#3v1");
        assert_eq!(format!("{body:?}"), "body#3v1");
        assert_eq!(body.to_string(), "body#3");
        // `particle == body` does not compile, which is the entire point.
    }

    /// Different ID types must not be interchangeable. This is a compile-time
    /// property; the test documents it and would fail to build if the newtypes were
    /// collapsed into aliases.
    #[test]
    fn id_types_are_distinct() {
        let field = FieldId::from_index(3);
        let species = SpeciesId::from_index(3);
        assert_eq!(field.index(), species.index());
        // `field == species` does not compile, which is the entire point.
    }

    #[test]
    fn particle_handles_carry_a_generation() {
        let a = ParticleId::new(5, 0);
        let b = ParticleId::new(5, 1);
        assert_ne!(a, b, "a reused slot must not compare equal across generations");
        assert_eq!(a.index(), b.index());
    }
}
