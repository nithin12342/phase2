//! h5-schema: the collision-free 200-feature contract.
//!
//! Every emitted feature carries a reliability flag. An absent modality emits
//! NaN + `Reliability::Absent`, never a silent zero.
pub mod catalog;
pub mod feature;
pub mod reliability;

pub use feature::{FeatureId, FeatureValue};
pub use reliability::Reliability;

/// The seven modality groups in the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Audio,
    Text,
    Video,
    Face,
    Tabular,
    CrossModal,
    Quality,
}

/// Whether a feature is a direct primitive or a fusion-derived interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Primitive,
    Fusion,
}