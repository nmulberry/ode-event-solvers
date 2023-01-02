//! # ODEs Solvers
//! `ode-solvers` is a collection of numerical methods to solve ordinary differential equations (ODEs).

// Re-export from external crate
pub use crate::na::{
    DVector, OVector, SVector, Vector1, Vector2, Vector3, Vector4, Vector5, Vector6,
};
use nalgebra as na;

// Declare modules
pub mod dop_shared;
pub mod euler;
pub use euler::Euler;
pub use dop_shared::System;
