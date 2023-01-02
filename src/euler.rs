//! Explicit euler method with fixed step size.

use crate::dop_shared::{IntegrationError, Stats, System};

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, Scalar};
use num_traits::Zero;
use simba::scalar::{ClosedAdd, ClosedMul, ClosedNeg, ClosedSub, SubsetOf};

/// Structure containing the parameters for the numerical integration.
pub struct Euler<V, F>
where
    F: System<V>,
{
    f: F,
    x: f64,
    y: V,
    x_end: f64,
    step_size: Vec<f64>,
    x_out: Vec<f64>,
    y_out: Vec<V>,
    stats: Stats,
}

impl<T, D: Dim, F> Euler<OVector<T, D>, F>
where
    f64: From<T>,
    T: Copy + SubsetOf<f64> + Scalar + ClosedAdd + ClosedMul + ClosedSub + ClosedNeg + Zero,
    F: System<OVector<T, D>>,
    OVector<T, D>: std::ops::Mul<f64, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<T, D>,
{
    /// Default initializer for the structure
    ///
    /// # Arguments
    ///
    /// * `f`           - Structure implementing the System<V> trait
    /// * `x`           - Initial value of the independent variable (usually time)
    /// * `y`           - Initial value of the dependent variable(s)
    /// * `x_end`       - Final value of the independent variable
    /// * `step_size`   - Step size(s) used in the method
    ///
    pub fn new(f:F, x: f64, y: OVector<T, D>, x_end: f64, step_size: Vec<f64>) -> Self {
        Euler {
            f,
            x,
            y,
            x_end,
            step_size,
            x_out: Vec::new(),
            y_out: Vec::new(),
            stats: Stats::new(),
        }
    }

    /// Core integration method.
    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        // Save initial values
        self.x_out.push(self.x);
        self.y_out.push(self.y.clone());
        // Call Observer 
        self.f.observer(self.x, &self.y);
        
        let num_steps = ((self.x_end - self.x)/ self.step_size[2]).ceil() as usize;
        let num_steps_per_obs = (self.step_size[2]/ self.step_size[1]).ceil() as usize;
        let num_steps_per_event = (self.step_size[1] / self.step_size[0]).ceil() as usize;

        for _ in 0..num_steps {
          for _ in 0..num_steps_per_obs {
            let y_new = self.e_step();
            self.y = y_new;
            for _ in 0..num_steps_per_event {
              let (x_new, y_new) = self.step();
              self.x = x_new;
              self.y = y_new;
              self.stats.num_eval += 1;
              self.stats.accepted_steps += 1;
            }
          }
          // Call Observer 
          self.f.observer(self.x, &self.y);
        }
        // final state
        self.x_out.push(self.x);
        self.y_out.push(self.y.clone());
        Ok(self.stats)
    }

    /// Performs one step of the forward euler method.
    fn step(&self) -> (f64, OVector<T, D>) {
        let (rows, cols) = self.y.shape_generic();
        let mut k = vec![OVector::zeros_generic(rows, cols); 3];

        self.f.ode(self.x, &self.y, &mut k[0]);
        let x_new = self.x + self.step_size[0];
        let y_new = &self.y
            + (k[0].clone())
                * (self.step_size[0]);
        (x_new, y_new)
    }

  fn e_step(&mut self) -> OVector<T, D> {
        // note: does not advance time (happens instantaneously)
        let (rows, cols) = self.y.shape_generic();
        let mut k = vec![OVector::zeros_generic(rows, cols); 3]; //dy
        self.f.event(self.x, &self.y, &mut k[0]);
        let y_new = &self.y
            + k[0].clone();
        y_new
    }

    /// Getter for the independent variable's output.
    pub fn x_out(&self) -> &Vec<f64> {
        &self.x_out
    }

    /// Getter for the dependent variables' output.
    pub fn y_out(&self) -> &Vec<OVector<T, D>> {
        &self.y_out
    }
}

