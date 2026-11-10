//! Observer laws at their use site: `observe sum(kinetic(atoms)) every 0.1 picosecond;`
//! (spec M6.2 §4.2).
//!
//! The law was checked where it was declared. Here the call is taken apart, its
//! parameters are bound by the same code a `force:` setting uses
//! ([`bind_arguments`](crate::user_force::bind_arguments)), its program is lowered
//! (`E0413` if it does not fit the register file), and what a run needs to record it is
//! gathered into an [`ObserverLawUse`]: the program, the values, the particle domain, the
//! reduction, the unit and the row.

use lattice_ir::{ObserverLawUse, Reduction};
use lattice_syntax::{Diagnostics, Expr, ExprKind};
use lattice_units::Dimension;

use crate::eval::Evaluator;
use crate::laws::{LawKind, Ty, TypedLaw};
use crate::lower_law::lower;
use crate::user_force::{bind_arguments, BindOptions};

/// How a row's unit reads: `J` when the law declared an energy unit by name (`joule`,
/// `J`), and the coherent SI form otherwise. A dimension alone cannot say `J`: a torque
/// declared in `newton * meter` has the dimension of an energy and is not one.
pub(crate) fn unit_label(dimension: Dimension, declared: Option<&str>) -> String {
    if dimension == Dimension::ENERGY && matches!(declared, Some("joule" | "J")) {
        "J".to_string()
    } else {
        dimension.to_string()
    }
}

/// Bind and lower the observer law `law` called as `call` (`kinetic(atoms, x0 = 2 meter)`)
/// over the particle domain `domain`, reduced by `reduction`.
///
/// `call`'s first argument is the set, already resolved by the caller; the rest are the
/// law's `param`s by name. Reports and returns `None` on any problem.
pub(crate) fn lower_use(
    law: &TypedLaw,
    domain: &str,
    reduction: Reduction,
    call: &Expr,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<ObserverLawUse> {
    debug_assert_eq!(law.kind, LawKind::Observer);
    let arguments = match &call.kind {
        ExprKind::Call(_, arguments) => arguments.get(1..).unwrap_or(&[]),
        _ => &[],
    };
    let bound = bind_arguments(
        law,
        arguments,
        call.span,
        BindOptions { cutoff: false, truncation: false },
        evaluator,
        diagnostics,
    );
    let lowered = match lower(law) {
        Ok(lowered) => Some(lowered),
        Err(diagnostic) => {
            diagnostics.push(*diagnostic);
            None
        }
    };
    let (bound, lowered) = (bound?, lowered?);
    let Ty::Scalar(dimension) = law.result.ty else {
        unreachable!("the checker gives an observer law a scalar result (E0417)")
    };
    Some(ObserverLawUse {
        law: law.name.clone(),
        domain: domain.to_string(),
        reduction,
        program: lowered.program,
        params: bound.params,
        unit: unit_label(dimension, law.declared_unit.as_deref()),
        row: format!("{domain}.{}.{}", law.name, reduction.name()),
    })
}
