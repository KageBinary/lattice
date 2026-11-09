//! Observer laws at their use site: `observe sum(kinetic(atoms)) every 0.1 picosecond;`
//! (spec M6.2 §4.2).
//!
//! The law was checked where it was declared. Here the call is taken apart, its
//! parameters are bound by the same code a `force:` setting uses
//! ([`bind_arguments`](crate::user_force::bind_arguments)), its program is lowered
//! (`E0413` if it does not fit the register file), and what a run needs to record it is
//! gathered into an [`ObserverLawUse`]: the program, the values, the particle domain, the
//! reduction, the unit and the row.

use lattice_domain_particle::LawObserver;
use lattice_ir::{ObserverLawUse, Reduction};
use lattice_syntax::{Diagnostic, Diagnostics, Expr, ExprKind, SourceFile};
use lattice_units::{Dimension, UnitRegistry};

use crate::eval::Evaluator;
use crate::laws::{LawKind, Ty, TypedLaw};
use crate::lower_law::lower;
use crate::user_force::{bind_arguments, BindOptions};

/// How a row's unit reads: `J` for an energy, the coherent SI form otherwise.
pub(crate) fn unit_label(dimension: Dimension) -> String {
    if dimension == Dimension::ENERGY {
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
        unit: unit_label(dimension),
        row: format!("{domain}.{}.{}", law.name, reduction.name()),
    })
}

/// Compile one observer law and bind it as the use site `use_site` would, outside any
/// model: for validation and tests that want a law observer without a scene around it.
///
/// `law_source` is the law's declaration; `use_site` is what follows `observe`, such as
/// `sum(kinetic(cloud))` or `max(height(cloud, floor = 1 meter))`. The set is not looked
/// up: the observer names it, and attaching it to a simulation audits that it exists.
/// Returns the rendered diagnostics when anything fails.
pub fn observer_from_source(
    law_source: &str,
    use_site: &str,
    interval: Option<f64>,
) -> Result<LawObserver, String> {
    let text = format!("project law {{ {law_source} observe {use_site}; }}");
    let file = SourceFile::new("law.lattice", text);
    let (project, mut diagnostics) = lattice_syntax::parse(&file);
    let Some(project) = project.filter(|_| !diagnostics.has_errors()) else {
        return Err(diagnostics.render(&file));
    };
    let units = UnitRegistry::si();
    let evaluator = Evaluator::new(&file, &units);
    let laws = crate::laws::check_laws(&project, &evaluator, &mut diagnostics);
    let shaped = project.observes().next().and_then(|statement| {
        let ExprKind::Call(reduction, arguments) = &statement.target.kind else { return None };
        let [reduced] = arguments.as_slice() else { return None };
        let ExprKind::Call(callee, inner) = &reduced.value.kind else { return None };
        let (law, set) = (callee.as_name()?, inner.first()?.value.as_name()?);
        Some((Reduction::parse(reduction.as_name()?)?, laws.iter().find(|l| l.name == law)?, set, &reduced.value))
    });
    let Some((reduction, law, set, call)) = shaped else {
        diagnostics.push(
            Diagnostic::error("expected `reduction(law(set, …))`").with_code("E0419").span(project.name.span),
        );
        return Err(diagnostics.render(&file));
    };
    match lower_use(law, set, reduction, call, &evaluator, &mut diagnostics) {
        Some(law) if !diagnostics.has_errors() => Ok(LawObserver::new(&law, interval)),
        _ => Err(diagnostics.render(&file)),
    }
}
