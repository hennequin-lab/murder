# Design

Murder holds stable libraries built on Raven. Applications and ongoing projects
live in their own repositories.

## Principles

- Small packages with one job and a documented `.mli`.
- Explicit shapes, dtypes, and errors; no silent promotion or broadcast.
- Pure functions over values where possible; state is a value passed and
  returned.
- No `utils` package. Shared code gets a domain name when it has a second user.
- User-facing error messages name the operation, the inputs, and the problem.
- Public docs explain behaviour from first principles, without comparisons to
  other numerical libraries.

## Promise

A package is promoted here when it has an owner, a README, tests, a documented
API, and a reason to be shared. Before that, it belongs in its own repository.
