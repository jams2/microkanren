# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.4.3] - 2023-10-20

### Fixed
- CI publish workflow

## [0.4.2] - 2023-10-20

### Fixed
- Order of execution of goals inside a disj

### Added
- CI workflows using GitHub actions

## [0.4.1] - 2023-10-05

### Changed
- Added Python 3.12 support
- Bumped `fastcons` dependency

## [0.4.0] - 2023-04-23

### Added
- `ltfd` goal

### Changed
- Moved finite domain goal constructors into fd.py
- Exit early from FD goals if any var has no domain
- When exiting early from FD goals, make sure a constraint is added to the store
- Use `immutables` map instead of pyrsistent map (better performance)
- Change `neq` signature from `neq((a, b), *rest_pairs)` to `neq(a, b, /, *rest)`
- Make `Constraint` frozen so it's hashable
- Store `Constraint` operands as tuples rather than lists/sets

## [0.3.0] - 2023-04-10

### Changed
- Use `Hooks` registry class instead of globals for hooks
- `reify_constraints` hook is called
- `reify_var` hook is used instead of unconditionally creating `ReifiedVar`
- Application of `conj` and `disj` to multiple goals is now left-associative to match `|` and `&` operators

### Added
- `reify_value` hook
- Basic usage info to README.md

### Removed
- Unused `StreamThunk` type

## [0.2.2] - 2023-04-08

### Fixed
- Bad requires-python specification
- `irun` providing continuation argument to goal

## [0.2.0] - 2023-04-07

### Changed

- Use `fastcons` extension module for `cons` and `nil`
- Don't call `to_python` on results in reifiers
