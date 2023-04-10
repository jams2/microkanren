# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
