---
author: Optuna team
title: Unknown-Constraint C-DTLZ
description: Unknown-constraint variant of C-DTLZ where constraints are hidden and infeasible points return NaN objective values.
tags: [benchmark, constrained, multi-objective, unknown constraints, C-DTLZ]
optuna_versions: [4.5.0]
license: MIT License
---

## Overview

This benchmark mirrors `dtlz_constrained`, but does not expose constraint values.
It evaluates C-DTLZ constraints internally and returns `NaN` objectives for infeasible points.

## APIs

### class `Problem(function_id: int, n_objectives: int, constraint_type: int, dimension: int | None = None)`

- Same arguments/combinations as `dtlz_constrained`.
- `evaluate(params)` returns objective values for feasible points, otherwise `NaN` values.

Constraint values are intentionally hidden.
