---
author: Optuna team
title: Unknown-Constraint BBOB-Constrained
description: Unknown-constraint variant of bbob-constrained where constraint values are hidden from samplers and infeasible points are returned as NaN.
tags: [benchmark, continuous optimization, constrained optimization, unknown constraints, BBOB, COCO]
optuna_versions: [4.1.0]
license: MIT License
---

## Overview

This benchmark mirrors `bbob_constrained` but does not expose constraint values.
Infeasible evaluations are treated as failed evaluations by returning `NaN` from `evaluate`.

## APIs

### class `Problem(function_id: int, dimension: int, instance_id: int = 1)`

- `function_id`: Function ID in `[1, 54]`.
- `dimension`: Dimension in `[2, 3, 5, 10, 20, 40]`.
- `instance_id`: Instance ID in `[1, 15]`.

#### Methods

- `search_space` -> `dict[str, BaseDistribution]`
- `directions` -> `list[StudyDirection]`
- `evaluate(params)` -> `float`

Constraint values are intentionally hidden.
