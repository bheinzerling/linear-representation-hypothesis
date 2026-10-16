# LRH usage codebook v0.1

## Purpose

This codebook characterizes how an episode uses the phrase “linear
representation hypothesis.” It records the proposition attached to the label,
the role that proposition plays in the local argument, and the evidence used
to support or challenge it. It does not ask coders to decide whether the LRH is
true.

Version 0.1 consolidates two independent open-coding passes over 60 episodes.
It is a preliminary codebook for a structured second pilot. Categories may
still change after that pilot.

## Unit and coding order

The unit is a usage episode: one or more nearby exact-phrase occurrences whose
paragraph-level context windows overlap. Code the episode in this order:

1. Determine whether the episode contains a substantive use.
2. Summarize the proposition that the passage attaches to the LRH.
3. Record the passage's rhetorical function and the author's commitment.
4. Identify the representational object, scope, and empirical or formal
   operationalization.
5. Record any causal claim and the function of local citations.
6. Assign confidence and state whether more context is required for
   classification or only desirable for verification.

Use multiple values when an episode performs several functions or invokes
several forms of evidence. Apply values only when the passage provides
positive textual evidence. Do not infer an unmentioned claim from the paper's
title, venue, method family, or citation count.

## 1. Episode status

Assign exactly one value.

| Value | Definition |
| --- | --- |
| `substantive` | The episode states, attributes, assumes, tests, qualifies, or challenges a proposition associated with the LRH. |
| `bibliography_only` | The phrase appears only in a reference entry. |
| `structural_fragment` | The phrase appears only in a heading, running label, figure fragment, or similar document structure. |
| `context_insufficient` | The visible text may be substantive, but the proposition cannot be classified from the available context. |

Bibliography-only and structural-fragment episodes remain in the provenance
record but are excluded from analyses of author use. A short sentence that
merely names a cited work can still be substantive when it attributes a claim
or locates the present work relative to that source.

A phrase occurrence in a heading remains substantive when the surrounding
episode states a recoverable LRH proposition. Assign `structural_fragment`
only when the heading, label, or fragment and its surrounding text provide no
such proposition.

## 2. Rhetorical function

Assign one or more values.

| Value | Apply when the episode… |
| --- | --- |
| `definition_or_formalization` | states what the LRH means or gives a mathematical formulation. |
| `background_or_attribution` | presents the LRH as prior knowledge, a literature claim, or a historical attribution. |
| `problem_setting` | uses ambiguity, disputed scope, or a missing explanation to motivate a research question. |
| `theoretical_premise` | uses the LRH as an assumption in an argument, theorem, or model. |
| `theoretical_explanation` | explains why a form of linear structure should arise or unifies several reported phenomena. |
| `method_motivation` | uses the LRH to justify the design of a probe, dictionary, intervention, predictor, or other method. |
| `empirical_test` | reports an observation or experiment intended to bear on an LRH claim. |
| `evidential_claim` | interprets a result or prior study as support or counterevidence. |
| `application_or_interpretation` | uses an LRH claim to interpret or enable a downstream application without locally testing the claim. |
| `qualification_or_refinement` | narrows the scope, changes the mathematical object, or distinguishes weak and strong formulations. |
| `extension_or_repair` | retains part of an LRH formulation while replacing or expanding another part. |
| `challenge_or_counterevidence` | presents a conceptual objection, counterexample, or adverse empirical result. |
| `literature_positioning` | contrasts or organizes prior LRH work without adding a local test. |

`method_motivation` concerns why a procedure is chosen.
`empirical_test` concerns an outcome interpreted as evidence about the LRH.
Assign both when a passage uses the hypothesis to motivate a method and then
interprets the method's result as evidence for the same hypothesis.

## 3. Author commitment

Assign one primary value. Record a target component when the stance applies to
only part of the LRH.

| Value | Definition |
| --- | --- |
| `endorses` | The authors assert the local LRH proposition as their own claim. |
| `adopts_as_assumption` | The authors use the proposition provisionally as a premise or design assumption. |
| `reports_support` | The authors interpret a result as evidence for the proposition. |
| `neutral_attribution` | The authors report another source's claim without a clear local endorsement or challenge. |
| `qualifies` | The authors accept or preserve a restricted component while limiting its scope or strength. |
| `challenges` | The authors reject, problematize, or report evidence against the target component. |
| `mixed` | The episode supports one component and challenges another, with neither relation adequately represented as a qualification alone. |
| `unclear` | The available context does not establish the authors' commitment. |

Do not treat conventional present-tense exposition as endorsement when the
sentence is explicitly attributed to another source. Do not code a passage as
challenging the entire LRH when it rejects only fixed directions, exhaustive
coverage, or another specified component.

## 4. Representational object

Assign all explicit values. Use `unspecified` only when no more precise value
is supported.

| Value | Definition |
| --- | --- |
| `fixed_direction_or_offset` | A feature or contrast corresponds to a stable vector direction or counterfactual offset. |
| `feature_subspace` | A feature or concept occupies a multidimensional linear subspace. |
| `manifold_within_subspace` | Feature values trace a curved or topologically structured set that may occupy a linear subspace. |
| `function_class` | A specified target or function class can be recovered by a bounded linear map. |
| `sparse_dictionary` | Activations decompose into combinations of elements from a shared feature dictionary. |
| `latent_coordinate_transform` | Learned representations recover latent variables up to an invertible or otherwise specified linear transform. |
| `weight_or_task_matrix` | The episode applies an LRH claim to weight matrices, task vectors, or model merging. |
| `curved_or_topological_structure` | A manifold, curved set, topology, or other nonlinear object is offered as an extension or alternative. |
| `unspecified` | The phrase is used without a recoverable mathematical object. |

A useful fixed direction does not entail an exhaustive sparse dictionary.
Linear probe success establishes accessibility under the stated task; it does
not by itself establish a fixed direction, additive composition, or causal
use. A curved feature set may lie within a linear subspace, so
`curved_or_topological_structure` and `feature_subspace` can co-occur.

## 5. Linear property

Assign all explicit values.

| Value | Apply when the proposition concerns… |
| --- | --- |
| `linear_accessibility` | prediction of a target by a linear probe or function. |
| `contrastive_alignment` | stable alignment among paired or counterfactual difference vectors. |
| `additive_composition` | summation of feature contributions in a representation. |
| `additive_intervention` | addition or subtraction of a vector to change a feature or behavior. |
| `projection_geometry` | semantic structure expressed by projection, inner product, orthogonality, cosine, or related geometry. |
| `sparse_reconstruction` | activation reconstruction from a small set of dictionary elements. |
| `linear_identifiability` | a linear or affine map between learned and latent coordinates. |
| `unspecified` | no particular linear property can be recovered. |

Record the object and property separately. A feature subspace is an object;
projection geometry or additive intervention is a property attributed to it.

## 6. Scope and strength

Code five axes independently.

| Axis | Values |
| --- | --- |
| Coverage | `some_features`, `specified_feature_class`, `broad_typicality`, `all_features`, `unstated` |
| Dimensionality | `one_dimensional`, `multidimensional`, `mixed`, `unstated` |
| Variation | `fixed_across_inputs`, `context_dependent`, `unstated` |
| Precision | `exact`, `approximate_or_noisy`, `mixed`, `unstated` |
| Decomposition strength | `local_feature`, `exhaustive_decomposition`, `unstated` |

Add any explicit modifiers from `sparse`, `nonnegative`, `overcomplete`,
`orthogonal`, and `bounded_norm`. Do not infer a modifier from the use of an
SAE, probe, or direction.

These axes prevent a local, approximate result for one feature from being
treated as support for an exact decomposition of every activation. An exact
statement can concern a single feature, and an exhaustive statement can permit
approximation.

Assign `all_features` only for explicit universal coverage or a complete
decomposition. Generic plural descriptions of “features” receive
`broad_typicality` unless the episode supplies a stronger quantifier. Assign
dimensionality and input variation only when the episode states them; the word
“direction” alone does not establish that a direction is fixed across inputs.

Assign `one_dimensional` when the passage specifies a single direction, axis,
line, or vector as the feature object. Assign `multidimensional` when it states
that a feature requires or occupies more than one dimension. Use `mixed` for
an explicit comparison or formulation covering both. “Low-dimensional” and
the unqualified word “subspace” leave dimensionality `unstated`.

Assign `exhaustive_decomposition` when the proposition covers complete
activation reconstruction. Use `local_feature` for one feature or a specified
feature class. When decomposition lies outside the local proposition, assign
`unstated`.

## 7. Evidence or operationalization

Assign all values that the episode uses to support, test, or instantiate its
LRH proposition.

| Value | Definition |
| --- | --- |
| `linear_probe` | Linear classification, regression, or another explicit probing test. |
| `counterfactual_difference_or_alignment` | Difference vectors, paired contrasts, directional alignment, or comparison with random offsets. |
| `intervention_or_steering` | Activation editing, steering, ablation, patching, or control through a direction or subspace. |
| `sparse_reconstruction_or_sae` | Sparse coding, dictionary recovery, reconstruction, or sparse-autoencoder evidence. |
| `geometry_or_topology` | Geometric residuals, curvature, dimensionality, topology, clustering, or manifold structure. |
| `formal_result_or_derivation` | A theorem, proof, identifiability result, optimization argument, or mathematical derivation. |
| `downstream_task_performance` | Prediction, distillation, defense, detoxification, or another application-level outcome. |
| `weight_or_model_manipulation` | Model merging, task arithmetic, or manipulation of weight-space directions or matrices. |
| `literature_assertion` | Prior literature is invoked without a local empirical or formal test in the episode. |
| `proposed_only` | A conjecture, planned test, or method is described without a result. |
| `none_stated` | No evidence or operationalization is stated. |

Method performance can be compatible with an LRH proposition without
isolating it. Record one evidence relation:

| Evidence relation | Definition |
| --- | --- |
| `direct_test` | The procedure is designed to distinguish the stated LRH proposition from a relevant alternative. |
| `compatible_but_nonisolating` | The result is consistent with the proposition, but other assumptions or mechanisms could produce it. |
| `premise_only` | The proposition motivates a method or argument and is not tested locally. |
| `unclear` | The relation between the procedure and proposition cannot be recovered. |
| `none` | The episode supplies no evidence or operationalization. |

A passage receives `direct_test` when it states a discriminating prediction
and reports the corresponding result. A proposal, method description, or
conditional prediction without a reported result receives `premise_only` or
`unclear`, as appropriate.

## 8. Causal implication and cross-level transfer

Assign one primary value.

| Value | Definition |
| --- | --- |
| `no_causal_claim` | The episode makes no claim about intervention or the model's causal use of a feature. |
| `causal_effect_asserted` | The episode claims that changing or using the feature affects behavior without reporting a local test. |
| `causal_effect_tested` | The episode reports an intervention or mediation test intended to bear on causal effect. |
| `conflates_accessibility_and_use` | The episode treats probe accessibility as sufficient evidence of causal use. |
| `unclear` | The causal implication cannot be determined. |

Also record whether selectivity was tested. Successful intervention supports
control through the intervention used. Evidence that the unmodified model
normally uses the same coordinate requires a separate argument.

Assign `causal_effect_asserted` only when the episode explicitly connects a
feature or intervention to model computation or behavior. A method described
in causal terminology receives `no_causal_claim` when the local LRH
proposition concerns only accessibility or geometry.

Assign a cross-level value:

| Value | Definition |
| --- | --- |
| `same_level` | The proposition remains about representation-space structure or operations. |
| `cross_level_transfer` | The episode extends a representation claim to weights, philosophical understanding, a value-distribution model, a high-level causal model, or another level requiring an additional argument. |
| `unclear` | The source and target levels cannot be recovered. |

When assigning `cross_level_transfer`, name the source and target levels in
the analytic memo.

## 9. Citation function

Assign all applicable values.

| Value | Definition |
| --- | --- |
| `foundational_attribution` | A source is credited with introducing or defining the LRH. |
| `supporting_evidence` | A citation is used as evidence for an LRH proposition. |
| `method_ancestry` | A citation supplies the method or operationalization used in the episode. |
| `critical_target` | A cited claim or source is qualified or challenged. |
| `neutral_reference` | A citation locates related work without another recoverable function. |
| `unresolved_numeric_citation` | A numeric marker is present, but its source or function cannot be resolved from the packet. |
| `none` | No citation functions locally in the episode. |

Record identifiable source names or citation markers separately. Do not use
external metadata to infer a citation's function during blinded coding.

## 10. Context priority and confidence

Assign one context-priority value and one confidence value.

| Context priority | Definition |
| --- | --- |
| `none` | The supplied passage is sufficient for classification and verification. |
| `desirable_for_verification` | The classification is supported, while additional definitions, methods, results, or bibliography mappings would strengthen verification. |
| `required_for_classification` | A central code cannot be assigned without additional passage context. |

| Confidence | Definition |
| --- | --- |
| `high` | The central classification is explicit in the passage. |
| `medium` | The classification is supported but requires a modest inference or leaves one consequential boundary unresolved. |
| `low` | The central classification depends on substantial inference or incomplete text. |

State the exact missing information whenever context priority is not `none`.
Requests for an entire paper are too broad; request the definition, theorem
assumptions, figure caption, result continuation, citation mapping, or
ablation that would resolve the stated issue.

Routine interest in additional detail receives `none`.

## Boundary rules for the second pilot

- Code accessibility, geometry, and causal use independently.
- Attach qualifications and challenges to the component they concern.
- Preserve the difference between a working assumption and reported support.
- Separate existence claims from broad or exhaustive decomposition claims.
- Record cross-level moves from activation geometry to weight-space linearity.
- Distinguish direct tests from outcomes that are merely compatible with the
  proposition.
- Treat headings, figure fragments, and bibliography entries as provenance,
  not substantive uses.
- Prefer `unclear` or a context request over completing an argument from
  background knowledge.

## Evidence quotation

Provide one short quotation of at most 30 words that supports the central
classification. The quotation is an audit aid, not a substitute for the
structured fields. Preserve the source wording and do not combine
non-contiguous fragments as if they formed one sentence.
