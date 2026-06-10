# PriVTE Evidence-Label Alignment Audit

Records: 25

## Label Counts

| label | count |
|---|---:|
| mild_risk | 14 |
| moderate_risk | 6 |
| no_observed_risk | 5 |

## Numeric Feature Direction

| feature | direction | no_observed_risk mean | mild_risk mean | moderate_risk mean |
|---|---|---:|---:|---:|
| sampled_frame_count | mixed_or_non_monotonic | 177.0 | 151.5 | 177.3333 |
| device_visible_ratio | monotonic_decreasing_with_label | 0.6306 | 0.6234 | 0.4985 |
| hand_visible_ratio | monotonic_decreasing_with_label | 0.12 | 0.0497 | 0.0337 |
| hand_device_proximity_ratio | monotonic_decreasing_with_label | 0.0736 | 0.0254 | 0.0077 |
| stable_screen_engagement_ratio | mixed_or_non_monotonic | 0.192 | 0.246 | 0.2058 |
| active_hand_device_interaction_ratio | monotonic_decreasing_with_label | 0.167 | 0.1669 | 0.0855 |
| device_region_activity_ratio | mixed_or_non_monotonic | 0.1074 | 0.1473 | 0.0828 |
| repetitive_operation_count | mixed_or_non_monotonic | 2.8 | 0.0714 | 0.1667 |
| temporal_engagement_episode_count | mixed_or_non_monotonic | 18.4 | 17.3571 | 17.8333 |
| temporal_active_operation_point_count | mixed_or_non_monotonic | 4.6 | 0.5714 | 1.0 |
| temporal_direct_operation_point_count | mixed_or_non_monotonic | 2.8 | 0.0714 | 0.1667 |
| temporal_repetitive_operation_point_count | monotonic_decreasing_with_label | 1.4 | 0.0 | 0.0 |
| temporal_stable_screen_point_count | mixed_or_non_monotonic | 29.4 | 38.7857 | 35.5 |
| temporal_visible_without_engagement_point_count | monotonic_decreasing_with_label | 63.4 | 45.2857 | 40.6667 |
| temporal_confounded_activity_point_count | mixed_or_non_monotonic | 9.8 | 13.0 | 8.0 |
| trace_risk_pattern_score | insufficient_labels | NA | NA | NA |
| trace_stable_engagement_share | insufficient_labels | NA | NA | NA |
| trace_active_operation_share | insufficient_labels | NA | NA | NA |
| trace_direct_operation_share | insufficient_labels | NA | NA | NA |
| trace_repetitive_operation_share | insufficient_labels | NA | NA | NA |
| trace_visible_without_engagement_share | insufficient_labels | NA | NA | NA |
| trace_confounded_activity_share | insufficient_labels | NA | NA | NA |

## Discordant Samples

| sample_id | target_label | flags | trace_score | active | direct | repetitive | stable |
|---|---|---|---:|---:|---:|---:|---:|
| evidence_internal_p000004 | moderate_risk | risk_label_but_low_video_behavior_signal | None | 0.0 | 0.0 | 0.0 | 15.0 |
| evidence_internal_p000005 | no_observed_risk | no_label_but_active_or_repetitive_video_signal | None | 15.0 | 9.0 | 7.0 | 53.0 |
| evidence_internal_p000021 | no_observed_risk | no_label_but_active_or_repetitive_video_signal | None | 5.0 | 4.0 | 0.0 | 29.0 |
