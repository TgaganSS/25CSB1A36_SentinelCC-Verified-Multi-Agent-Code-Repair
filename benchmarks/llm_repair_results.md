# LLM repair evaluation (none)

Programs failing to build at the start: **29** (blocked and skipped: 1).

Repaired by rules only: **21/29**. Repaired with rules + LLM: **21/29** (extra: 0).

| Program | Errors at start | Rules only | Rules + LLM | Proposals accepted | Rejected (reason) |
|---|---|---|---|---|---|
| break_outside_loop.cpp | 1 | built | built | n/a | n/a (no model) |
| const_assignment.cpp | 1 | built | built | n/a | n/a (no model) |
| continue_outside_loop.cpp | 1 | built | built | n/a | n/a (no model) |
| invalid_array_index.cpp | 1 | built | built | n/a | n/a (no model) |
| invalid_operands.cpp | 1 | fails | fails | n/a | n/a (no model) |
| invalid_return_type.cpp | 1 | fails | fails | n/a | n/a (no model) |
| invalid_use_of_void.cpp | 1 | built | built | n/a | n/a (no model) |
| mini/tc01_missing_semicolon.cpp | 2 | built | built | n/a | n/a (no model) |
| mini/tc02_undeclared_variable.cpp | 1 | built | built | n/a | n/a (no model) |
| mini/tc03_missing_include.cpp | 3 | built | built | n/a | n/a (no model) |
| mini/tc05_missing_closing_brace.cpp | 1 | built | built | n/a | n/a (no model) |
| mini/tc06_wrong_stream_operator.cpp | 2 | built | built | n/a | n/a (no model) |
| mini/tc07_keyword_typo.cpp | 1 | built | built | n/a | n/a (no model) |
| mini/tc10_buffer_overflow_gets.cpp | 1 | built | built | n/a | n/a (no model) |
| mini/tc13_use_after_free.cpp | 1 | fails | fails | n/a | n/a (no model) |
| mini/tc14_format_string_vuln.cpp | 1 | built | built | n/a | n/a (no model) |
| mini/tc15_combined_errors_and_threats.cpp | 2 | fails | fails | n/a | n/a (no model) |
| mini/tc16_redefinition_double_free.cpp | 1 | built | built | n/a | n/a (no model) |
| missing_brace.cpp | 1 | built | built | n/a | n/a (no model) |
| missing_header.cpp | 2 | built | built | n/a | n/a (no model) |
| multiple_definition.cpp | 1 | built | built | n/a | n/a (no model) |
| no_matching_function.cpp | 1 | fails | fails | n/a | n/a (no model) |
| pointer_misuse.cpp | 1 | built | built | n/a | n/a (no model) |
| redefination.cpp | 1 | built | built | n/a | n/a (no model) |
| sizeof_function.cpp | 1 | fails | fails | n/a | n/a (no model) |
| syntax_error.cpp | 1 | built | built | n/a | n/a (no model) |
| type_error.cpp | 1 | fails | fails | n/a | n/a (no model) |
| undeclared_var.cpp | 1 | fails | fails | n/a | n/a (no model) |
| wrong_main_signature.cpp | 1 | built | built | n/a | n/a (no model) |
