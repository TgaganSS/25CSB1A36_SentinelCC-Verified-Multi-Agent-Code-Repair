"""Deterministic repair rules added after the first LLM evaluation (each fix is re-verified by compiling)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from auto_healer import attempt_fix


def fx(src, msg, line, col=0, cat="other"):
    return attempt_fix(src, {"message": msg, "line": line, "column": col, "category": cat})


def test_missing_std_header_is_added_even_if_a_comment_mentions_it():
    src = "// needs #include <vector>\n#include <iostream>\nint main(){ std::vector<int> v; }\n"
    out = fx(src, "'vector' is not a member of 'std'", 3, cat="missing_include")
    assert out and "#include <vector>" in out.split("\n", 1)[1]


def test_cout_without_header():
    out = fx("using namespace std;\nint main(){ cout << 1; }\n", "'cout' was not declared in this scope", 2)
    assert out and "#include <iostream>" in out


def test_void_main_becomes_int_main():
    out = fx("#include <cstdio>\nvoid main() { }\n", "'::main' must return 'int'", 2)
    assert out and "int main(" in out


def test_void_main_with_bare_return_is_left_alone():
    assert fx("void main() { return; }\n", "'::main' must return 'int'", 1) is None


def test_stray_break_and_continue_removed():
    assert "break" not in fx("int main(){\n  break;\n  return 0;\n}\n", "break statement not within loop or switch", 2)
    assert "continue" not in fx("int main(){\n  continue;\n}\n", "continue statement not within a loop", 2)


def test_const_assignment_drops_const():
    out = fx("int main(){\n  const int x=5;\n  x=10;\n}\n", "assignment of read-only variable ‘x’", 3, cat="access_error")
    assert out and "const" not in out


def test_deref_of_int():
    out = fx("int main(){ int x=1; int y = *x; }\n", "invalid type argument of unary '*' (have 'int')", 1, col=27)
    assert out and "*x" not in out


def test_double_index_cast():
    out = fx("int main(){\n int a[5];\n double i=2.5;\n a[i]=1;\n}\n", "invalid types 'int [5][double]' for array subscript", 4)
    assert out and "static_cast<int>(i)" in out


def test_gets_to_fgets_with_real_size():
    out = fx("char u[32];\nint main(){ gets(u); }\n", "'gets' was not declared in this scope", 2)
    assert out and "fgets(u, 32, stdin)" in out


def test_global_duplicate_variable_is_commented_out():
    out = fx("int x = 10;\nint x = 20;\nint main(){}\n", "redefinition of 'int x'", 2, cat="redefinition")
    assert out and out.count("int x") == 2 and "// removed duplicate" in out


def test_ambiguous_cases_are_not_guessed():
    assert fx("void foo(int x){}\nint main(){ foo(1,2); }\n", "too many arguments to function 'void foo(int)'", 2) is None
    assert fx("int main(){ return \"hello\"; }\n", "invalid conversion from 'const char*' to 'int'", 1) is None


def test_assignment_to_undeclared_becomes_declaration():
    out = fx("int main(){\n  int a=1,b=2;\n  result = a + b;\n}\n", "'result' was not declared in this scope", 3)
    assert out and "auto result = a + b;" in out


def test_bare_use_of_undeclared_is_not_guessed():
    assert fx("#include <iostream>\nint main(){ std::cout << y; }\n", "'y' was not declared in this scope", 2) is None


def test_void_result_assignment_becomes_call():
    out = fx("void foo(){}\nint main(){\n  int x = foo();\n}\n", "void value not ignored as it ought to be", 3)
    assert out and "foo();" in out and "int x" not in out
    assert fx("void foo(){}\nint main(){\n  int x = foo();\n  return x;\n}\n", "void value not ignored as it ought to be", 3) is None


def test_duplicate_function_is_commented_out_and_flagged():
    src = "int foo() {\n  return 1;\n}\n\nint foo() {\n  return 2;\n}\nint main(){}\n"
    out = fx(src, "redefinition of 'int foo()'", 5, cat="redefinition")
    assert out and "review it" in out and out.count("return 1") == 1 and "// return 2;" in out


def test_c_library_header_for_strlen():
    out = fx("#include <iostream>\nint main(){ return strlen(\"a\"); }\n", "'strlen' was not declared in this scope", 2)
    assert out and "#include <cstring>" in out
