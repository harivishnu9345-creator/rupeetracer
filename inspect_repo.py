#!/usr/bin/env python3
"""
RupeeTrace repo inspector.

Usage:
    python inspect_repo.py            # inspects the current folder
    python inspect_repo.py path/to/rupeetrace

What it prints (and saves to repo_report.txt next to the script's target folder):
  1. Folder tree with file sizes, line counts, row counts for CSVs, EMPTY flags
  2. Checklist of expected files: PRESENT / EMPTY / FOUND ELSEWHERE / MISSING
  3. Unresolved imports: modules or names that your code imports but nobody has written yet
  4. Test inventory: number of test functions per test file

Paste the whole report into the chat to see exactly what is left.
"""
import ast
import os
import sys
from pathlib import Path

SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules",
    ".idea", ".vscode", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
}
LOCAL_PKGS = {"config", "data", "ledger", "features", "models", "engine",
              "dashboard", "eval", "tests"}

# (path, optional?)
EXPECTED = [
    ("config.py", False),
    ("requirements.txt", False),
    ("README.md", False),
    ("data/schemas.py", False),
    ("data/generator.py", False),
    ("data/ibm_loader.py", True),
    ("data/out/accounts.csv", False),
    ("data/out/transactions.csv", False),
    ("data/out/scenarios.csv", False),
    ("ledger/lots.py", False),
    ("ledger/hashchain.py", False),
    ("features/behavior.py", False),
    ("features/graph.py", False),
    ("features/provenance.py", False),
    ("models/train.py", False),
    ("models/explain.py", False),
    ("models/stream_scorer.py", False),
    ("models/saved/thresholds.json", False),
    ("engine/fusion.py", False),
    ("engine/orchestrator.py", False),
    ("engine/cases.py", False),
    ("engine/engine.py", False),
    ("engine/mock_provider.py", False),
    ("engine/api.py", False),
    ("dashboard/app.py", False),
    ("engine/metrics.py", False),
    ("engine/recording.py", False),
    ("dashboard/components/__init__.py", False),
    ("dashboard/components/charts.py", False),
    ("dashboard/components/ui.py", False),
    ("eval/rules_baseline.py", False),
    ("eval/metrics.py", False),
    ("eval/ablation.py", False),
    ("eval/buckets.py", False),
    ("eval/latency.py", False),
    ("eval/gc_latency_check.py", False),
    ("eval/out/predictions_A_rules.csv", False),
    ("eval/out/predictions_B_behavior_anomaly.csv", False),
    ("eval/out/predictions_C_plus_graph.csv", False),
    ("eval/out/predictions_D_plus_provenance.csv", False),
    ("tests/test_data.py", False),
    ("tests/test_eval.py", False),
    ("tests/test_ibm_loader.py", False),
    ("tests/test_provenance.py", False),
    ("tests/test_models.py", False),
    ("tests/test_ledger.py", False),
    ("tests/test_hashchain.py", False),
    ("tests/test_no_leakage.py", False),
    ("tests/test_features.py", False),
    ("tests/test_fusion.py", False),
    ("tests/test_orchestrator.py", False),
    ("tests/test_cases.py", False),
    ("tests/test_mock_provider.py", False),
    ("tests/test_components.py", False),
    ("tests/test_api.py", False),
]

OUT = []


def emit(line=""):
    print(line)
    OUT.append(line)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def walk_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            files.append(Path(dirpath) / fn)
    return files


# ---------------------------------------------------------------- 1. tree
def print_tree(root):
    emit("=" * 70)
    emit("1. FOLDER TREE")
    emit("=" * 70)

    def rec(folder, prefix=""):
        entries = sorted(
            [e for e in folder.iterdir() if e.name not in SKIP_DIRS],
            key=lambda e: (e.is_file(), e.name.lower()),
        )
        for i, e in enumerate(entries):
            last = i == len(entries) - 1
            branch = "`-- " if last else "|-- "
            if e.is_dir():
                emit(f"{prefix}{branch}{e.name}/")
                rec(e, prefix + ("    " if last else "|   "))
            else:
                size = e.stat().st_size
                note = ""
                if size == 0:
                    note = "  <-- EMPTY"
                elif e.suffix in (".py", ".md", ".txt", ".json"):
                    n = count_lines(e)
                    note = f"  ({n} lines)"
                elif e.suffix == ".csv":
                    n = count_lines(e)
                    note = f"  ({max((n or 1) - 1, 0)} rows)"
                emit(f"{prefix}{branch}{e.name}  [{human(size)}]{note}")

    emit(f"{root.name}/")
    rec(root)
    emit()


# ------------------------------------------------------------- 2. checklist
def checklist(root, files):
    emit("=" * 70)
    emit("2. EXPECTED FILES CHECKLIST")
    emit("=" * 70)
    by_name = {}
    for f in files:
        by_name.setdefault(f.name, []).append(f)

    missing, elsewhere, empty, present = [], [], [], 0
    for rel, optional in EXPECTED:
        p = root / rel
        tag = " (optional)" if optional else ""
        if p.exists():
            if p.stat().st_size == 0 and p.name != "__init__.py":
                emit(f"  [EMPTY]      {rel}{tag}")
                empty.append(rel)
            else:
                emit(f"  [PRESENT]    {rel}{tag}")
                present += 1
        elif p.name in by_name:
            where = ", ".join(str(x.relative_to(root)) for x in by_name[p.name][:3])
            emit(f"  [ELSEWHERE]  {rel}{tag}  -> found at: {where}")
            elsewhere.append(rel)
        else:
            emit(f"  [MISSING]    {rel}{tag}")
            if not optional:
                missing.append(rel)

    emit()
    emit(f"Summary: {present} present, {len(empty)} empty, "
         f"{len(elsewhere)} in the wrong place, {len(missing)} missing (required).")
    if elsewhere:
        emit("  Wrong-place files need to be moved into the folder shown above.")
    if missing:
        emit("  STILL TO GENERATE: " + ", ".join(missing))
    emit()


# --------------------------------------------------------- 3. import checks
def top_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            # names defined inside if/try at top level (e.g. optional imports)
            sub = ast.Module(body=getattr(node, "body", []) +
                             getattr(node, "orelse", []), type_ignores=[])
            names |= top_level_names(sub)
    return names


def resolve_module(root, parts):
    """Return the .py file (or package __init__) for a dotted module, else None."""
    base = root.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.is_dir():
        return base  # namespace package without __init__
    return None


def check_imports(root, files):
    emit("=" * 70)
    emit("3. UNRESOLVED IMPORTS (code that depends on something not written yet)")
    emit("=" * 70)

    missing_modules = {}  # module -> {file: set(names)}
    missing_names = {}    # (module_file_rel) -> {name: set(importers)}
    parse_errors = []
    name_cache = {}

    def names_of(path):
        if path not in name_cache:
            try:
                name_cache[path] = top_level_names(
                    ast.parse(path.read_text(encoding="utf-8", errors="ignore")))
            except SyntaxError:
                name_cache[path] = None
        return name_cache[path]

    py_files = [f for f in files if f.suffix == ".py"]
    for f in py_files:
        rel = str(f.relative_to(root))
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as e:
            parse_errors.append(f"{rel}: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    if parts[0] in LOCAL_PKGS and resolve_module(root, parts) is None:
                        missing_modules.setdefault(a.name, {}).setdefault(rel, set())
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    base_dir = f.parent
                    for _ in range(node.level - 1):
                        base_dir = base_dir.parent
                    try:
                        rel_parts = list(base_dir.relative_to(root).parts)
                    except ValueError:
                        continue
                    parts = rel_parts + (node.module.split(".") if node.module else [])
                else:
                    if not node.module:
                        continue
                    parts = node.module.split(".")
                    if parts[0] not in LOCAL_PKGS:
                        continue
                mod_name = ".".join(parts) if parts else "(package root)"
                target = resolve_module(root, parts) if parts else root
                if target is None:
                    entry = missing_modules.setdefault(mod_name, {}).setdefault(rel, set())
                    entry.update(a.name for a in node.names)
                    continue
                if target.is_file():
                    defined = names_of(target)
                    if defined is None:
                        continue
                    for a in node.names:
                        if a.name == "*" or a.name in defined:
                            continue
                        # maybe it's a submodule of a package
                        sub = target.parent / a.name
                        if target.name == "__init__.py" and (
                                sub.with_suffix(".py").is_file() or sub.is_dir()):
                            continue
                        key = str(target.relative_to(root))
                        missing_names.setdefault(key, {}).setdefault(a.name, set()).add(rel)
                else:  # directory without __init__: names must be submodules
                    for a in node.names:
                        sub = target / a.name
                        if not (sub.with_suffix(".py").is_file() or sub.is_dir()):
                            missing_modules.setdefault(
                                f"{mod_name}.{a.name}", {}).setdefault(rel, set())

    if not missing_modules and not missing_names:
        emit("  None. Every project import resolves.")
    if missing_modules:
        emit("  Modules that are imported but do not exist:")
        for mod in sorted(missing_modules):
            importers = missing_modules[mod]
            emit(f"    - {mod}")
            for imp, names in sorted(importers.items()):
                tail = f"  needs: {', '.join(sorted(names))}" if names else ""
                emit(f"        imported by {imp}{tail}")
    if missing_names:
        emit("  Modules that exist but lack names other files import:")
        for mod in sorted(missing_names):
            emit(f"    - {mod}")
            for name, importers in sorted(missing_names[mod].items()):
                emit(f"        missing `{name}`  (used by {', '.join(sorted(importers))})")
    if parse_errors:
        emit("  Files with syntax errors (cannot be analysed):")
        for p in parse_errors:
            emit(f"    - {p}")
    emit("  (Imports inside try/except may be optional, so treat those as hints.)")
    emit()


# ------------------------------------------------------------- 4. tests
def test_inventory(root, files):
    emit("=" * 70)
    emit("4. TEST INVENTORY")
    emit("=" * 70)
    tests = [f for f in files if f.suffix == ".py" and f.parent.name == "tests"
             and f.name.startswith("test_")]
    if not tests:
        emit("  No test files found.")
        emit()
        return
    total = 0
    for f in tests:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            emit(f"  {f.relative_to(root)}: SYNTAX ERROR")
            continue
        n = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test_"):
                n += 1
        total += n
        emit(f"  {str(f.relative_to(root)):45s} {n} test functions")
    emit(f"  Total: {total} test functions")
    emit()


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not root.is_dir():
        print(f"Not a folder: {root}")
        sys.exit(1)
    files = walk_files(root)
    emit(f"Inspecting: {root}")
    emit(f"Total files: {len(files)}")
    emit()
    print_tree(root)
    checklist(root, files)
    check_imports(root, files)
    test_inventory(root, files)
    report = root / "repo_report.txt"
    try:
        report.write_text("\n".join(OUT), encoding="utf-8")
        print(f"Report saved to: {report}")
    except Exception as e:
        print(f"(could not save report: {e})")


if __name__ == "__main__":
    main()
