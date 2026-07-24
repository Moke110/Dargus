# ---------------------------------------------------------------------------
# NOTE: ``dargus/cli.py`` has been replaced by the ``dargus/cli/`` package.
# All imports that previously targeted ``dargus.cli`` (e.g.
# ``from dargus.cli import main``) are resolved through this __init__ and
# continue to work.  The full logic now lives in ``dargus.cli.main``.
#
# A module-level ``__getattr__`` provides a lazy fallback for any symbol
# not explicitly imported above, ensuring that code written against the
# old single-file ``dargus.cli`` module (e.g. ``from dargus.cli import
# _some_private_helper``) continues to function.
# ---------------------------------------------------------------------------

from dargus.cli.main import (  # noqa: F401
    _arrow_menu,
    _clear_dbase,
    _cli_confirm,
    _count_tests,
    _draw_progress,
    _json_arg,
    _print_troubleshooting,
    _resolve_config_value,
    _restore_working_dbase,
    _run_model_wizard,
    _run_test_bulk_input,
    _run_test_dbase,
    _run_test_ingest,
    _run_test_suite,
    _scan_data_dir,
    _write_ingest_report,
    main,
)


def __getattr__(name: str):
    """Lazy fallback: resolve any symbol not explicitly imported from
    ``dargus.cli.main``.  This preserves backward compatibility for code
    that was written against the old single-file ``dargus/cli.py`` module
    (since a ``dargus/cli.py`` file and ``dargus/cli/`` package cannot
    physically coexist in Python).
    """
    import dargus.cli.main as _main

    if hasattr(_main, name):
        return getattr(_main, name)
    raise AttributeError(f"module 'dargus.cli' has no attribute {name!r}")
