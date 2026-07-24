# ---------------------------------------------------------------------------
# NOTE: ``dargus/cli.py`` has been replaced by the ``dargus/cli/`` package.
# All imports that previously targeted ``dargus.cli`` (e.g.
# ``from dargus.cli import main``) are resolved through this __init__ and
# continue to work.  The full logic now lives in ``dargus.cli.main``.
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
