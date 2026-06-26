"""Local Flask UI for Probability Cup workflows."""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

from flask import (
    abort,
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from webapp import engine_bridge
from webapp import prediction_assistant
from webapp.calibration import (
    compute_brier_columns,
    find_largest_crowd_deviations,
    find_largest_rbp_losses,
    find_largest_rbp_wins,
    generate_guardrail_suggestions,
    load_settled_history_csv,
    summarize_by_event_type,
    summarize_by_probability_bucket,
    summarize_settled_performance,
    to_normalized_csv,
)
from webapp.db import connect, init_db
from webapp.forms import float_option, form_value, require_fields
from webapp.history import (
    create_session,
    get_session,
    latest_assistant_snapshot,
    latest_context_snapshot,
    latest_prediction_snapshot,
    latest_question_snapshot,
    latest_scoring_snapshot,
    list_sessions,
    log_run,
    save_assistant_snapshot,
    save_context_snapshot,
    save_prediction_snapshot,
    save_question_snapshot,
    save_scoring_snapshot,
    snapshot_history,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "data" / "cup" / "examples"


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "local-probability-cup-dev"

    @app.before_request
    def _ensure_db() -> None:
        with connect() as conn:
            init_db(conn)

    @app.get("/")
    def index():
        with connect() as conn:
            sessions = list_sessions(conn)
        return render_template("index.html", sessions=sessions)

    @app.post("/sessions")
    def create_session_route():
        values = {
            "home_team": form_value(request.form, "home_team"),
            "away_team": form_value(request.form, "away_team"),
            "match_id": form_value(request.form, "match_id"),
            "match_date": form_value(request.form, "match_date"),
            "notes": form_value(request.form, "notes"),
        }
        try:
            require_fields(values, ["home_team", "away_team", "match_id"])
            with connect() as conn:
                session_id = create_session(conn, **values)
                log_run(conn, "info", "created session", session_id)
            return redirect(url_for("session_detail", session_id=session_id))
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

    @app.post("/sessions/sample")
    def load_sample_session():
        try:
            with connect() as conn:
                session_id = create_session(
                    conn,
                    match_id="norway_senegal_sample",
                    home_team="Norway",
                    away_team="Senegal",
                    match_date="2026-06-22",
                    notes="Dummy smoke-test data, not official odds/probabilities/results.",
                )
                questions = _read_example("norway_senegal_questions.csv")
                predictions = engine_bridge.run_prediction_csv(
                    questions,
                    _read_example("norway_senegal_manual_odds.csv"),
                    _read_example("norway_senegal_model_probs.csv"),
                )
                scored = engine_bridge.run_scoring_csv(
                    predictions,
                    _read_example("norway_senegal_results.csv"),
                )
                save_question_snapshot(conn, session_id, questions, source="dummy_sample")
                save_prediction_snapshot(
                    conn, session_id, predictions,
                    engine_bridge.summarize_predictions_csv(predictions),
                )
                save_scoring_snapshot(
                    conn, session_id, scored,
                    engine_bridge.summarize_scoring_csv(scored),
                )
                log_run(conn, "info", "loaded dummy Norway/Senegal sample", session_id)
            return redirect(url_for("session_detail", session_id=session_id))
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

    @app.get("/sessions/<int:session_id>")
    def session_detail(session_id: int):
        with connect() as conn:
            session = get_session(conn, session_id)
            q = latest_question_snapshot(conn, session_id)
            p = latest_prediction_snapshot(conn, session_id)
            s = latest_scoring_snapshot(conn, session_id)
            c = latest_context_snapshot(conn, session_id)
            a = latest_assistant_snapshot(conn, session_id)
        return render_template(
            "session_detail.html",
            session=session,
            question=q,
            prediction=p,
            scoring=s,
            context_snapshot=c,
            assistant_snapshot=a,
            question_preview=_preview_csv(q["csv_text"] if q else ""),
            prediction_preview=_preview_csv(p["csv_text"] if p else ""),
            scoring_preview=_preview_csv(s["csv_text"] if s else ""),
        )

    @app.route("/sessions/<int:session_id>/import", methods=["GET", "POST"])
    def import_questions(session_id: int):
        with connect() as conn:
            session = get_session(conn, session_id)
        parsed_csv = ""
        summary: dict[str, Any] = {}
        if request.method == "POST":
            raw_text = request.form.get("raw_text", "")
            try:
                parsed_csv = engine_bridge.parse_raw_questions_to_csv(
                    raw_text,
                    session["home_team"],
                    session["away_team"],
                    session["match_id"],
                    session["match_date"],
                )
                summary = _import_summary(parsed_csv)
                with connect() as conn:
                    save_question_snapshot(conn, session_id, parsed_csv, source="raw_text")
                    log_run(conn, "info", "imported raw questions", session_id)
                flash("Questions imported into a new snapshot.", "success")
            except Exception as exc:
                flash(str(exc), "error")
        return render_template(
            "import_questions.html",
            session=session,
            parsed_csv=parsed_csv,
            parsed_rows=_preview_csv(parsed_csv, limit=100),
            summary=summary,
        )

    @app.route("/sessions/<int:session_id>/questions", methods=["GET", "POST"])
    def review_questions(session_id: int):
        with connect() as conn:
            session = get_session(conn, session_id)
            latest = latest_question_snapshot(conn, session_id)
        csv_text = latest["csv_text"] if latest else ""
        if request.method == "POST":
            csv_text = request.form.get("csv_text", "")
            try:
                with connect() as conn:
                    save_question_snapshot(conn, session_id, csv_text, source="manual_review")
                    log_run(conn, "info", "saved reviewed question CSV", session_id)
                flash("Question CSV saved as a new snapshot.", "success")
                return redirect(url_for("review_questions", session_id=session_id))
            except Exception as exc:
                flash(str(exc), "error")
        return render_template("review_questions.html", session=session, csv_text=csv_text)

    @app.route("/sessions/<int:session_id>/predictions", methods=["GET", "POST"])
    def predictions(session_id: int):
        with connect() as conn:
            session = get_session(conn, session_id)
            latest_q = latest_question_snapshot(conn, session_id)
            latest_p = latest_prediction_snapshot(conn, session_id)
        question_csv = latest_q["csv_text"] if latest_q else ""
        odds_csv = ""
        model_probs_csv = ""
        prediction_csv = latest_p["csv_text"] if latest_p else ""
        summary = _summary_from_snapshot(latest_p)
        risks = engine_bridge.flag_prediction_risks(prediction_csv) if prediction_csv else []
        workbench_rows = engine_bridge.question_csv_to_manual_probability_rows(question_csv)
        if request.method == "POST":
            question_csv = request.form.get("question_csv", "")
            odds_csv = request.form.get("odds_csv", "")
            model_probs_csv = request.form.get("model_probs_csv", "")
            workbench_rows = engine_bridge.question_csv_to_manual_probability_rows(question_csv)
            action = request.form.get("action", "generate_predictions")
            if action == "generate_manual_odds":
                for row in workbench_rows:
                    field = f"manual_probability_percent_{row['row_index']}"
                    row["manual_probability_percent"] = request.form.get(field, "")
                try:
                    odds_csv = engine_bridge.manual_probability_rows_to_odds_csv(
                        workbench_rows,
                        match_id=session["match_id"],
                    )
                    flash("Manual odds CSV generated from entered probabilities.", "success")
                except Exception as exc:
                    flash(str(exc), "error")
            else:
                odds_csv = _field_or_upload("odds_csv", "odds_file")
                model_probs_csv = _field_or_upload("model_probs_csv", "model_probs_file")
                options = {
                    "market_weight": float_option(request.form, "market_weight", 0.70),
                    "model_weight": float_option(request.form, "model_weight", 0.30),
                    "manual_weight": float_option(request.form, "manual_weight", 1.0),
                    "min_prob": float_option(request.form, "min_prob", 0.01),
                    "max_prob": float_option(request.form, "max_prob", 0.99),
                }
                try:
                    prediction_csv = engine_bridge.run_prediction_csv(
                        question_csv, odds_csv, model_probs_csv, options
                    )
                    summary = engine_bridge.summarize_predictions_csv(prediction_csv)
                    risks = engine_bridge.flag_prediction_risks(prediction_csv)
                    with connect() as conn:
                        save_prediction_snapshot(conn, session_id, prediction_csv, summary)
                        log_run(conn, "info", "generated predictions", session_id)
                    flash("Predictions saved as a new snapshot.", "success")
                except Exception as exc:
                    flash(str(exc), "error")
        return render_template(
            "predictions.html",
            session=session,
            question_csv=question_csv,
            odds_csv=odds_csv,
            model_probs_csv=model_probs_csv,
            workbench_rows=workbench_rows,
            prediction_csv=prediction_csv,
            prediction_rows=_preview_csv(prediction_csv, limit=100),
            summary=summary,
            risks=risks,
        )

    @app.route("/sessions/<int:session_id>/live", methods=["GET", "POST"])
    def live_prediction(session_id: int):
        with connect() as conn:
            session = get_session(conn, session_id)
            latest_q = latest_question_snapshot(conn, session_id)
            latest_context = latest_context_snapshot(conn, session_id)
        question_csv = latest_q["csv_text"] if latest_q else ""
        saved_context = _context_from_snapshot(latest_context)
        match_context = _live_match_context(request.form if request.method == "POST" else saved_context)
        assistant_rows = []
        prediction_csv = ""
        submission_sheet: list[dict[str, str]] = []
        submission_sheet_text = ""
        summary: dict[str, Any] = {}
        audit_summary: dict[str, Any] = {}
        if request.method == "POST" and question_csv:
            try:
                assistant_rows = _live_assistant_rows(question_csv, match_context, request.form)
                odds_csv = prediction_assistant.assistant_rows_to_manual_odds_csv(
                    assistant_rows,
                    match_id=session["match_id"],
                )
                prediction_csv = engine_bridge.run_prediction_csv(question_csv, odds_csv)
                summary = engine_bridge.summarize_predictions_csv(prediction_csv)
                with connect() as conn:
                    context_id = save_context_snapshot(
                        conn,
                        session_id,
                        _context_snapshot_json(match_context),
                        source="live_prediction",
                    )
                    assistant_id = save_assistant_snapshot(
                        conn,
                        session_id,
                        _assistant_snapshot_json(assistant_rows),
                        source="live_prediction",
                    )
                    save_prediction_snapshot(conn, session_id, prediction_csv, summary)
                    log_run(conn, "info", "generated live-mode predictions", session_id)
                submission_sheet = _submission_sheet(assistant_rows)
                submission_sheet_text = _submission_sheet_text(submission_sheet)
                audit_summary = {
                    "context_snapshot_id": context_id,
                    "assistant_snapshot_id": assistant_id,
                    "prediction_snapshot_saved": True,
                    "final_probabilities_entered": sum(
                        1 for row in assistant_rows if row.get("final_probability_percent")
                    ),
                    "blank_final_probabilities": sum(
                        1 for row in assistant_rows if not row.get("final_probability_percent")
                    ),
                }
                flash("Live predictions saved as a new prediction snapshot.", "success")
                if any(not row.get("final_probability_percent") for row in assistant_rows):
                    flash("Some final probabilities were blank; only entered probabilities were included.", "error")
            except Exception as exc:
                flash(str(exc), "error")
                assistant_rows = _live_assistant_rows(
                    question_csv,
                    match_context,
                    request.form,
                    validate_final=False,
                )
        else:
            assistant_rows = _live_assistant_rows(question_csv, match_context, {})
        total_questions = len(assistant_rows)
        completion_count = sum(1 for row in assistant_rows if row.get("final_probability_percent"))
        return render_template(
            "live_prediction.html",
            session=session,
            has_questions=bool(question_csv.strip()),
            match_context=match_context,
            assistant_rows=assistant_rows,
            total_questions=total_questions,
            completion_count=completion_count,
            missing_count=total_questions - completion_count,
            submission_sheet=submission_sheet,
            submission_sheet_text=submission_sheet_text,
            prediction_csv=prediction_csv,
            summary=summary,
            audit_summary=audit_summary,
            latest_context=latest_context,
        )

    @app.route("/sessions/<int:session_id>/scoring", methods=["GET", "POST"])
    def scoring(session_id: int):
        with connect() as conn:
            session = get_session(conn, session_id)
            latest_p = latest_prediction_snapshot(conn, session_id)
            latest_s = latest_scoring_snapshot(conn, session_id)
            latest_c = latest_context_snapshot(conn, session_id)
            latest_a = latest_assistant_snapshot(conn, session_id)
        predictions_csv = latest_p["csv_text"] if latest_p else ""
        scoring_csv = latest_s["csv_text"] if latest_s else ""
        summary = _summary_from_snapshot(latest_s)
        final_probability_rows = _final_probability_rows_from_snapshot(latest_a)
        if request.method == "POST":
            predictions_csv = request.form.get("predictions_csv", "")
            results_csv = _field_or_upload("results_csv", "results_file")
            try:
                scoring_csv = engine_bridge.run_scoring_csv(predictions_csv, results_csv)
                summary = engine_bridge.summarize_scoring_csv(scoring_csv)
                with connect() as conn:
                    save_scoring_snapshot(conn, session_id, scoring_csv, summary)
                    log_run(conn, "info", "generated scoring snapshot", session_id)
                flash("Scoring saved as a new snapshot.", "success")
            except Exception as exc:
                flash(str(exc), "error")
        return render_template(
            "scoring.html",
            session=session,
            predictions_csv=predictions_csv,
            scoring_csv=scoring_csv,
            scoring_rows=_preview_csv(scoring_csv, limit=100),
            summary=summary,
            context_snapshot=latest_c,
            assistant_snapshot=latest_a,
            final_probability_rows=final_probability_rows,
        )

    @app.get("/sessions/<int:session_id>/history")
    def history(session_id: int):
        with connect() as conn:
            session = get_session(conn, session_id)
            hist = snapshot_history(conn, session_id)
        return render_template("history.html", session=session, history=hist)

    @app.get("/sessions/<int:session_id>/download/<kind>/latest")
    def download_latest_snapshot(session_id: int, kind: str):
        latest_loaders = {
            "questions": latest_question_snapshot,
            "predictions": latest_prediction_snapshot,
            "scoring": latest_scoring_snapshot,
        }
        loader = latest_loaders.get(kind)
        if loader is None:
            abort(404)
        with connect() as conn:
            session = get_session(conn, session_id)
            row = loader(conn, session_id)
        if row is None:
            abort(404)
        filename = f"{_safe_filename_part(session['match_id'])}_{kind}_latest.csv"
        return Response(
            row["csv_text"],
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/sessions/<int:session_id>/snapshots/<kind>/latest")
    def view_latest_json_snapshot(session_id: int, kind: str):
        latest_loaders = {
            "context": latest_context_snapshot,
            "assistant": latest_assistant_snapshot,
        }
        loader = latest_loaders.get(kind)
        if loader is None:
            abort(404)
        with connect() as conn:
            session = get_session(conn, session_id)
            row = loader(conn, session_id)
        if row is None:
            abort(404)
        return render_template(
            "snapshot_text.html",
            session=session,
            title=f"Latest {kind} snapshot",
            snapshot=row,
            text=_json_snapshot_text(row, kind),
        )

    @app.get("/sessions/<int:session_id>/snapshots/<kind>/<int:snapshot_id>")
    def view_json_snapshot(session_id: int, kind: str, snapshot_id: int):
        tables = {
            "context": ("context_snapshots", "context_json"),
            "assistant": ("assistant_snapshots", "assistant_json"),
        }
        table_info = tables.get(kind)
        if table_info is None:
            abort(404)
        table, column = table_info
        with connect() as conn:
            session = get_session(conn, session_id)
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND session_id = ?",
                (snapshot_id, session_id),
            ).fetchone()
        if row is None:
            abort(404)
        return render_template(
            "snapshot_text.html",
            session=session,
            title=f"{kind.title()} snapshot {snapshot_id}",
            snapshot=row,
            text=row[column],
        )

    @app.route("/calibration", methods=["GET", "POST"])
    def calibration():
        csv_text = ""
        normalized_csv = ""
        results: dict[str, Any] = {}
        with connect() as conn:
            live_snapshot_sessions = [
                row for row in list_sessions(conn)
                if row.get("latest_assistant_at") or row.get("latest_context_at")
            ]
        if request.method == "POST":
            csv_text = _field_or_upload("csv_text", "csv_file")
            try:
                df = load_settled_history_csv(csv_text)
                scored = compute_brier_columns(df)
                normalized_csv = to_normalized_csv(scored)
                results = {
                    "summary": summarize_settled_performance(scored),
                    "by_event_type": _records(summarize_by_event_type(scored)),
                    "by_bucket": _records(summarize_by_probability_bucket(scored)),
                    "largest_wins": _records(_display_rows(find_largest_rbp_wins(scored))),
                    "largest_losses": _records(_display_rows(find_largest_rbp_losses(scored))),
                    "largest_deviations": _records(_display_rows(find_largest_crowd_deviations(scored))),
                    "suggestions": generate_guardrail_suggestions(scored),
                }
                flash("Settled history analyzed locally. No data was saved.", "success")
            except Exception as exc:
                flash(str(exc), "error")
        return render_template(
            "calibration.html",
            csv_text=csv_text,
            normalized_csv=normalized_csv,
            results=results,
            live_snapshot_sessions=live_snapshot_sessions,
        )

    @app.post("/calibration/download")
    def download_calibration_csv():
        normalized_csv = request.form.get("normalized_csv", "")
        if not normalized_csv.strip():
            flash("No normalized settled-history CSV is available to download.", "error")
            return redirect(url_for("calibration"))
        return Response(
            normalized_csv,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=settled_history_normalized.csv"},
        )

    @app.get("/sessions/<int:session_id>/download/<kind>/<int:snapshot_id>")
    def download_snapshot(session_id: int, kind: str, snapshot_id: int):
        tables = {
            "questions": "question_snapshots",
            "predictions": "prediction_snapshots",
            "scoring": "scoring_snapshots",
        }
        table = tables.get(kind)
        if table is None:
            flash("Unknown snapshot type.", "error")
            return redirect(url_for("history", session_id=session_id))
        with connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND session_id = ?",
                (snapshot_id, session_id),
            ).fetchone()
        if row is None:
            flash("Snapshot not found.", "error")
            return redirect(url_for("history", session_id=session_id))
        return Response(
            row["csv_text"],
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={kind}_{snapshot_id}.csv"},
        )

    return app


def _read_example(name: str) -> str:
    return (EXAMPLE_DIR / name).read_text(encoding="utf-8")


def _preview_csv(csv_text: str, limit: int = 8) -> list[dict[str, str]]:
    if not csv_text.strip():
        return []
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for idx, row in enumerate(reader):
        if idx >= limit:
            break
        rows.append({k: "" if v is None else v for k, v in row.items()})
    return rows


def _import_summary(csv_text: str) -> dict[str, Any]:
    df = engine_bridge._read_csv_text(csv_text)  # local UI bridge helper
    if df.empty:
        return {}
    return {
        "total": int(len(df)),
        "status_counts": {str(k): int(v) for k, v in df["status"].value_counts().to_dict().items()},
        "event_type_counts": {str(k): int(v) for k, v in df["event_type"].value_counts().to_dict().items()},
    }


def _summary_from_snapshot(snapshot: Any) -> dict[str, Any]:
    if not snapshot or "summary_json" not in snapshot.keys():
        return {}
    try:
        return json.loads(snapshot["summary_json"])
    except Exception:
        return {}


def _live_match_context(form: Any) -> dict[str, str]:
    return {
        "favorite_team": form_value(form, "favorite_team") if form else "",
        "expected_match_script": form_value(form, "expected_match_script") if form else "",
        "tournament_context": form_value(form, "tournament_context") if form else "",
        "player_context": form_value(form, "player_context") if form else "",
        "user_notes": form_value(form, "user_notes") if form else "",
    }


def _context_from_snapshot(snapshot: Any) -> dict[str, str]:
    empty = _live_match_context({})
    if not snapshot:
        return empty
    try:
        data = json.loads(snapshot["context_json"])
    except Exception:
        return empty
    return {key: str(data.get(key, "")) for key in empty}


def _context_snapshot_json(context: dict[str, str]) -> str:
    return json.dumps(_live_match_context(context), indent=2, sort_keys=True)


def _assistant_snapshot_json(rows: list[dict[str, Any]]) -> str:
    keep = [
        "question_id",
        "raw_question",
        "event_type",
        "normalized_event_type",
        "suggested_probability_percent",
        "suggested_range",
        "confidence",
        "reasoning",
        "risk_flags",
        "exposure_warnings",
        "final_probability_percent",
    ]
    payload = [
        {key: _jsonable(row.get(key, "")) for key in keep}
        for row in rows
    ]
    return json.dumps(payload, indent=2, sort_keys=True)


def _live_assistant_rows(
    question_csv: str,
    match_context: dict[str, str],
    form: Any,
    validate_final: bool = True,
) -> list[dict[str, Any]]:
    question_rows = _preview_csv(question_csv, limit=10)
    rows = []
    for index, row in enumerate(question_rows):
        final_value = form.get(f"final_probability_percent_{index}", "") if form else ""
        row["final_probability_percent"] = final_value if validate_final else ""
        suggested = prediction_assistant.suggest_probability_for_question(row, match_context)
        if not validate_final:
            suggested["final_probability_percent"] = final_value
        rows.append(suggested)
    exposure_warnings = []
    if validate_final:
        exposure_warnings = prediction_assistant.detect_match_script_exposure(rows)
    if exposure_warnings:
        for row in rows:
            row["exposure_warnings"] = exposure_warnings
    return rows


def _submission_sheet(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    sheet = []
    for index, row in enumerate(rows, start=1):
        if not row.get("final_probability_percent"):
            continue
        sheet.append({
            "q_number": f"Q{index}",
            "question_id": str(row.get("question_id", "")),
            "raw_question": str(row.get("raw_question", "")),
            "final_probability_percent": str(row.get("final_probability_percent", "")),
        })
    return sheet


def _submission_sheet_text(rows: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{row['q_number']} {row['final_probability_percent']}% - {row['raw_question']}"
        for row in rows
    )


def _final_probability_rows_from_snapshot(snapshot: Any) -> list[dict[str, str]]:
    if not snapshot:
        return []
    try:
        rows = json.loads(snapshot["assistant_json"])
    except Exception:
        return []
    final_rows = []
    for index, row in enumerate(rows, start=1):
        final = str(row.get("final_probability_percent", "")).strip()
        if not final:
            continue
        final_rows.append({
            "q_number": f"Q{index}",
            "question_id": str(row.get("question_id", "")),
            "raw_question": str(row.get("raw_question", "")),
            "final_probability_percent": final,
        })
    return final_rows


def _json_snapshot_text(row: Any, kind: str) -> str:
    column = "context_json" if kind == "context" else "assistant_json"
    return row[column]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _field_or_upload(field_name: str, file_name: str) -> str:
    upload = request.files.get(file_name)
    if upload and upload.filename:
        return upload.read().decode("utf-8")
    return request.form.get(field_name, "")


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "session"


def _records(df: Any) -> list[dict[str, Any]]:
    return [
        {key: _format_value(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _display_rows(df: Any) -> Any:
    columns = [
        "question_id",
        "event_type",
        "selection",
        "raw_question",
        "user_prob",
        "crowd_prob",
        "actual_result",
        "platform_rbp",
        "platform_rbp_numeric",
        "user_brier",
        "crowd_brier",
        "abs_user_crowd_deviation",
    ]
    return df[[column for column in columns if column in df.columns]]


def _format_value(value: Any) -> Any:
    try:
        if value != value:
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return round(value, 4)
    return value


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
