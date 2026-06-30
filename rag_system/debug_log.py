import json
import time

LOG_PATH = "/home/revinr/Soykot/Rag1_system/.cursor/debug-99d8cb.log"


def debug_log(location, message, data=None, hypothesis_id=None, run_id="pre-fix"):
    # #region agent log
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "99d8cb",
                        "timestamp": int(time.time() * 1000),
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "hypothesisId": hypothesis_id,
                        "runId": run_id,
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
