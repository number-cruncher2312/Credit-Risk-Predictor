MONITORING_CONFIG = {
    "cadence": "monthly",


    "reference": {
        "type": "fixed",
        "dataset": "training_v1"
    },


    "psi": {
        "warning_threshold": 0.1,
        "critical_threshold": 0.25,
        "history_path": "data/psi_history.csv",
    },

    "csi": {
        "enabled": True,
        "top_n_features": 5,
        "warning_threshold": 0.10,
        "critical_threshold": 0.25,
    }
}
