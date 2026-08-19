"""Physics-informed digital-twin simulator for military battle-tank
preventive maintenance.

Sensors are simulated from physics-based equations and coupled to a
digital-twin state that evolves over time.  Fault profiles modify the
physical parameters so that the simulated readings degrade in a
physically consistent way, providing labelled datasets for AI-based
anomaly detection, fault diagnosis and remaining-useful-life (RUL)
prediction.
"""

__version__ = "0.1.0"