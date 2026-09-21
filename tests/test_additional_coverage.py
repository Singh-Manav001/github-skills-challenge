import runpy
from importlib import import_module

import pytest

from src.aiops_pipeline import load_data, run_pipeline
from src.anomaly_detector import AnomalyDetector
from src.calculations import area_of_circle, get_nth_fibonacci
from src.event_producer import EventProducer
from src.event_topic import EventTopic


def test_calculations_reject_negative_values():
    with pytest.raises(ValueError, match="Radius cannot be negative"):
        area_of_circle(-1)

    with pytest.raises(ValueError, match="n cannot be negative"):
        get_nth_fibonacci(-1)


def test_calculations_compute_nontrivial_values():
    assert area_of_circle(2) == pytest.approx(12.566370614359172)
    assert get_nth_fibonacci(10) == 55


def test_detector_reports_each_anomaly_reason():
    detector = AnomalyDetector()
    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "payment-service",
        "response_time_ms": 501,
        "cpu_percent": 81,
        "memory_percent": 81,
        "log_level": "WARNING",
    }

    event = detector.detect(record)

    assert event["reasons"] == [
        "High response time",
        "High CPU utilization",
        "High memory utilization",
        "Error log detected",
    ]
    assert event["source"] == record


def test_producer_rejects_empty_event():
    topic = EventTopic("anomaly-events")

    assert EventProducer(topic).publish(None) is False
    assert topic.get_messages() == []


def test_topic_clear_removes_published_messages():
    topic = EventTopic("anomaly-events")
    topic.publish({"type": "ANOMALY"})

    topic.clear()

    assert topic.get_messages() == []


def test_pipeline_loads_data_and_consumes_anomalies():
    data = load_data("data/service_data.json")
    result = run_pipeline("data/service_data.json")

    assert result["records_processed"] == len(data)
    assert len(result["anomalies_detected"]) == 2
    assert result["events_consumed"] == []


def test_pipeline_script_prints_results(monkeypatch, capsys):
    monkeypatch.syspath_prepend("src")
    event_consumer_module = import_module("event_consumer")

    class ReportingConsumer:
        def __init__(self, topic):
            self.topic = topic

        def consume(self):
            return [{
                "service": "payment-service",
                "timestamp": "2026-09-20T10:05:00",
                "type": "ANOMALY",
                "reasons": ["High response time"],
            }]

    monkeypatch.setattr(event_consumer_module, "EventConsumer", ReportingConsumer)
    runpy.run_path("src/aiops_pipeline.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "AIOps Pipeline Result" in output
    assert "Records processed: 10" in output
    assert "Anomalies detected: 2" in output
    assert "Events consumed: 1" in output
    assert "Service: payment-service" in output