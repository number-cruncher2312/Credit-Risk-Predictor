from dataclasses import dataclass, field



@dataclass
class DriftMetric:
    name: str
    value: float
    status: str
    expected_distribution: list[float] = field(default_factory=list)
    actual_distribution: list[float] = field(default_factory=list)
    bucket_labels: list[str] = field(default_factory=list)


@dataclass
class DriftResults:
    psi: DriftMetric
    csi: dict[str, DriftMetric] = field(default_factory=dict)

