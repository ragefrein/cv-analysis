from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Team = Literal["home", "away", "unknown"]
EventType = Literal["pass", "shot", "goal", "own_goal", "assist", "interception", "turnover", "recovery", "save", "block", "dribble", "foul", "yellow_card", "red_card", "substitution", "kick_in", "corner", "penalty", "possession"]
Outcome = Literal["successful", "failed", "unknown", "on_target", "off_target", "blocked", "goal"]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class MatchCreate(Input):
    title: str = Field(min_length=1, max_length=160)
    home_name: str = Field(min_length=1, max_length=80)
    away_name: str = Field(min_length=1, max_length=80)
    home_color: str = Field(default="#10b981", pattern=r"^#[0-9a-fA-F]{6}$")
    away_color: str = Field(default="#f59e0b", pattern=r"^#[0-9a-fA-F]{6}$")
    played_at: str = Field(default="", max_length=40)
    venue: str = Field(default="", max_length=160)


class MatchPatch(Input):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    home_name: str | None = Field(default=None, min_length=1, max_length=80)
    away_name: str | None = Field(default=None, min_length=1, max_length=80)
    home_color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    away_color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    played_at: str | None = Field(default=None, max_length=40)
    venue: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=5000)
    status: Literal["review", "finalized"] | None = None


class PlayerInput(Input):
    name: str = Field(min_length=1, max_length=100)
    number: str = Field(default="", max_length=6)
    team: Literal["home", "away"]
    role: Literal["player", "goalkeeper"] = "player"

    @field_validator("number", mode="before")
    @classmethod
    def number_to_string(cls, value):
        return str(value) if value is not None else ""


class PlayerPatch(Input):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    number: str | None = Field(default=None, max_length=6)
    team: Literal["home", "away"] | None = None
    role: Literal["player", "goalkeeper"] | None = None


class TrackInput(Input):
    player_id: str | None = None
    team: Team | None = None
    excluded: bool | None = None


class EventInput(Input):
    type: EventType
    time: float = Field(ge=0)
    end_time: float | None = Field(default=None, ge=0)
    team: Team = "unknown"
    player_id: str | None = None
    receiver_id: str | None = None
    outcome: Outcome = "unknown"
    status: Literal["pending", "confirmed", "rejected"] = "confirmed"
    notes: str = Field(default="", max_length=3000)
    x: float | None = Field(default=None, ge=0, le=200)
    y: float | None = Field(default=None, ge=0, le=200)
    related_event_id: str | None = None

    @model_validator(mode="after")
    def time_range(self):
        if self.end_time is not None and self.end_time <= self.time:
            raise ValueError("Waktu akhir harus sesudah waktu mulai.")
        if self.type == "possession" and self.end_time is None:
            raise ValueError("Penguasaan bola memerlukan waktu akhir.")
        allowed = {
            "pass": {"unknown", "successful", "failed"},
            "shot": {"unknown", "on_target", "off_target", "blocked", "goal"},
            "goal": {"unknown", "goal"},
            "own_goal": {"unknown", "goal"},
            "dribble": {"unknown", "successful", "failed"},
            "penalty": {"unknown", "successful", "failed"},
        }.get(self.type, {"unknown", "successful"})
        if self.outcome not in allowed:
            raise ValueError("Hasil tidak sesuai jenis kejadian; gunakan hasil yang relevan atau belum diketahui.")
        if self.type != "shot" and self.outcome == "goal" and self.type not in ("goal", "own_goal"):
            raise ValueError("Hasil gol hanya untuk tembakan atau kejadian gol.")
        return self


class EventPatch(Input):
    type: EventType | None = None
    time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    team: Team | None = None
    player_id: str | None = None
    receiver_id: str | None = None
    outcome: Outcome | None = None
    status: Literal["pending", "confirmed", "rejected"] | None = None
    notes: str | None = Field(default=None, max_length=3000)
    x: float | None = Field(default=None, ge=0, le=200)
    y: float | None = Field(default=None, ge=0, le=200)
    related_event_id: str | None = None


class ReviewInput(Input):
    ids: list[str] = Field(min_length=1, max_length=2000)
    status: Literal["confirmed", "rejected"]


class JobInput(Input):
    sample_fps: float = Field(default=5, ge=1, le=15)
    confidence: float = Field(default=0.25, ge=0.05, le=0.9)
    detect_ball: bool = True
    auto_court: bool = True


class CalibrationInput(Input):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    court_length: float = Field(default=40, ge=10, le=100)
    court_width: float = Field(default=20, ge=5, le=60)
    image_points: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    court_points: list[tuple[float, float]] | None = Field(default=None, min_length=4, max_length=4)
    quality: Literal["manual"] = "manual"
    notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def valid_points(self):
        if self.end_time <= self.start_time:
            raise ValueError("Waktu akhir harus sesudah waktu mulai.")
        if any(not 0 <= p <= 1 for point in self.image_points for p in point):
            raise ValueError("Koordinat gambar harus dalam rentang 0 sampai 1.")
        if self.court_points is None:
            self.court_points = [(0, 0), (self.court_length, 0), (self.court_length, self.court_width), (0, self.court_width)]
        if any(not 0 <= x <= self.court_length or not 0 <= y <= self.court_width for x, y in self.court_points):
            raise ValueError("Titik lapangan berada di luar dimensi lapangan.")
        for points in (self.image_points, self.court_points):
            # Convex, ordered quadrilateral; crossing or degenerate points cannot calibrate.
            cross = []
            for i in range(4):
                a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
                cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
            if not (all(v > 1e-7 for v in cross) or all(v < -1e-7 for v in cross)):
                raise ValueError("Empat titik harus berurutan mengelilingi area dan tidak segaris.")
        return self
