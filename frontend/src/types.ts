export type Team = 'home' | 'away' | 'unknown';
export interface Match {
  id: string; title: string; home_name: string; away_name: string; home_color: string; away_color: string;
  played_at: string | null; venue: string | null; status: string; video_name: string | null;
  video_url: string | null; poster_url: string | null; duration: number; width: number; height: number; fps: number;
  is_demo: boolean; created_at: string; notes: string | null; score?: {home: number; away: number};
}
export interface Player {id: string; match_id: string; name: string; number: string | number | null; team: Team; role: string}
export interface Track {excluded?: boolean; id: string; match_id: string; track_id: number; player_id: string | null; team: Team; start_time: number; end_time: number; detection_count: number; confidence: number}
export interface Event {id: string; match_id: string; type: string; time: number; end_time: number | null; team: Team; player_id: string | null; receiver_id: string | null; related_event_id?: string | null; outcome: string; status: string; source: string; confidence: number | null; notes: string | null; x: number | null; y: number | null}
export interface Job {id: string; match_id: string; status: string; progress: number; stage: string; error: string | null; created_at: string; updated_at: string; options: Record<string, unknown>; metrics: Record<string, unknown>}
export interface Calibration {id: string; match_id: string; start_time: number; end_time: number; court_length: number; court_width: number; image_points: number[][]; court_points: number[][]; quality: string; notes: string}
export interface TeamStats {goals: number; passes: number; passes_successful: number; passes_failed: number; passes_unresolved: number; pass_accuracy: number | null; shots: number; shots_on_target: number; assists: number; interceptions: number; turnovers: number; recoveries: number; saves: number; blocks: number; dribbles: number; fouls: number; yellow_cards: number; red_cards: number; possession_seconds: number; possession_pct: number | null}
export interface PlayerStats extends TeamStats, Player {}
export interface Stats {score: {home: number; away: number}; teams: {home: TeamStats; away: TeamStats}; players: PlayerStats[]; pending_count: number; confirmed_count: number; rejected_count: number; coverage: {duration: number; analyzed_seconds: number; ball_visible_seconds: number; identity_seconds: number; calibrated_seconds: number}; pass_network: {from_player_id: string; to_player_id: string; count: number; team: Team}[]; event_timeline: {minute: number; home: number; away: number}[]; warnings: string[]}
export interface Detail {match: Match; players: Player[]; tracks: Track[]; events: Event[]; jobs: Job[]; calibrations: Calibration[]; stats: Stats}
export interface Detection {excluded?: boolean; track_id: number; kind: string; bbox: number[]; confidence: number; team: Team; player_id: string | null}
export interface Health {status: string; ai: {available: boolean; model_ready: boolean; model: string; device: string; message: string}; ffmpeg_available: boolean}
export type Page = 'overview' | 'review' | 'players' | 'reports' | 'calibration';
