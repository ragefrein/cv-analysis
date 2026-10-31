export async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {method, headers: body === undefined ? {} : {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
  if (!response.ok) {
    let message = `Permintaan gagal (${response.status}).`;
    try {const data = await response.json(); message = typeof data.detail === 'string' ? data.detail : Array.isArray(data.detail) ? data.detail.map((v: {msg: string}) => v.msg).join('; ') : data.error || message;} catch { /* response may not be JSON */ }
    throw new Error(message);
  }
  return response.status === 204 ? undefined as T : response.json();
}
export function uploadVideo(matchId: string, file: File, progress: (p: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open('POST', `/api/matches/${matchId}/video`);
    xhr.upload.onprogress = e => {if (e.lengthComputable) progress(Math.round(e.loaded / e.total * 100));};
    xhr.onload = () => {if (xhr.status >= 200 && xhr.status < 300) resolve(); else {let message = 'Video gagal diunggah.'; try {message = JSON.parse(xhr.responseText).detail || message;} catch { /* use default */ } reject(new Error(message));}};
    xhr.onerror = () => reject(new Error('Koneksi terputus saat upload. Silakan ulangi.'));
    const data = new FormData(); data.append('file', file); xhr.send(data);
  });
}
export const clock = (value: number | undefined | null) => {const v = Math.max(0, Math.floor(value || 0)); return `${Math.floor(v / 60).toString().padStart(2, '0')}:${(v % 60).toString().padStart(2, '0')}`;};
export const percent = (v: number | undefined | null) => v == null ? '—' : `${Math.round(v)}%`;
export const eventNames: Record<string, string> = {pass:'Passing',shot:'Tembakan',goal:'Gol',own_goal:'Gol bunuh diri',assist:'Assist',interception:'Interception',turnover:'Turnover',recovery:'Recovery',save:'Save kiper',block:'Block',dribble:'Dribel',foul:'Pelanggaran',yellow_card:'Kartu kuning',red_card:'Kartu merah',substitution:'Pergantian',kick_in:'Kick-in',corner:'Corner',penalty:'Penalti',possession:'Penguasaan bola'};
export const outcomes: Record<string, string> = {successful:'Sukses',failed:'Gagal',unknown:'Belum diketahui',on_target:'On target',off_target:'Off target',blocked:'Diblok',goal:'Gol'};
export const statuses: Record<string, string> = {pending:'Perlu review',confirmed:'Terkonfirmasi',rejected:'Ditolak',queued:'Dalam antrean',running:'Sedang diproses',completed:'Selesai',failed:'Gagal',cancelled:'Dibatalkan',interrupted:'Terhenti',draft:'Draft',ready:'Siap dianalisis',analyzed:'Sudah dianalisis'};
